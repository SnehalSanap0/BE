"""Tests for multi-format audio loading, stereo downmixing and sliding-window segment analysis."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from configs.config import CONFIG
from forensics.segments import analyze_segments, window_bounds
from forensics.verdict import LIKELY_GENUINE, LIKELY_SYNTHETIC
from preprocessing.audio_loader import ffmpeg_available, load_audio, read_audio_info
from preprocessing.transforms import convert_to_mono

SAMPLE_RATE = CONFIG.audio.sample_rate


def _tone(frequency: float, seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    time_axis = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.3 * np.sin(2 * np.pi * frequency * time_axis)).astype(np.float32)


class ConvertToMonoTests(unittest.TestCase):
    def test_soundfile_layout_frames_by_channels(self) -> None:
        stereo = np.stack([_tone(220, 1.0), _tone(220, 1.0)], axis=1)  # (frames, 2)
        mono = convert_to_mono(stereo)
        self.assertEqual(mono.shape, (SAMPLE_RATE,))

    def test_librosa_layout_channels_by_frames(self) -> None:
        stereo = np.stack([_tone(220, 1.0), _tone(220, 1.0)], axis=0)  # (2, frames)
        self.assertEqual(convert_to_mono(stereo).shape, (SAMPLE_RATE,))


class AudioLoaderTests(unittest.TestCase):
    def test_native_formats_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for extension, subtype in ((".wav", "PCM_16"), (".flac", "PCM_16"), (".ogg", "VORBIS")):
                path = Path(temp_dir) / f"clip{extension}"
                sf.write(str(path), _tone(220, 1.0), SAMPLE_RATE, subtype=subtype)
                audio, sample_rate = load_audio(path)
                self.assertEqual(sample_rate, SAMPLE_RATE)
                self.assertAlmostEqual(audio.shape[0] / sample_rate, 1.0, places=1)

    def test_corrupt_file_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.wav"
            path.write_bytes(b"RIFF....garbage")
            with self.assertRaises(ValueError):
                load_audio(path)

    @unittest.skipUnless(ffmpeg_available(), "ffmpeg not on PATH")
    def test_m4a_decoded_via_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            sf.write(str(source), _tone(220, 2.0), SAMPLE_RATE)
            target = Path(temp_dir) / "voice.m4a"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source), "-c:a", "aac", str(target)],
                check=True,
            )
            audio, sample_rate = load_audio(target)
            self.assertEqual(sample_rate, SAMPLE_RATE)
            self.assertAlmostEqual(audio.shape[0] / sample_rate, 2.0, delta=0.1)
            self.assertEqual(read_audio_info(target).container_format, "M4A")

    @unittest.skipUnless(ffmpeg_available(), "ffmpeg not on PATH")
    def test_playlist_disguised_as_m4a_is_not_followed(self) -> None:
        # A classic ffmpeg attack: an HLS playlist renamed to a media extension
        # makes ffmpeg fetch/read whatever the playlist points at.
        with tempfile.TemporaryDirectory() as temp_dir:
            secret = Path(temp_dir) / "secret.wav"
            sf.write(str(secret), _tone(220, 1.0), SAMPLE_RATE)
            playlist = Path(temp_dir) / "evidence.m4a"
            playlist.write_text(
                f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{secret.as_uri()}\n#EXT-X-ENDLIST\n"
            )
            with self.assertRaises(ValueError):
                load_audio(playlist)


class _HighFrequencyScorer:
    """Scores a window by its share of spectral energy above 2 kHz (a stand-in "detector")."""

    def fake_scores(self, processed_windows: list[np.ndarray]) -> np.ndarray:
        scores = []
        for window in processed_windows:
            spectrum = np.abs(np.fft.rfft(window)) ** 2
            frequencies = np.fft.rfftfreq(window.shape[0], 1 / SAMPLE_RATE)
            scores.append(spectrum[frequencies > 2000].sum() / max(spectrum.sum(), 1e-12))
        return np.array(scores)


class SegmentTests(unittest.TestCase):
    def test_window_bounds_cover_the_whole_signal(self) -> None:
        bounds = window_bounds(total_samples=10, window_samples=4, hop_samples=2)
        self.assertEqual(bounds, [(0, 4), (2, 6), (4, 8), (6, 10)])
        self.assertEqual(window_bounds(11, 4, 3), [(0, 4), (3, 7), (6, 10), (7, 11)])
        self.assertEqual(window_bounds(3, 4, 2), [(0, 3)])

    def test_spliced_segment_is_located(self) -> None:
        # 20 s of "genuine" 220 Hz audio with a 2 s "synthetic" 3 kHz splice at 10-12 s.
        audio = np.concatenate([_tone(220, 10), _tone(3000, 2), _tone(220, 8)])
        analysis = analyze_segments(
            audio, _HighFrequencyScorer(), hop_seconds=1.0, min_speech_fraction=0.25,
            genuine_below=0.05, synthetic_above=0.95,
        )
        suspicious = analysis.most_suspicious
        self.assertLessEqual(suspicious.start_seconds, 10.0)
        self.assertGreaterEqual(suspicious.end_seconds, 12.0)
        self.assertNotEqual(suspicious.verdict, LIKELY_GENUINE)
        self.assertEqual(analysis.segments[0].verdict, LIKELY_GENUINE)
        # Windows far from the splice stay genuine.
        far_windows = [s for s in analysis.scored if s.end_seconds <= 9.0 or s.start_seconds >= 13.0]
        self.assertTrue(all(segment.verdict == LIKELY_GENUINE for segment in far_windows))

    def test_window_fully_inside_splice_is_flagged(self) -> None:
        audio = np.concatenate([_tone(220, 8), _tone(3000, 6), _tone(220, 8)])
        analysis = analyze_segments(
            audio, _HighFrequencyScorer(), hop_seconds=1.0, min_speech_fraction=0.25,
            genuine_below=0.05, synthetic_above=0.95,
        )
        self.assertEqual(analysis.most_suspicious.verdict, LIKELY_SYNTHETIC)

    def test_silent_windows_are_skipped(self) -> None:
        audio = np.concatenate([_tone(220, 4), np.zeros(8 * SAMPLE_RATE, dtype=np.float32), _tone(220, 4)])
        analysis = analyze_segments(
            audio, _HighFrequencyScorer(), hop_seconds=2.0, min_speech_fraction=0.25,
            genuine_below=0.05, synthetic_above=0.95,
        )
        skipped = [segment for segment in analysis.segments if segment.fake_score is None]
        self.assertTrue(skipped)
        self.assertTrue(all(segment.speech_fraction < 0.25 for segment in skipped))

    def test_all_silence_still_scores_one_window(self) -> None:
        analysis = analyze_segments(
            np.zeros(8 * SAMPLE_RATE, dtype=np.float32), _HighFrequencyScorer(), hop_seconds=2.0,
            min_speech_fraction=0.25, genuine_below=0.05, synthetic_above=0.95,
        )
        self.assertEqual(len(analysis.scored), 1)


if __name__ == "__main__":
    unittest.main()
