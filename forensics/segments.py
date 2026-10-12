"""Sliding-window analysis of a whole recording.

The model was trained on single 4-second clips, so a long recording (a phone
call, a voice note) is scored as overlapping windows of that same length.
This finds a synthetic sentence spliced into the middle of genuine audio,
which scoring only the first window would miss. Windows that are mostly
silence are skipped: the model never saw near-silent clips in training, so
their scores carry no meaning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import librosa
import numpy as np

from configs.config import CONFIG
from forensics.verdict import classify
from preprocessing.transforms import convert_to_mono, preprocess_audio, resample_audio


class WindowScorer(Protocol):
    """Anything that can score preprocessed windows (an InferenceService, or a test stub)."""

    def fake_scores(self, processed_windows: list[np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True)
class Segment:
    """One analysis window.

    Attributes:
        start_seconds: Window start, from the beginning of the recording.
        end_seconds: Window end.
        speech_fraction: Fraction of the window detected as non-silent.
        fake_score: Probability synthetic, or ``None`` if skipped as silence.
        verdict: Three-way verdict for this window, or ``None`` if skipped.
    """

    start_seconds: float
    end_seconds: float
    speech_fraction: float
    fake_score: float | None
    verdict: str | None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of every field."""
        return asdict(self)


@dataclass(frozen=True)
class SegmentAnalysis:
    """All windows of one recording.

    Attributes:
        segments: Every window, in time order, scored or skipped.
        window_seconds: Window length (the model's training clip length).
        hop_seconds: Step between window starts.
    """

    segments: tuple[Segment, ...]
    window_seconds: float
    hop_seconds: float

    @property
    def scored(self) -> tuple[Segment, ...]:
        """Windows that were scored (not skipped as silence)."""
        return tuple(segment for segment in self.segments if segment.fake_score is not None)

    @property
    def most_suspicious(self) -> Segment:
        """The scored window with the highest fake score."""
        return max(self.scored, key=lambda segment: segment.fake_score)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the analysis."""
        return {
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def window_bounds(total_samples: int, window_samples: int, hop_samples: int) -> list[tuple[int, int]]:
    """Start/end sample indices of overlapping windows covering the whole signal.

    The last window is aligned to the end of the signal so no tail is left
    unscored. A signal shorter than one window yields a single window.

    Args:
        total_samples: Signal length.
        window_samples: Window length.
        hop_samples: Step between window starts.

    Returns:
        ``(start, end)`` pairs in time order.
    """
    if total_samples <= window_samples:
        return [(0, total_samples)]
    starts = list(range(0, total_samples - window_samples + 1, hop_samples))
    if starts[-1] + window_samples < total_samples:
        starts.append(total_samples - window_samples)
    return [(start, start + window_samples) for start in starts]


#: Peak amplitude below which a whole recording is treated as silent (about -80 dBFS).
_ABSOLUTE_SILENCE_PEAK = 1e-4


def _speech_mask(audio: np.ndarray, top_db: float) -> np.ndarray:
    """Boolean per-sample mask of non-silent regions (``librosa.effects.split``).

    ``librosa.effects.split`` measures silence relative to the loudest frame,
    so pure digital silence would otherwise count as all "speech".
    """
    mask = np.zeros(audio.shape[0], dtype=bool)
    if audio.size == 0 or np.max(np.abs(audio)) < _ABSOLUTE_SILENCE_PEAK:
        return mask
    for start, end in librosa.effects.split(audio, top_db=top_db):
        mask[start:end] = True
    return mask


def to_model_rate_mono(raw_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert raw samples to mono at ``CONFIG.audio.sample_rate`` (no trimming or padding)."""
    return resample_audio(convert_to_mono(raw_audio), sample_rate)


def analyze_segments(
    audio: np.ndarray,
    scorer: WindowScorer,
    hop_seconds: float,
    min_speech_fraction: float,
    genuine_below: float,
    synthetic_above: float,
) -> SegmentAnalysis:
    """Score a recording in overlapping model-length windows.

    Args:
        audio: Mono waveform at ``CONFIG.audio.sample_rate`` (see :func:`to_model_rate_mono`).
        scorer: Scores preprocessed windows (normally an ``InferenceService``).
        hop_seconds: Step between window starts.
        min_speech_fraction: Windows with less non-silent audio than this are skipped.
        genuine_below: Lower verdict threshold.
        synthetic_above: Upper verdict threshold.

    Returns:
        A :class:`SegmentAnalysis` with at least one scored window -- if every
        window is mostly silent, the one with the most speech is scored anyway.
    """
    sample_rate = CONFIG.audio.sample_rate
    window_samples = CONFIG.audio.target_length_samples
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))

    bounds = window_bounds(audio.shape[0], window_samples, hop_samples)
    speech = _speech_mask(audio, top_db=CONFIG.audio.top_db)
    fractions = [float(speech[start:end].mean()) if end > start else 0.0 for start, end in bounds]

    to_score = [index for index, fraction in enumerate(fractions) if fraction >= min_speech_fraction]
    if not to_score:
        to_score = [int(np.argmax(fractions))]

    processed = [preprocess_audio(audio[bounds[index][0] : bounds[index][1]], sample_rate) for index in to_score]
    scores = dict(zip(to_score, (float(score) for score in scorer.fake_scores(processed))))

    segments = []
    for index, (start, end) in enumerate(bounds):
        score = scores.get(index)
        segments.append(
            Segment(
                start_seconds=start / sample_rate,
                end_seconds=end / sample_rate,
                speech_fraction=fractions[index],
                fake_score=score,
                verdict=classify(score, genuine_below, synthetic_above) if score is not None else None,
            )
        )
    return SegmentAnalysis(
        segments=tuple(segments),
        window_seconds=window_samples / sample_rate,
        hop_seconds=hop_samples / sample_rate,
    )
