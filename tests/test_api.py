"""HTTP-level tests of the Flask API: input validation, hostile files, and the report endpoint.

A stub model replaces the trained network, so the tests exercise the real upload handling, audio decoding,
window analysis and report generation without needing weights or a GPU.
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

from deployment.app import create_app
from forensics.analyze import analyze_evidence
from forensics.custody_log import CustodyLog
from forensics.evidence import sha256_file
from forensics.verdict import ValidationErrorRates
from preprocessing.audio_loader import ffmpeg_available

SAMPLE_RATE = 16000


def _wav_bytes(seconds: float = 6.0) -> bytes:
    time_axis = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    buffer = io.BytesIO()
    sf.write(buffer, 0.3 * np.sin(2 * np.pi * 220 * time_axis), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class _StubService:
    """Fixed-score stand-in for InferenceService."""

    def __init__(self, checkpoint_path: Path, fake_score: float) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.checkpoint_epoch = 1
        self._fake_score = fake_score

    def fake_scores(self, windows: list[np.ndarray]) -> np.ndarray:
        return np.full(len(windows), self._fake_score)

    def explain_waveform(self, raw_audio, sample_rate, output_stem):  # noqa: ANN001
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        label = "fake" if self._fake_score > 0.5 else "real"
        confidence = self._fake_score if label == "fake" else 1 - self._fake_score
        return SimpleNamespace(
            predicted_label=label, confidence=confidence, processed_audio=np.zeros(4 * SAMPLE_RATE, np.float32),
            processed_sample_rate=SAMPLE_RATE, spectrogram_image=image, overlay_image=image,
            peak_region_caption="1.0-2.0 kHz, 0.50-1.50s",
        )


_RATES = ValidationErrorRates("stub.json", 10, 10, 0.1, 0.0, 0.1, 0.0, {"Kathbath (genuine)": 0.1})


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        checkpoint = root / "best_model.pt"
        checkpoint.write_bytes(b"weights")
        real_analyze = analyze_evidence

        def analyze_into_temp(*args, **kwargs):  # noqa: ANN002, ANN003
            # Reports and the custody log go to a temporary folder, never into outputs/.
            return real_analyze(*args, report_dir=root / "reports", custody_log=CustodyLog(root / "custody.jsonl"),
                                **kwargs)

        self._patches = [
            mock.patch("forensics.analyze.validated_error_rates", lambda service: _RATES),
            mock.patch("deployment.app.analyze_evidence", analyze_into_temp),
        ]
        for patcher in self._patches:
            patcher.start()
        self.client = create_app(_StubService(checkpoint, fake_score=0.99)).test_client()
        self.root = root

    def tearDown(self) -> None:
        for patcher in self._patches:
            patcher.stop()
        self._temp.cleanup()

    def _post(self, url: str, data: dict):
        return self.client.post(url, data=data, content_type="multipart/form-data")

    def test_health(self) -> None:
        self.assertEqual(self.client.get("/health").json, {"status": "ok"})

    def test_predict_returns_verdict_and_segments(self) -> None:
        response = self._post("/predict", {"audio": (io.BytesIO(_wav_bytes()), "clip.wav")})
        self.assertEqual(response.status_code, 200)
        body = response.json
        self.assertEqual(body["verdict"], "LIKELY SYNTHETIC")
        self.assertGreaterEqual(len(body["segments"]), 2)
        self.assertEqual(len(body["most_suspicious_window_seconds"]), 2)

    def test_missing_audio_field_rejected(self) -> None:
        self.assertEqual(self._post("/predict", {}).status_code, 400)

    def test_unsupported_extension_rejected(self) -> None:
        response = self._post("/predict", {"audio": (io.BytesIO(b"x"), "program.exe")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file extension", response.json["error"])

    def test_corrupt_audio_rejected_not_crashed(self) -> None:
        response = self._post("/predict", {"audio": (io.BytesIO(b"RIFF....not really audio"), "bad.wav")})
        self.assertEqual(response.status_code, 400)

    def test_path_traversal_filename_is_sanitised(self) -> None:
        response = self._post("/predict", {"audio": (io.BytesIO(_wav_bytes()), "../../../evil.wav")})
        self.assertEqual(response.status_code, 200)

    @unittest.skipUnless(ffmpeg_available(), "ffmpeg not on PATH")
    def test_playlist_disguised_as_m4a_rejected(self) -> None:
        playlist = b"#EXTM3U\n#EXTINF:1,\nfile:///C:/Windows/win.ini\n"
        response = self._post("/predict", {"audio": (io.BytesIO(playlist), "evidence.m4a")})
        self.assertEqual(response.status_code, 400)

    @unittest.skipUnless(ffmpeg_available(), "ffmpeg not on PATH")
    def test_m4a_upload_accepted(self) -> None:
        wav_path = self.root / "source.wav"
        wav_path.write_bytes(_wav_bytes())
        m4a_path = self.root / "voice.m4a"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(wav_path), "-c:a", "aac", str(m4a_path)],
                       check=True)
        response = self._post("/predict", {"audio": (io.BytesIO(m4a_path.read_bytes()), "voice.m4a")})
        self.assertEqual(response.status_code, 200)

    def test_forensic_report_does_not_require_case_and_examiner(self) -> None:
        response = self._post("/forensic-report", {"audio": (io.BytesIO(_wav_bytes()), "clip.wav")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertEqual(response.data[:4], b"%PDF")
        response.close()

    def test_forensic_report_returns_pdf(self) -> None:
        wav = _wav_bytes()
        response = self._post("/forensic-report", {"audio": (io.BytesIO(wav), "clip.wav"),
                                                    "case_id": "FIR-1/2026", "examiner": "Tester"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertEqual(response.data[:4], b"%PDF")
        response.close()
        self.assertEqual(response.headers["X-Verdict"], "LIKELY SYNTHETIC")
        self.assertIn("X-Report-Id", response.headers)
        entries = [json.loads(line) for line in (self.root / "custody.jsonl").read_text().splitlines()]
        self.assertEqual(entries[-1]["case_id"], "FIR-1/2026")


if __name__ == "__main__":
    unittest.main()
