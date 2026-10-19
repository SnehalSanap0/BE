"""Tests for forensic hashing, verdicts, custody-log tamper detection and report generation."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from forensics.analyze import analyze_evidence
from forensics.custody_log import CustodyLog
from forensics.evidence import describe_evidence, sha256_file
from forensics.verdict import (
    INCONCLUSIVE,
    LIKELY_GENUINE,
    LIKELY_SYNTHETIC,
    ValidationErrorRates,
    classify,
    load_validation_error_rates,
)


def _fake_rates() -> ValidationErrorRates:
    """A made-up validated state, for tests that need a model with a measured error rate."""
    return ValidationErrorRates(
        source_file="stub.json", genuine_count=10, synthetic_count=10, genuine_called_synthetic=0.1,
        synthetic_called_genuine=0.0, genuine_inconclusive=0.1, synthetic_inconclusive=0.0,
        per_source_wrong={"Kathbath (genuine)": 0.1},
    )


#: Patch target making the stub model count as validated.
_VALIDATED = mock.patch("forensics.analyze.validated_error_rates", lambda service: _fake_rates())


def _write_tone(path: Path, seconds: float, sample_rate: int = 16000) -> Path:
    """Write a short sine tone as a 16-bit WAV."""
    time_axis = np.arange(int(seconds * sample_rate)) / sample_rate
    sf.write(str(path), 0.3 * np.sin(2 * np.pi * 220 * time_axis), sample_rate, subtype="PCM_16")
    return path


class _StubService:
    """Stands in for InferenceService: fixed score for every window, no model weights needed."""

    def __init__(self, checkpoint_path: Path, predicted_label: str, confidence: float) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.checkpoint_epoch = 3
        self._predicted_label = predicted_label
        self._confidence = confidence

    @property
    def _fake_score(self) -> float:
        return self._confidence if self._predicted_label == "fake" else 1.0 - self._confidence

    def fake_scores(self, processed_windows: list[np.ndarray]) -> np.ndarray:
        return np.full(len(processed_windows), self._fake_score)

    def explain_waveform(self, raw_audio: np.ndarray, sample_rate: int, output_stem: str) -> SimpleNamespace:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        return SimpleNamespace(
            predicted_label=self._predicted_label,
            confidence=self._confidence,
            processed_audio=np.zeros(4 * 16000, dtype=np.float32),
            processed_sample_rate=16000,
            spectrogram_image=image,
            overlay_image=image,
            peak_region_caption="1.0-2.0 kHz, 0.50-1.50s",
        )


class EvidenceTests(unittest.TestCase):
    def test_sha256_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "blob.bin"
            path.write_bytes(b"evidence" * 1000)
            self.assertEqual(sha256_file(path), hashlib.sha256(b"evidence" * 1000).hexdigest())

    def test_describe_evidence_reads_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_tone(Path(temp_dir) / "call.wav", seconds=2.0, sample_rate=8000)
            info = describe_evidence(path, display_name="original.wav")
            self.assertEqual(info.filename, "original.wav")
            self.assertEqual(info.sample_rate, 8000)
            self.assertAlmostEqual(info.duration_seconds, 2.0, places=2)
            self.assertEqual(info.sha256, sha256_file(path))

    def test_describe_evidence_rejects_non_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fake.wav"
            path.write_text("not audio")
            with self.assertRaises(ValueError):
                describe_evidence(path)


class VerdictTests(unittest.TestCase):
    def test_three_way_bands(self) -> None:
        self.assertEqual(classify(0.01, 0.05, 0.95), LIKELY_GENUINE)
        self.assertEqual(classify(0.05, 0.05, 0.95), INCONCLUSIVE)
        self.assertEqual(classify(0.50, 0.05, 0.95), INCONCLUSIVE)
        self.assertEqual(classify(0.95, 0.05, 0.95), INCONCLUSIVE)
        self.assertEqual(classify(0.99, 0.05, 0.95), LIKELY_SYNTHETIC)

    def _write_scores(self, path: Path, checkpoint: str) -> Path:
        predictions = [
            {"source_tag": "real", "filepath": "a.wav", "true_label": "real", "fake_score": 0.99},
            {"source_tag": "real", "filepath": "b.wav", "true_label": "real", "fake_score": 0.01},
            {"source_tag": "real_cv_m", "filepath": "c.wav", "true_label": "real", "fake_score": 0.50},
            {"source_tag": "fake", "filepath": "d.wav", "true_label": "fake", "fake_score": 0.99},
        ]
        # The second run repeats a.wav: it must be counted once.
        payload = {
            "checkpoint": checkpoint,
            "runs": [{"predictions": predictions}, {"predictions": predictions[:1]}],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_error_rates_deduplicate_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scores = self._write_scores(root / "scores.json", "ckpt/best_model.pt")
            rates = load_validation_error_rates(
                scores, 0.05, 0.95, expected_checkpoint=root / "ckpt" / "best_model.pt", project_root=root
            )
            self.assertIsNotNone(rates)
            self.assertEqual(rates.genuine_count, 3)
            self.assertEqual(rates.synthetic_count, 1)
            self.assertAlmostEqual(rates.genuine_called_synthetic, 1 / 3)
            self.assertAlmostEqual(rates.genuine_inconclusive, 1 / 3)
            self.assertEqual(rates.synthetic_called_genuine, 0.0)
            self.assertAlmostEqual(rates.per_source_wrong["Kathbath (genuine)"], 0.5)

    def test_error_rates_withheld_for_other_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scores = self._write_scores(root / "scores.json", "ckpt/best_model.pt")
            rates = load_validation_error_rates(
                scores, 0.05, 0.95, expected_checkpoint=root / "other" / "best_model.pt", project_root=root
            )
            self.assertIsNone(rates)

    def test_missing_scores_file(self) -> None:
        self.assertIsNone(load_validation_error_rates(Path("does/not/exist.json"), 0.05, 0.95))


class CustodyLogTests(unittest.TestCase):
    def _log_with_entries(self, directory: Path, count: int = 3) -> CustodyLog:
        log = CustodyLog(directory / "custody.jsonl")
        for index in range(count):
            log.append("analysis", {"case_id": f"CASE-{index}", "verdict": LIKELY_GENUINE})
        return log

    def test_intact_chain_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log = self._log_with_entries(Path(temp_dir))
            result = log.verify()
            self.assertTrue(result.is_valid)
            self.assertEqual(result.entry_count, 3)

    def test_edited_entry_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log = self._log_with_entries(Path(temp_dir))
            lines = log.log_path.read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1].replace(LIKELY_GENUINE, LIKELY_SYNTHETIC)
            log.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = log.verify()
            self.assertFalse(result.is_valid)
            self.assertEqual(result.first_bad_line, 2)

    def test_deleted_entry_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log = self._log_with_entries(Path(temp_dir))
            lines = log.log_path.read_text(encoding="utf-8").splitlines()
            del lines[1]
            log.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertFalse(log.verify().is_valid)


class AnalyzeEvidenceTests(unittest.TestCase):
    @_VALIDATED
    def test_end_to_end_report_and_custody_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best_model.pt"
            checkpoint.write_bytes(b"weights")
            evidence = _write_tone(root / "evidence.wav", seconds=10.0, sample_rate=8000)
            original_hash = sha256_file(evidence)
            log = CustodyLog(root / "custody.jsonl")

            analysis = analyze_evidence(
                evidence,
                case_id="FIR 12/2026",
                examiner="Test Examiner",
                service=_StubService(checkpoint, predicted_label="fake", confidence=0.99),
                model_label="stub model",
                report_dir=root / "reports",
                custody_log=log,
            )

            findings = analysis.findings
            self.assertEqual(findings.verdict, LIKELY_SYNTHETIC)
            self.assertAlmostEqual(findings.fake_score, 0.99)
            self.assertTrue(findings.integrity_verified)
            self.assertEqual(sha256_file(evidence), original_hash)
            # 10 s at 8 kHz -> windows 0-4, 2-6, 4-8, 6-10: the whole file is covered.
            self.assertEqual(len(findings.segment_analysis.scored), 4)
            self.assertEqual(findings.segment_analysis.segments[-1].end_seconds, 10.0)
            # Many-windows caveat and low-sample-rate caveat.
            self.assertEqual(len(findings.warnings), 2)
            self.assertTrue(analysis.pdf_path.name.startswith("FIR_12_2026_"))
            self.assertEqual(analysis.pdf_path.read_bytes()[:4], b"%PDF")

            report = json.loads(analysis.json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["evidence"]["filename"], "evidence.wav")
            self.assertIsNotNone(report["validation_error_rates"])
            self.assertTrue(report["result"]["validated"])
            self.assertEqual(len(report["segment_analysis"]["segments"]), 4)

            entry = analysis.custody_entry
            self.assertEqual(entry["evidence_filename"], "evidence.wav")
            self.assertEqual(entry["verdict"], LIKELY_SYNTHETIC)
            self.assertTrue(log.verify().is_valid)

    @_VALIDATED
    def test_long_recording_lists_every_flagged_window(self) -> None:
        # 200 s at hop 2 s -> 99 windows, all flagged: the list must paginate, not clip or truncate.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best_model.pt"
            checkpoint.write_bytes(b"weights")
            evidence = _write_tone(root / "long.wav", seconds=200.0)
            analysis = analyze_evidence(
                evidence,
                case_id="LONG",
                examiner="Test Examiner",
                service=_StubService(checkpoint, predicted_label="fake", confidence=0.99),
                model_label="stub model",
                report_dir=root / "reports",
                custody_log=CustodyLog(root / "custody.jsonl"),
            )
            flagged = [s for s in analysis.findings.segment_analysis.scored if s.verdict != LIKELY_GENUINE]
            self.assertEqual(len(flagged), 99)
            page_count = len(re.findall(rb"/Type\s*/Page[^s]", analysis.pdf_path.read_bytes()))
            # >= 1 text page + 1 figure page + 3 list pages (43 rows per page).
            self.assertGreaterEqual(page_count, 5)

    def test_unvalidated_model_never_issues_a_decisive_verdict(self) -> None:
        # No _VALIDATED patch: the stub's checkpoint has no held-out validation scores.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best_model.pt"
            checkpoint.write_bytes(b"weights")
            evidence = _write_tone(root / "evidence.wav", seconds=6.0)
            analysis = analyze_evidence(
                evidence,
                case_id="UNVALIDATED",
                examiner="Test Examiner",
                service=_StubService(checkpoint, predicted_label="fake", confidence=0.999),
                model_label="stub model",
                report_dir=root / "reports",
                custody_log=CustodyLog(root / "custody.jsonl"),
            )
            findings = analysis.findings
            self.assertFalse(findings.validated)
            self.assertEqual(findings.verdict, INCONCLUSIVE)
            self.assertTrue(all(s.verdict == INCONCLUSIVE for s in findings.segment_analysis.scored))
            self.assertAlmostEqual(findings.fake_score, 0.999)  # the score itself is still reported
            self.assertTrue(any("MODEL NOT VALIDATED" in w for w in findings.warnings))
            self.assertEqual(analysis.custody_entry["verdict"], INCONCLUSIVE)

    def test_blank_case_id_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best_model.pt"
            checkpoint.write_bytes(b"weights")
            evidence = _write_tone(root / "evidence.wav", seconds=1.0)
            analysis = analyze_evidence(
                evidence,
                case_id="  ",
                examiner="Test Examiner",
                service=_StubService(checkpoint, "real", 0.99),
                model_label="stub model",
                report_dir=root / "reports",
                custody_log=CustodyLog(root / "custody.jsonl"),
            )
            self.assertIsNotNone(analysis)
            self.assertTrue(analysis.pdf_path.exists())


if __name__ == "__main__":
    unittest.main()
