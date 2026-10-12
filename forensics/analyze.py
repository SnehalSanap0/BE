"""Run one forensic analysis end to end: hash, predict, verdict, report, custody log.

Usage:
    python -m forensics.analyze --audio evidence.wav --case-id FIR-123/2026 --examiner "A. Examiner"
    python -m forensics.analyze --verify-log

The evidence file is only ever read. Its SHA-256 is taken before and after
the model runs; a mismatch is reported as an integrity failure in the report.
The whole recording is scored in overlapping windows (see
:mod:`forensics.segments`); the verdict follows the most suspicious window,
which is also the one Grad-CAM explains.
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from configs.config import CONFIG, ROOT_DIR
from deployment.inference_service import InferenceService
from explainability.visualizer import render_colormap_legend
from forensics.custody_log import CustodyLog
from forensics.evidence import describe_evidence, sha256_file
from forensics.report import ForensicFindings, write_json_report, write_pdf_report
from explainability.explain import ExplanationResult
from forensics.segments import SegmentAnalysis, analyze_segments, to_model_rate_mono
from forensics.verdict import LIKELY_SYNTHETIC, ValidationErrorRates, classify, load_validation_error_rates
from preprocessing.audio_loader import load_audio
from preprocessing.validator import AudioValidator
from utils.logger import get_logger

logger = get_logger(__name__)

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ForensicAnalysis:
    """Outcome of :func:`analyze_evidence`.

    Attributes:
        findings: Everything the report states.
        pdf_path: Written PDF report.
        json_path: Written JSON report.
        custody_entry: The custody-log entry recorded for this analysis.
    """

    findings: ForensicFindings
    pdf_path: Path
    json_path: Path
    custody_entry: dict


def _safe_filename_part(text: str) -> str:
    """Reduce a case ID to characters safe for a file name (e.g. ``FIR 12/2026`` -> ``FIR_12_2026``)."""
    return _UNSAFE_FILENAME_CHARS.sub("_", text).strip("_") or "case"


def _display_path(path: Path) -> str:
    """Show a path relative to the project root when possible (reports should not leak local directories)."""
    try:
        return Path(path).resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return Path(path).name


#: Thresholds that leave no decisive band: every score is INCONCLUSIVE.
_NO_DECISIVE_BAND = (0.0, 1.0)


def validated_error_rates(service: InferenceService) -> ValidationErrorRates | None:
    """Known error rates for ``service``'s checkpoint at the configured thresholds, if measured.

    Args:
        service: Loaded model.

    Returns:
        The rates, or ``None`` when no held-out validation scores exist for this
        exact checkpoint (the model is then *not validated*).
    """
    return load_validation_error_rates(
        CONFIG.forensics.validation_scores,
        CONFIG.forensics.genuine_below,
        CONFIG.forensics.synthetic_above,
        expected_checkpoint=service.checkpoint_path,
        project_root=ROOT_DIR,
    )


def decision_thresholds(error_rates: ValidationErrorRates | None) -> tuple[float, float]:
    """Verdict thresholds to use: the configured ones for a validated model, none for an unvalidated one.

    A decisive verdict from a model whose error rate has never been measured
    would be a claim the project cannot back, so such a model only ever
    reports INCONCLUSIVE (its scores are still shown).

    Args:
        error_rates: Result of :func:`validated_error_rates`.

    Returns:
        ``(genuine_below, synthetic_above)``.
    """
    if error_rates is None:
        return _NO_DECISIVE_BAND
    return CONFIG.forensics.genuine_below, CONFIG.forensics.synthetic_above


def analyze_recording(
    audio_path: Path,
    service: InferenceService,
    display_name: str | None = None,
    verdict_thresholds: tuple[float, float] | None = None,
) -> tuple[SegmentAnalysis, ExplanationResult]:
    """Score a whole recording in windows and Grad-CAM-explain its most suspicious window.

    Args:
        audio_path: Audio file on disk, any supported format.
        service: Loaded model.
        display_name: Name used in error messages and the overlay file name.
        verdict_thresholds: ``(genuine_below, synthetic_above)`` for per-window
            verdicts. Defaults to :func:`decision_thresholds` for ``service``,
            i.e. no decisive verdicts if the model is not validated.

    Returns:
        ``(segment_analysis, explanation_of_most_suspicious_window)``.

    Raises:
        ValueError: If the audio fails validation or cannot be decoded.
    """
    audio_path = Path(audio_path)
    display_name = display_name or audio_path.name
    validation = AudioValidator().validate(audio_path)
    if not validation.is_valid:
        raise ValueError(f"Invalid audio file '{display_name}': {validation.reason}")

    thresholds = CONFIG.forensics
    genuine_below, synthetic_above = verdict_thresholds or decision_thresholds(validated_error_rates(service))
    raw_audio, sample_rate = load_audio(audio_path)
    audio = to_model_rate_mono(raw_audio, sample_rate)
    segment_analysis = analyze_segments(
        audio,
        scorer=service,
        hop_seconds=thresholds.segment_hop_seconds,
        min_speech_fraction=thresholds.min_speech_fraction,
        genuine_below=genuine_below,
        synthetic_above=synthetic_above,
    )
    suspicious = segment_analysis.most_suspicious
    start_sample = int(round(suspicious.start_seconds * CONFIG.audio.sample_rate))
    end_sample = int(round(suspicious.end_seconds * CONFIG.audio.sample_rate))
    explanation = service.explain_waveform(
        audio[start_sample:end_sample], CONFIG.audio.sample_rate, output_stem=Path(display_name).stem
    )
    return segment_analysis, explanation


def analyze_evidence(
    audio_path: Path,
    case_id: str,
    examiner: str,
    service: InferenceService,
    model_label: str,
    original_filename: str | None = None,
    report_dir: Path | None = None,
    custody_log: CustodyLog | None = None,
) -> ForensicAnalysis:
    """Analyse one evidence file and write its PDF/JSON reports and custody-log entry.

    Args:
        audio_path: Evidence audio file on disk (read only).
        case_id: Case / FIR / reference number. Required, non-empty.
        examiner: Name of the person running the analysis. Required, non-empty.
        service: Loaded model to analyse with.
        model_label: Human-readable description of ``service``'s model for the report.
        original_filename: Name to record for the evidence when ``audio_path``
            is a temporary copy (e.g. a web upload).
        report_dir: Where reports are written. Defaults to ``CONFIG.forensics.report_dir``.
        custody_log: Log to append to. Defaults to ``CONFIG.forensics.custody_log``.

    Returns:
        A :class:`ForensicAnalysis`.

    Raises:
        ValueError: If ``case_id``/``examiner`` is blank or the audio is invalid.
    """
    case_id = case_id.strip()
    examiner = examiner.strip()
    if not case_id or not examiner:
        raise ValueError("Both a case ID and an examiner name are required for a forensic report.")

    report_dir = report_dir or CONFIG.forensics.report_dir
    custody_log = custody_log or CustodyLog(CONFIG.forensics.custody_log)
    thresholds = CONFIG.forensics

    evidence = describe_evidence(audio_path, display_name=original_filename)
    error_rates = validated_error_rates(service)
    genuine_below, synthetic_above = decision_thresholds(error_rates)
    segment_analysis, result = analyze_recording(
        audio_path, service, display_name=evidence.filename, verdict_thresholds=(genuine_below, synthetic_above)
    )
    suspicious = segment_analysis.most_suspicious
    evidence_sha256_after = sha256_file(audio_path)

    # The recording's score is its most suspicious window: one spliced synthetic sentence is enough.
    fake_score = suspicious.fake_score
    verdict = classify(fake_score, genuine_below, synthetic_above)
    analysed_seconds = segment_analysis.window_seconds

    warnings: list[str] = []
    if error_rates is None:
        warnings.append(
            "MODEL NOT VALIDATED: no held-out error rate has been measured for this model, so no decisive "
            "verdict is issued. The scores are shown for information only and must not be used as evidence."
        )
    scored_count = len(segment_analysis.scored)
    skipped_count = len(segment_analysis.segments) - scored_count
    if scored_count > 1:
        warnings.append(
            f"The verdict is the highest of {scored_count} window scores. The error rates above were measured "
            "on single clips, so the chance of a false alarm somewhere in a long genuine recording is higher. "
            "Review the flagged windows individually."
        )
    if skipped_count:
        warnings.append(f"{skipped_count} mostly silent window(s) were skipped and not scored.")
    if suspicious.speech_fraction < thresholds.min_speech_fraction:
        warnings.append(
            "The recording contains very little detectable speech; the score comes from a mostly silent window "
            "and is unreliable."
        )
    if evidence.is_lossy:
        warnings.append(
            f"The evidence uses lossy compression ({evidence.container_format}/{evidence.encoding}). "
            "The model was trained on uncompressed/lossless audio; in project tests, genuine speech "
            "re-encoded as low-bitrate Opus (WhatsApp-style) was scored as synthetic throughout. "
            "A LIKELY SYNTHETIC verdict on compressed audio must not be relied upon without expert review."
        )
    if evidence.sample_rate < CONFIG.audio.sample_rate:
        warnings.append(
            f"Evidence sample rate ({evidence.sample_rate} Hz) is below the model's {CONFIG.audio.sample_rate} Hz "
            "training rate (e.g. telephone audio). The model was not validated on such audio."
        )
    if evidence_sha256_after != evidence.sha256:
        warnings.append("Evidence hash changed during analysis. The result must not be relied upon.")

    generated_at = datetime.now(timezone.utc)
    report_id = f"{generated_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    findings = ForensicFindings(
        report_id=report_id,
        generated_at_utc=generated_at.isoformat(timespec="seconds"),
        case_id=case_id,
        examiner=examiner,
        evidence=evidence,
        evidence_sha256_after=evidence_sha256_after,
        model_label=model_label,
        checkpoint_path=_display_path(service.checkpoint_path),
        checkpoint_sha256=service.checkpoint_sha256,
        checkpoint_epoch=service.checkpoint_epoch,
        fake_score=fake_score,
        verdict=verdict,
        genuine_below=genuine_below,
        synthetic_above=synthetic_above,
        validated=error_rates is not None,
        analysed_seconds=analysed_seconds,
        peak_region_caption=result.peak_region_caption,
        segment_analysis=segment_analysis,
        error_rates=error_rates,
        warnings=tuple(warnings),
        spectrogram_image=result.spectrogram_image,
        overlay_image=result.overlay_image,
        legend_image=render_colormap_legend(),
    )

    stem = f"{_safe_filename_part(case_id)}_{report_id}"
    pdf_path = write_pdf_report(findings, report_dir / f"{stem}.pdf")
    json_path = write_json_report(findings, report_dir / f"{stem}.json")

    custody_entry = custody_log.append(
        "analysis",
        {
            "case_id": case_id,
            "examiner": examiner,
            "report_id": report_id,
            "evidence_filename": evidence.filename,
            "evidence_sha256": evidence.sha256,
            "integrity_verified": findings.integrity_verified,
            "checkpoint_sha256": service.checkpoint_sha256,
            "verdict": verdict,
            "fake_score": round(fake_score, 6),
            "windows_scored": scored_count,
            "windows_flagged_synthetic": sum(
                segment.verdict == LIKELY_SYNTHETIC for segment in segment_analysis.scored
            ),
            "report_pdf_sha256": sha256_file(pdf_path),
            "report_json_sha256": sha256_file(json_path),
        },
    )
    logger.info("Forensic analysis %s for case '%s': %s (score %.4f).", report_id, case_id, verdict, fake_score)
    return ForensicAnalysis(findings=findings, pdf_path=pdf_path, json_path=json_path, custody_entry=custody_entry)


def main() -> None:
    """CLI entrypoint: analyse one file, or verify the custody log."""
    parser = argparse.ArgumentParser(description="Forensic deepfake analysis with PDF report and custody log.")
    parser.add_argument("--audio", type=Path, help="Evidence audio file.")
    parser.add_argument("--case-id", help="Case / FIR / reference number.")
    parser.add_argument("--examiner", help="Name of the examiner running the analysis.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to use (default: deployment.model_checkpoint).")
    parser.add_argument("--verify-log", action="store_true", help="Verify the custody log's hash chain and exit.")
    args = parser.parse_args()

    if args.verify_log:
        verification = CustodyLog(CONFIG.forensics.custody_log).verify()
        if verification.is_valid:
            print(f"Custody log OK: {verification.entry_count} entries, hash chain intact.")
            return
        print(f"Custody log TAMPERED at line {verification.first_bad_line}: {verification.reason}")
        sys.exit(1)

    if not (args.audio and args.case_id and args.examiner):
        parser.error("--audio, --case-id and --examiner are required unless --verify-log is given.")

    service = InferenceService(checkpoint_path=args.checkpoint)
    analysis = analyze_evidence(
        args.audio,
        case_id=args.case_id,
        examiner=args.examiner,
        service=service,
        model_label="EfficientNet-B0 deepfake detector (mel-spectrogram input)",
    )
    findings = analysis.findings
    print(f"Verdict: {findings.verdict} (probability synthetic {findings.fake_score:.4f})")
    print(f"Evidence SHA-256: {findings.evidence.sha256} (integrity {'OK' if findings.integrity_verified else 'FAILED'})")
    for warning in findings.warnings:
        print(f"Warning: {warning}")
    print(f"PDF report:  {analysis.pdf_path}")
    print(f"JSON report: {analysis.json_path}")


if __name__ == "__main__":
    main()
