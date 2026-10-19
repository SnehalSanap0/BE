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

import numpy as np

from configs.config import CONFIG, ROOT_DIR
from deployment.inference_service import InferenceService
from explainability.visualizer import render_colormap_legend
from forensics.custody_log import CustodyLog
from forensics.evidence import describe_evidence, sha256_file
from forensics.report import ForensicFindings, write_json_report, write_pdf_report
from explainability.explain import ExplanationResult
from forensics.segments import SegmentAnalysis, analyze_segments, to_model_rate_mono
from forensics.verdict import INCONCLUSIVE, LIKELY_GENUINE, LIKELY_SYNTHETIC, ValidationErrorRates, classify, load_validation_error_rates
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
    scores_path = CONFIG.forensics.validation_scores
    if "english" in str(service.checkpoint_path).lower():
        english_scores = ROOT_DIR / "outputs" / "english" / "cross_source_scores.json"
        if english_scores.exists():
            scores_path = english_scores

    return load_validation_error_rates(
        scores_path,
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


def aggregate_recording_verdict(
    segment_analysis: SegmentAnalysis,
    genuine_below: float,
    synthetic_above: float,
) -> tuple[float, str, list[str]]:
    """Aggregate window scores into a robust recording-level score and verdict.

    Prevents single-window false alarms (e.g. mic clicks, breath pops, or trailing silence)
    from declaring an otherwise authentic recording 99.99% synthetic, while reliably
    catching contiguous spliced synthetic utterances.

    Args:
        segment_analysis: Evaluated windows across the recording.
        genuine_below: Score threshold below which a window is likely genuine.
        synthetic_above: Score threshold above which a window is likely synthetic.

    Returns:
        A tuple of (fake_score, verdict, aggregation_warnings).
    """
    scored = segment_analysis.scored
    if not scored:
        suspicious = segment_analysis.most_suspicious
        score = suspicious.fake_score if suspicious.fake_score is not None else 0.5
        return score, classify(score, genuine_below, synthetic_above), []

    if len(scored) == 1:
        score = scored[0].fake_score if scored[0].fake_score is not None else 0.5
        return score, classify(score, genuine_below, synthetic_above), []

    scores = [s.fake_score for s in scored if s.fake_score is not None]
    if not scores:
        return 0.5, INCONCLUSIVE, []

    synthetic_segments = [s for s in scored if s.verdict == LIKELY_SYNTHETIC]
    genuine_segments = [s for s in scored if s.verdict == LIKELY_GENUINE]
    total_scored = len(scores)
    synth_ratio = len(synthetic_segments) / total_scored
    genuine_ratio = len(genuine_segments) / total_scored

    # Check for consecutive flagged synthetic windows (contiguous spliced utterance)
    consecutive_synthetic = False
    for i in range(len(scored) - 1):
        if (
            scored[i].verdict == LIKELY_SYNTHETIC
            and scored[i + 1].verdict == LIKELY_SYNTHETIC
            and abs(scored[i + 1].start_seconds - scored[i].end_seconds) <= (segment_analysis.hop_seconds * 1.5)
        ):
            consecutive_synthetic = True
            break

    warnings: list[str] = []

    # Case 1: Decisive Synthetic Audio (Dominant Synthetic Evidence)
    # The recording is predominantly synthetic (>= 40% synthetic AND synthetic outnumbers genuine),
    # or all scored windows are synthetic.
    if synth_ratio >= 0.40 and len(synthetic_segments) >= len(genuine_segments):
        fake_score = max(s.fake_score for s in synthetic_segments)
        verdict = LIKELY_SYNTHETIC
        if synth_ratio < 1.0:
            warnings.append(
                f"Predominantly synthetic speech detected: {len(synthetic_segments)} of {total_scored} "
                f"window(s) ({synth_ratio:.1%}) flagged synthetic."
            )
        return fake_score, verdict, warnings

    # Case 2: Spliced or Localized Suspicious Segment in Mixed/Genuine Recording
    # A localized contiguous cluster exists, but genuine speech outnumbers synthetic windows.
    if consecutive_synthetic and len(genuine_segments) >= len(synthetic_segments):
        fake_score = float(np.median(scores))
        verdict = INCONCLUSIVE
        first_s = synthetic_segments[0].start_seconds
        last_s = synthetic_segments[-1].end_seconds
        warnings.append(
            f"Localized suspicious segment detected ({len(synthetic_segments)} window(s) flagged synthetic, "
            f"e.g. {first_s:.1f}s–{last_s:.1f}s), but {len(genuine_segments)} window(s) ({genuine_ratio:.1%}) "
            "are decisively genuine. The file is not uniformly synthetic; verdict held as INCONCLUSIVE. "
            "Review flagged regions on the timeline for potential splicing."
        )
        return fake_score, verdict, warnings

    # Case 3: Predominantly Genuine Speech with Isolated Artifacts / Transients
    # Genuine windows outnumber synthetic windows, and synthetic spikes are isolated (no contiguous cluster).
    if genuine_ratio >= 0.35 and len(genuine_segments) > len(synthetic_segments):
        fake_score = float(np.median(scores))
        verdict = LIKELY_GENUINE if fake_score < genuine_below else INCONCLUSIVE
        if synthetic_segments:
            warnings.append(
                f"{len(synthetic_segments)} isolated window(s) ({synth_ratio:.1%}) triggered high synthetic scores, "
                f"but {len(genuine_segments)} window(s) ({genuine_ratio:.1%}) are decisively genuine without "
                "contiguous synthetic cluster support. Isolated spikes typically stem from mic gating, breath "
                "pops, or trailing silence; overall consensus is predominantly genuine."
            )
        return fake_score, verdict, warnings

    # Case 4: Ambiguous / Inconclusive
    fake_score = float(np.median(scores))
    verdict = classify(fake_score, genuine_below, synthetic_above)
    if synthetic_segments:
        warnings.append(
            f"{len(synthetic_segments)} of {total_scored} window(s) flagged synthetic. "
            "Overall consensus is held as inconclusive; inspect timeline for individual windows."
        )
    return fake_score, verdict, warnings


def analyze_evidence(
    audio_path: Path,
    case_id: str = "",
    examiner: str = "",
    service: InferenceService | None = None,
    model_label: str = "EfficientNet-B0 deepfake detector",
    original_filename: str | None = None,
    report_dir: Path | None = None,
    custody_log: CustodyLog | None = None,
) -> ForensicAnalysis:
    """Analyse one audio file and write its PDF/JSON reports and log entry.

    Args:
        audio_path: Audio file on disk.
        case_id: Optional reference / case identifier.
        examiner: Optional name of the person or system running the analysis.
        service: Loaded model to analyse with (defaults to a fresh InferenceService).
        model_label: Human-readable description of the model for the report.
        original_filename: Name to record for the audio when ``audio_path``
            is a temporary copy (e.g. a web upload).
        report_dir: Where reports are written. Defaults to ``CONFIG.forensics.report_dir``.
        custody_log: Log to append to. Defaults to ``CONFIG.forensics.custody_log``.

    Returns:
        A :class:`ForensicAnalysis`.

    Raises:
        ValueError: If the audio is invalid.
    """
    case_id = (case_id or "").strip()
    examiner = (examiner or "").strip()

    service = service or InferenceService()
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

    # Compute recording-level verdict and consensus score
    fake_score, verdict, agg_warnings = aggregate_recording_verdict(
        segment_analysis, genuine_below, synthetic_above
    )
    analysed_seconds = segment_analysis.window_seconds

    warnings: list[str] = []
    if error_rates is None:
        warnings.append(
            "MODEL NOT VALIDATED: no held-out error rate has been measured for this model, so no decisive "
            "verdict is issued. The scores are shown for information only."
        )
    warnings.extend(agg_warnings)
    scored_count = len(segment_analysis.scored)
    skipped_count = len(segment_analysis.segments) - scored_count
    if scored_count > 1 and not agg_warnings:
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
            f"The audio uses lossy compression ({evidence.container_format}/{evidence.encoding}). "
            "The model was trained on uncompressed/lossless audio; in project tests, genuine speech "
            "re-encoded as low-bitrate Opus (WhatsApp-style) was scored as synthetic throughout. "
            "A LIKELY SYNTHETIC verdict on compressed audio must not be relied upon without careful review."
        )
    if evidence.sample_rate < CONFIG.audio.sample_rate:
        warnings.append(
            f"Audio sample rate ({evidence.sample_rate} Hz) is below the model's {CONFIG.audio.sample_rate} Hz "
            "training rate (e.g. telephone audio). The model was not validated on such audio."
        )

    generated_at = datetime.now(timezone.utc)
    report_id = f"{generated_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    findings = ForensicFindings(
        report_id=report_id,
        generated_at_utc=generated_at.isoformat(timespec="seconds"),
        case_id=case_id,
        examiner=examiner,
        evidence=evidence,
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

    audio_stem = Path(original_filename or audio_path).stem
    stem_prefix = _safe_filename_part(case_id) if case_id else _safe_filename_part(audio_stem)
    stem = f"{stem_prefix}_{report_id}"
    pdf_path = write_pdf_report(findings, report_dir / f"{stem}.pdf")
    json_path = write_json_report(findings, report_dir / f"{stem}.json")

    custody_entry = custody_log.append(
        "analysis",
        {
            "case_id": case_id,
            "examiner": examiner,
            "report_id": report_id,
            "evidence_filename": evidence.filename,
            "verdict": verdict,
            "fake_score": round(fake_score, 6),
            "windows_scored": scored_count,
            "windows_flagged_synthetic": sum(
                segment.verdict == LIKELY_SYNTHETIC for segment in segment_analysis.scored
            ),
            "report_pdf": str(pdf_path.name),
            "report_json": str(json_path.name),
        },
    )
    logger.info("Analysis %s: %s (score %.4f).", report_id, verdict, fake_score)
    return ForensicAnalysis(findings=findings, pdf_path=pdf_path, json_path=json_path, custody_entry=custody_entry)


def main() -> None:
    """CLI entrypoint: analyse one file, or verify the custody log."""
    parser = argparse.ArgumentParser(description="Deepfake analysis with PDF report and summary.")
    parser.add_argument("--audio", type=Path, help="Audio file to analyse.")
    parser.add_argument("--case-id", default="", help="Optional reference / case identifier.")
    parser.add_argument("--examiner", default="", help="Optional analyst / examiner name.")
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

    if not args.audio:
        parser.error("--audio is required unless --verify-log is given.")

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
    for warning in findings.warnings:
        print(f"Warning: {warning}")
    print(f"PDF report:  {analysis.pdf_path}")
    print(f"JSON report: {analysis.json_path}")


if __name__ == "__main__":
    main()
