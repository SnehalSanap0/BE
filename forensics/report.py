"""Forensic report writers: a human-readable PDF and a machine-readable JSON twin.

The PDF is built with matplotlib's PDF backend, which the project already
depends on, so no extra reporting library is needed. Page 1 carries the
case, evidence integrity, model identity, verdict, known error rates and
limitations (continuing onto extra pages as needed); then a figure page with the score
timeline, spectrogram and Grad-CAM; then the full list of flagged windows, paginated.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from forensics.evidence import EvidenceInfo  # noqa: E402
from forensics.segments import SegmentAnalysis  # noqa: E402
from forensics.verdict import INCONCLUSIVE, LIKELY_GENUINE, LIKELY_SYNTHETIC, ValidationErrorRates  # noqa: E402

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]

_A4_INCHES = (8.27, 11.69)
_WRAP_CHARS = 90
_BOTTOM_MARGIN = 0.055
_VERDICT_COLORS = {LIKELY_GENUINE: "#16a34a", INCONCLUSIVE: "#d97706", LIKELY_SYNTHETIC: "#dc2626"}

DISCLAIMER = (
    "This report is the output of an automated statistical classifier and is intended to support, "
    "not replace, examination by a qualified forensic expert. A 'likely synthetic' verdict is not "
    "proof that any person acted deceptively, and a 'likely genuine' verdict is not proof of "
    "authenticity. The result must not be the sole basis for any legal, financial or "
    "identity-verification decision."
)


@dataclass(frozen=True)
class ForensicFindings:
    """Everything a report states about one analysis.

    Attributes:
        report_id: Unique ID of this report.
        generated_at_utc: ISO-8601 UTC timestamp of the analysis.
        evidence: Metadata of the evidence audio file.
        fake_score: Model probability that the audio is synthetic.
        verdict: Three-way verdict (see :mod:`forensics.verdict`).
        genuine_below: Lower verdict threshold used.
        synthetic_above: Upper verdict threshold used.
        analysed_seconds: Length of audio actually fed to the model.
        peak_region_caption: Time/frequency region that most influenced the prediction.
        error_rates: Known validation error rates at these thresholds, if available.
        case_id: Optional reference / case identifier.
        examiner: Optional examiner / analyst name.
        evidence_sha256_after: Optional hash for backwards compatibility.
        model_label: Human-readable model description.
        checkpoint_path: Checkpoint file used.
        checkpoint_sha256: SHA-256 of the checkpoint file if tracked.
        checkpoint_epoch: Training epoch the checkpoint was saved at.
        segment_analysis: Per-window scores over the whole recording, if computed.
        validated: ``False`` if no held-out error rate exists for the model; the verdict is then
            always INCONCLUSIVE and the scores are informational only.
        warnings: Analysis-specific caveats (e.g. many windows, little speech).
        spectrogram_image: ``uint8`` RGB spectrogram the model saw.
        overlay_image: ``uint8`` RGB Grad-CAM overlay.
        legend_image: ``uint8`` RGB Grad-CAM colorbar legend, if available.
    """

    report_id: str
    generated_at_utc: str
    evidence: EvidenceInfo
    fake_score: float
    verdict: str
    genuine_below: float
    synthetic_above: float
    analysed_seconds: float
    peak_region_caption: str
    error_rates: ValidationErrorRates | None
    case_id: str = ""
    examiner: str = ""
    evidence_sha256_after: str = ""
    model_label: str = ""
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    checkpoint_epoch: int = 0
    segment_analysis: SegmentAnalysis | None = None
    validated: bool = True
    warnings: tuple[str, ...] = ()
    spectrogram_image: np.ndarray | None = field(default=None, repr=False)
    overlay_image: np.ndarray | None = field(default=None, repr=False)
    legend_image: np.ndarray | None = field(default=None, repr=False)

    @property
    def integrity_verified(self) -> bool:
        """``True`` if integrity is verified."""
        if not self.evidence_sha256_after or not self.evidence.sha256:
            return True
        return self.evidence.sha256 == self.evidence_sha256_after

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the findings (images excluded)."""
        data = {
            "report_id": self.report_id,
            "generated_at_utc": self.generated_at_utc,
            "evidence": self.evidence.to_dict(),
            "model": {
                "label": self.model_label,
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_epoch": self.checkpoint_epoch,
            },
            "result": {
                "verdict": self.verdict,
                "fake_score": self.fake_score,
                "genuine_below": self.genuine_below,
                "synthetic_above": self.synthetic_above,
                "validated": self.validated,
                "analysed_seconds": self.analysed_seconds,
                "peak_region": self.peak_region_caption,
            },
            "validation_error_rates": self.error_rates.to_dict() if self.error_rates else None,
            "segment_analysis": self.segment_analysis.to_dict() if self.segment_analysis else None,
            "warnings": list(self.warnings),
            "disclaimer": DISCLAIMER,
        }
        if self.case_id:
            data["case_id"] = self.case_id
        if self.examiner:
            data["examiner"] = self.examiner
        return data


def write_json_report(findings: ForensicFindings, output_path: Path) -> Path:
    """Write the findings as pretty-printed JSON.

    Args:
        findings: The analysis to report.
        output_path: Destination ``.json`` path; parent directories are created.

    Returns:
        ``output_path``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(findings.to_dict(), file_handle, indent=2, ensure_ascii=False)
    return output_path


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _error_rate_lines(error_rates: ValidationErrorRates | None) -> list[str]:
    """Describe known error rates in plain language."""
    if error_rates is None:
        return [
            "NONE. No held-out error rate has been measured for this model, so no decisive verdict is "
            "issued and the scores below are informational only."
        ]
    lines = [
        f"Measured on {error_rates.genuine_count} genuine and {error_rates.synthetic_count} synthetic clips "
        f"from speech sources the model never saw in training ({error_rates.source_file}):",
        f"  - Genuine speech wrongly reported LIKELY SYNTHETIC: {_percent(error_rates.genuine_called_synthetic)}",
        f"  - Synthetic speech wrongly reported LIKELY GENUINE: {_percent(error_rates.synthetic_called_genuine)}",
        f"  - Reported INCONCLUSIVE: {_percent(error_rates.genuine_inconclusive)} of genuine, "
        f"{_percent(error_rates.synthetic_inconclusive)} of synthetic clips",
        "  Wrong decisive verdict by source:",
    ]
    lines += [f"    {source}: {_percent(rate)}" for source, rate in sorted(error_rates.per_source_wrong.items())]
    return lines


def _segment_summary_lines(findings: ForensicFindings) -> list[str]:
    """Describe how much of the recording was analysed and where the verdict came from."""
    analysis = findings.segment_analysis
    if analysis is None:
        return [f"Model input window: {findings.analysed_seconds:.2f} s"]
    scored = analysis.scored
    flagged = sum(segment.verdict == LIKELY_SYNTHETIC for segment in scored)
    inconclusive = sum(segment.verdict == INCONCLUSIVE for segment in scored)
    suspicious = analysis.most_suspicious
    return [
        f"Coverage: whole recording, {len(analysis.segments)} window(s) of {analysis.window_seconds:.1f} s "
        f"every {analysis.hop_seconds:.1f} s; {len(scored)} scored, "
        f"{len(analysis.segments) - len(scored)} skipped as silence",
        f"Windows LIKELY SYNTHETIC: {flagged}   INCONCLUSIVE: {inconclusive}",
        f"Verdict taken from the most suspicious window: {suspicious.start_seconds:.1f}-"
        f"{suspicious.end_seconds:.1f} s (explained by the Grad-CAM figure)",
    ]


def _page_one_sections(findings: ForensicFindings) -> list[tuple[str, list[str]]]:
    """Build (heading, lines) pairs for the text page."""
    evidence = findings.evidence
    analysis_details = [
        f"Report ID: {findings.report_id}",
        f"Generated (UTC): {findings.generated_at_utc}",
    ]
    if findings.case_id:
        analysis_details.append(f"Reference ID: {findings.case_id}")
    if findings.examiner:
        analysis_details.append(f"Examiner: {findings.examiner}")

    return [
        (
            "Analysis Details",
            analysis_details,
        ),
        (
            "Audio File",
            [
                f"File: {evidence.filename}",
                f"Format: {evidence.container_format} / {evidence.encoding}   Size: {evidence.size_bytes:,} bytes",
                f"Duration: {evidence.duration_seconds:.2f} s   Sample rate: {evidence.sample_rate} Hz   Channels: {evidence.channels}",
            ],
        ),
        (
            "Model & Configuration",
            [
                f"Architecture: {findings.model_label}",
                f"Checkpoint: {findings.checkpoint_path} (epoch {findings.checkpoint_epoch})",
            ],
        ),
        (
            "Findings & Decision",
            [
                f"Synthetic probability: {findings.fake_score:.4f} ({findings.fake_score * 100:.1f}%)",
                (
                    f"Decision rule: below {findings.genuine_below} = LIKELY GENUINE; above "
                    f"{findings.synthetic_above} = LIKELY SYNTHETIC; otherwise INCONCLUSIVE"
                    if findings.validated
                    else "Decision rule: none - model not validated, every score is INCONCLUSIVE"
                ),
                *_segment_summary_lines(findings),
                f"Most influential region: {findings.peak_region_caption}",
            ],
        ),
        ("Validation & Error Rates", _error_rate_lines(findings.error_rates)),
        ("Warnings", list(findings.warnings) or ["None."]),
        ("Disclaimer", [DISCLAIMER]),
    ]


def _draw_text_page(pdf: PdfPages, findings: ForensicFindings) -> None:
    """Render page 1: modern header, verdict banner, and clearly spaced text sections."""
    figure = plt.figure(figsize=_A4_INCHES)
    current_page = 1
    
    def _draw_header(fig: plt.Figure, is_continuation: bool = False) -> None:
        title = "Audio Deepfake Analysis Report"
        sub = f"Report ID: {findings.report_id}" + (" (continued)" if is_continuation else f"   |   Generated: {findings.generated_at_utc}")
        fig.text(0.07, 0.958, title, fontsize=16, weight="bold", color="#0f172a", family="sans-serif")
        fig.text(0.07, 0.938, sub, fontsize=8.5, color="#64748b", family="sans-serif")
        fig.add_artist(plt.Line2D([0.07, 0.93], [0.928, 0.928], color="#e2e8f0", linewidth=1.2))

    def _draw_footer(fig: plt.Figure, page_num: int) -> None:
        fig.add_artist(plt.Line2D([0.07, 0.93], [0.038, 0.038], color="#e2e8f0", linewidth=0.8))
        fig.text(0.07, 0.022, "Audio Deepfake Analysis & Forensic Verification Platform", fontsize=7.5, color="#94a3b8", family="sans-serif")
        fig.text(0.93, 0.022, f"Page {page_num}", fontsize=7.5, color="#94a3b8", ha="right", family="sans-serif")

    _draw_header(figure)

    # 2. Verdict & Score Banner
    verdict_color = _VERDICT_COLORS.get(findings.verdict, "#475569")
    figure.text(
        0.07,
        0.884,
        f"  VERDICT: {findings.verdict}  ",
        fontsize=12.5,
        weight="bold",
        color="white",
        family="sans-serif",
        bbox={"facecolor": verdict_color, "edgecolor": "none", "boxstyle": "round,pad=0.55"},
    )
    figure.text(
        0.46,
        0.886,
        f"Synthetic Probability: {findings.fake_score * 100:.1f}%",
        fontsize=11.5,
        weight="bold",
        color="#0f172a",
        family="sans-serif",
    )

    if not findings.validated:
        figure.text(
            0.07,
            0.850,
            "MODEL NOT VALIDATED - scores are informational only; no decisive verdict.",
            fontsize=8.5,
            weight="bold",
            color=_VERDICT_COLORS[LIKELY_SYNTHETIC],
            family="sans-serif",
        )
        y_position = 0.820
    else:
        y_position = 0.838

    line_height = 0.0195

    def _ensure_room(needed: float) -> None:
        """Start a continuation page when the next item would run past the bottom margin."""
        nonlocal figure, y_position, current_page
        if y_position - needed >= _BOTTOM_MARGIN:
            return
        _draw_footer(figure, current_page)
        pdf.savefig(figure)
        plt.close(figure)
        current_page += 1
        figure = plt.figure(figsize=_A4_INCHES)
        _draw_header(figure, is_continuation=True)
        y_position = 0.895

    # 3. Text Sections with refined typography, section dividers, and spacing
    for heading, lines in _page_one_sections(findings):
        _ensure_room(line_height * 3.2)
        
        # Section Heading with subtle accent divider
        figure.text(0.07, y_position, heading.upper(), fontsize=9.5, weight="bold", color="#1e3a8a", family="sans-serif")
        figure.add_artist(plt.Line2D([0.07, 0.93], [y_position - 0.003, y_position - 0.003], color="#cbd5e1", linewidth=0.75))
        y_position -= line_height * 1.35
        
        is_disclaimer = (heading == "Disclaimer")
        
        for line in lines:
            indent = len(line) - len(line.lstrip(" "))
            is_bullet = line.strip().startswith("-") or indent > 0
            
            clean_line = line.strip()
            if clean_line.startswith("- "):
                clean_line = "•  " + clean_line[2:]
            
            wrapped = textwrap.wrap(clean_line, width=86 - int(indent * 1.5)) or [""]
            for piece in wrapped:
                _ensure_room(line_height)
                if is_disclaimer:
                    font_size = 8.0
                    text_color = "#64748b"
                    font_style = "italic"
                    x_pos = 0.07
                elif is_bullet:
                    font_size = 8.5
                    text_color = "#334155"
                    font_style = "normal"
                    x_pos = 0.085
                else:
                    font_size = 8.8
                    text_color = "#0f172a"
                    font_style = "normal"
                    x_pos = 0.07
                
                figure.text(
                    x_pos,
                    y_position,
                    piece,
                    fontsize=font_size,
                    color=text_color,
                    style=font_style,
                    family="sans-serif",
                )
                y_position -= line_height
        
        # Clean breathing room between sections
        y_position -= line_height * 0.75

    _draw_footer(figure, current_page)
    pdf.savefig(figure)
    plt.close(figure)


def _draw_figure_page(pdf: PdfPages, findings: ForensicFindings) -> None:
    """Render the figure page: score timeline and spectrogram + Grad-CAM of the most suspicious window."""
    figure = plt.figure(figsize=_A4_INCHES)
    figure.text(0.07, 0.958, "Model Evidence & Visualizations", fontsize=15, weight="bold", color="#0f172a", family="sans-serif")
    figure.text(0.07, 0.938, f"Report ID: {findings.report_id}   |   Grad-CAM & Segment Timeline", fontsize=8.5, color="#64748b", family="sans-serif")
    figure.add_artist(plt.Line2D([0.07, 0.93], [0.928, 0.928], color="#e2e8f0", linewidth=1.2))

    timeline_axes = figure.add_axes([0.1, 0.74, 0.83, 0.17])
    plot_segment_timeline(timeline_axes, findings)

    window_text = ""
    if findings.segment_analysis is not None:
        suspicious = findings.segment_analysis.most_suspicious
        window_text = f" ({suspicious.start_seconds:.1f}-{suspicious.end_seconds:.1f} s)"
    figure.text(0.07, 0.675, f"Most suspicious window{window_text}", fontsize=11, weight="bold", color="#0f172a", family="sans-serif")
    figure.text(
        0.07,
        0.655,
        "Left: mel-spectrogram the model analysed (time left to right, low frequencies at the bottom). "
        "Right: Grad-CAM -",
        fontsize=8.5,
        color="#475569",
        family="sans-serif",
    )
    figure.text(0.07, 0.641, "warmer colours mark the regions that most influenced the verdict.", fontsize=8.5, color="#475569", family="sans-serif")

    panels = [("Spectrogram", findings.spectrogram_image), ("Grad-CAM overlay", findings.overlay_image)]
    for index, (title, image) in enumerate(panels):
        axes = figure.add_axes([0.07 + index * 0.45, 0.39, 0.41, 0.23])
        axes.set_title(title, fontsize=10, color="#1e293b", family="sans-serif")
        axes.axis("off")
        if image is not None:
            # Feature row 0 is the lowest mel band; origin="lower" puts low frequencies at the bottom.
            axes.imshow(image, origin="lower", aspect="auto")
        else:
            axes.text(0.5, 0.5, "not available", ha="center", va="center", color="#94a3b8", family="sans-serif")

    if findings.legend_image is not None:
        legend_axes = figure.add_axes([0.25, 0.335, 0.5, 0.035])
        legend_axes.imshow(findings.legend_image)
        legend_axes.axis("off")

    figure.text(0.07, 0.31, f"Most influential region: {findings.peak_region_caption}", fontsize=9, color="#1e293b", family="sans-serif")
    figure.text(
        0.07,
        0.293,
        "Grad-CAM shows where the model looked, not why; it is an aid to interpretation, not independent proof.",
        fontsize=8,
        style="italic",
        color="#64748b",
        family="sans-serif",
    )

    flagged = _flagged_windows(findings)
    if findings.segment_analysis is not None:
        figure.text(0.07, 0.26, "Windows not judged LIKELY GENUINE", fontsize=11, weight="bold", color="#0f172a", family="sans-serif")
        summary = (
            f"{len(flagged)} window(s) - listed in full on the following page(s)."
            if flagged
            else "None - every scored window was LIKELY GENUINE."
        )
        figure.text(0.07, 0.238, summary, fontsize=8.5, color="#334155", family="sans-serif")
    
    # Figure page footer
    figure.add_artist(plt.Line2D([0.07, 0.93], [0.038, 0.038], color="#e2e8f0", linewidth=0.8))
    figure.text(0.07, 0.022, "Audio Deepfake Analysis & Forensic Verification Platform", fontsize=7.5, color="#94a3b8", family="sans-serif")
    figure.text(0.93, 0.022, "Figure Page", fontsize=7.5, color="#94a3b8", ha="right", family="sans-serif")
    
    pdf.savefig(figure)
    plt.close(figure)

    if flagged:
        _draw_flagged_window_pages(pdf, findings, flagged)


_ROW_HEIGHT = 0.019


def _flagged_windows(findings: ForensicFindings) -> list:
    """Scored windows not judged LIKELY GENUINE, in time order."""
    if findings.segment_analysis is None:
        return []
    return [segment for segment in findings.segment_analysis.scored if segment.verdict != LIKELY_GENUINE]


def _draw_flagged_window_pages(pdf: PdfPages, findings: ForensicFindings, flagged: list) -> None:
    """List every flagged window as a table, paginated so no row is ever cut off."""
    suspicious = findings.segment_analysis.most_suspicious
    header = f"{'Time (s)':<18}{'Score':>8}   {'Speech':>6}   Verdict"
    rows = [
        f"{segment.start_seconds:7.1f}-{segment.end_seconds:<9.1f}{segment.fake_score:8.4f}   "
        f"{segment.speech_fraction:6.0%}   {segment.verdict}"
        + ("   <- most suspicious" if segment == suspicious else "")
        for segment in flagged
    ]
    first_row_y = 0.866
    rows_per_page = int((first_row_y - _BOTTOM_MARGIN) / _ROW_HEIGHT)
    page_count = -(-len(rows) // rows_per_page)

    for page_index in range(page_count):
        figure = plt.figure(figsize=_A4_INCHES)
        title = "Windows not judged LIKELY GENUINE"
        if page_count > 1:
            title += f" ({page_index + 1}/{page_count})"
        figure.text(0.07, 0.955, title, fontsize=14, weight="bold")
        figure.text(0.07, 0.93, f"Report {findings.report_id}", fontsize=8.5, color="#444444")
        figure.text(
            0.07,
            0.914,
            f"{len(flagged)} window(s), in time order. Speech = non-silent share of the window.",
            fontsize=8.5,
            color="#444444",
        )
        figure.text(0.07, 0.89, header, fontsize=8.5, family="monospace", weight="bold")
        page_rows = rows[page_index * rows_per_page : (page_index + 1) * rows_per_page]
        for row_index, row in enumerate(page_rows):
            figure.text(0.07, first_row_y - row_index * _ROW_HEIGHT, row, fontsize=8.5, family="monospace")
        pdf.savefig(figure)
        plt.close(figure)


def plot_segment_timeline(axes: plt.Axes, findings: ForensicFindings) -> None:
    """Plot each window's fake score across the recording, with the verdict bands shaded.

    Args:
        axes: Matplotlib axes to draw into.
        findings: Findings carrying a :class:`~forensics.segments.SegmentAnalysis`.
    """
    duration = max(findings.evidence.duration_seconds, 1e-3)
    if findings.validated:
        axes.axhspan(findings.synthetic_above, 1.0, color=_VERDICT_COLORS[LIKELY_SYNTHETIC], alpha=0.10)
        axes.axhspan(0.0, findings.genuine_below, color=_VERDICT_COLORS[LIKELY_GENUINE], alpha=0.10)
        axes.axhline(findings.synthetic_above, color=_VERDICT_COLORS[LIKELY_SYNTHETIC], linewidth=0.8, linestyle="--")
        axes.axhline(findings.genuine_below, color=_VERDICT_COLORS[LIKELY_GENUINE], linewidth=0.8, linestyle="--")

    analysis = findings.segment_analysis
    if analysis is not None:
        for segment in analysis.segments:
            if segment.fake_score is None:
                axes.axvspan(segment.start_seconds, segment.end_seconds, color="#999999", alpha=0.08, linewidth=0)
                continue
            color = _VERDICT_COLORS.get(segment.verdict, "#333333")
            axes.hlines(segment.fake_score, segment.start_seconds, segment.end_seconds, color=color, linewidth=3)
        scored = analysis.scored
        centers = [(segment.start_seconds + segment.end_seconds) / 2 for segment in scored]
        axes.plot(centers, [segment.fake_score for segment in scored], color="#555555", linewidth=0.8)

    axes.set_xlim(0, duration)
    axes.set_ylim(-0.03, 1.03)
    axes.set_xlabel("Time in recording (s)", fontsize=8.5)
    axes.set_ylabel("Probability synthetic", fontsize=8.5)
    axes.set_title(
        (
            "Score per window (red band: likely synthetic, green band: likely genuine, grey: skipped silence)"
            if findings.validated
            else "Score per window (model not validated - informational only; grey: skipped silence)"
        ),
        fontsize=9,
    )
    axes.tick_params(labelsize=8)


def render_timeline_image(findings: ForensicFindings) -> np.ndarray:
    """Render the segment timeline as an RGB image (for UIs such as the Gradio demo).

    Args:
        findings: Findings carrying a :class:`~forensics.segments.SegmentAnalysis`.

    Returns:
        ``uint8`` RGB array.
    """
    figure, axes = plt.subplots(figsize=(9, 2.6), dpi=110)
    plot_segment_timeline(axes, findings)
    figure.tight_layout()
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return image


def write_pdf_report(findings: ForensicFindings, output_path: Path) -> Path:
    """Write the PDF report (text pages, figure page, flagged-window pages).

    Args:
        findings: The analysis to report.
        output_path: Destination ``.pdf`` path; parent directories are created.

    Returns:
        ``output_path``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _draw_text_page(pdf, findings)
        _draw_figure_page(pdf, findings)
        metadata = pdf.infodict()
        metadata["Title"] = f"Audio deepfake analysis report - {findings.report_id}"
        if findings.examiner:
            metadata["Author"] = findings.examiner
    return output_path
