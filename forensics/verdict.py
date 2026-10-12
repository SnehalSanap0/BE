"""Three-way forensic verdict and the model's known error rates at those thresholds.

A binary real/fake label at 0.5 is the wrong output for evidentiary use: the
cross-source experiments showed scores on unseen corpora are poorly
calibrated (docs/research_log.md, 2026-07-24), so borderline scores are
reported as INCONCLUSIVE instead of being forced into a class. Every report
also states how often the model was wrong at these exact thresholds on
external validation data -- a court needs the method's known error rate, not
just its answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LIKELY_GENUINE = "LIKELY GENUINE"
INCONCLUSIVE = "INCONCLUSIVE"
LIKELY_SYNTHETIC = "LIKELY SYNTHETIC"

#: Human-readable names for the source tags used in cross-source validation files.
SOURCE_DISPLAY_NAMES: dict[str, str] = {
    "real": "Kathbath (genuine)",
    "real_cv_m": "Common Voice male (genuine)",
    "real_cv_f": "Common Voice female (genuine)",
    "fake": "IndicSynth (synthetic)",
}


def classify(fake_score: float, genuine_below: float, synthetic_above: float) -> str:
    """Map a fake probability onto the three-way verdict.

    Args:
        fake_score: Model probability that the clip is synthetic, in ``[0, 1]``.
        genuine_below: Scores strictly below this are LIKELY GENUINE.
        synthetic_above: Scores strictly above this are LIKELY SYNTHETIC.

    Returns:
        One of :data:`LIKELY_GENUINE`, :data:`INCONCLUSIVE`, :data:`LIKELY_SYNTHETIC`.
    """
    if fake_score > synthetic_above:
        return LIKELY_SYNTHETIC
    if fake_score < genuine_below:
        return LIKELY_GENUINE
    return INCONCLUSIVE


@dataclass(frozen=True)
class ValidationErrorRates:
    """How the three-way verdict behaved on held-out, never-trained-on sources.

    Attributes:
        source_file: Validation score file these rates were computed from.
        genuine_count: Number of unique genuine validation clips.
        synthetic_count: Number of unique synthetic validation clips.
        genuine_called_synthetic: Fraction of genuine clips given LIKELY SYNTHETIC (false alarm).
        synthetic_called_genuine: Fraction of synthetic clips given LIKELY GENUINE (miss).
        genuine_inconclusive: Fraction of genuine clips given INCONCLUSIVE.
        synthetic_inconclusive: Fraction of synthetic clips given INCONCLUSIVE.
        per_source_wrong: ``{source display name: fraction given the wrong decisive verdict}``.
    """

    source_file: str
    genuine_count: int
    synthetic_count: int
    genuine_called_synthetic: float
    synthetic_called_genuine: float
    genuine_inconclusive: float
    synthetic_inconclusive: float
    per_source_wrong: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of every field."""
        return {
            "source_file": self.source_file,
            "genuine_count": self.genuine_count,
            "synthetic_count": self.synthetic_count,
            "genuine_called_synthetic": self.genuine_called_synthetic,
            "synthetic_called_genuine": self.synthetic_called_genuine,
            "genuine_inconclusive": self.genuine_inconclusive,
            "synthetic_inconclusive": self.synthetic_inconclusive,
            "per_source_wrong": dict(self.per_source_wrong),
        }


def load_validation_error_rates(
    scores_path: Path,
    genuine_below: float,
    synthetic_above: float,
    expected_checkpoint: Path | None = None,
    project_root: Path | None = None,
) -> ValidationErrorRates | None:
    """Compute verdict error rates from a saved cross-source evaluation file.

    Reads the clip-level ``fake_score`` predictions written by
    ``scripts/evaluate_cross_source_protocol.py``. Sampling manifests overlap
    across sampling seeds, so clips are de-duplicated by file path first.

    Args:
        scores_path: A ``cross_source_seed_<n>.json`` file.
        genuine_below: Lower verdict threshold (see :func:`classify`).
        synthetic_above: Upper verdict threshold (see :func:`classify`).
        expected_checkpoint: If given, the rates are only returned when the
            file was produced by this checkpoint -- another model's error
            rates must never appear in a report.
        project_root: Directory the file's relative ``checkpoint`` path is
            resolved against. Defaults to the current directory.

    Returns:
        The error rates, or ``None`` if the file does not exist or belongs to
        a different checkpoint (reports then state that no validated error
        rate is available).
    """
    scores_path = Path(scores_path)
    if not scores_path.exists():
        return None

    with scores_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if expected_checkpoint is not None:
        recorded = Path(str(payload.get("checkpoint", "")).replace("\\", "/"))
        if not recorded.is_absolute():
            recorded = (project_root or Path.cwd()) / recorded
        if recorded.resolve() != Path(expected_checkpoint).resolve():
            return None

    unique_predictions: dict[str, dict] = {}
    for run in payload["runs"]:
        for prediction in run["predictions"]:
            unique_predictions.setdefault(prediction["filepath"], prediction)

    verdict_counts: dict[str, dict[str, int]] = {"real": {}, "fake": {}}
    source_totals: dict[str, int] = {}
    source_wrong: dict[str, int] = {}
    for prediction in unique_predictions.values():
        true_label = prediction["true_label"]
        verdict = classify(prediction["fake_score"], genuine_below, synthetic_above)
        verdict_counts[true_label][verdict] = verdict_counts[true_label].get(verdict, 0) + 1

        source_tag = prediction["source_tag"]
        wrong_verdict = LIKELY_SYNTHETIC if true_label == "real" else LIKELY_GENUINE
        source_totals[source_tag] = source_totals.get(source_tag, 0) + 1
        source_wrong[source_tag] = source_wrong.get(source_tag, 0) + int(verdict == wrong_verdict)

    genuine_count = sum(verdict_counts["real"].values())
    synthetic_count = sum(verdict_counts["fake"].values())

    def _fraction(label: str, verdict: str, total: int) -> float:
        return verdict_counts[label].get(verdict, 0) / total if total else 0.0

    return ValidationErrorRates(
        source_file=scores_path.name,
        genuine_count=genuine_count,
        synthetic_count=synthetic_count,
        genuine_called_synthetic=_fraction("real", LIKELY_SYNTHETIC, genuine_count),
        synthetic_called_genuine=_fraction("fake", LIKELY_GENUINE, synthetic_count),
        genuine_inconclusive=_fraction("real", INCONCLUSIVE, genuine_count),
        synthetic_inconclusive=_fraction("fake", INCONCLUSIVE, synthetic_count),
        per_source_wrong={
            SOURCE_DISPLAY_NAMES.get(tag, tag): source_wrong[tag] / source_totals[tag] for tag in source_totals
        },
    )
