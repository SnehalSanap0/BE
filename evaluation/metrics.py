"""Classification metrics for real-vs-fake audio detection.

Single responsibility: turn arrays of ground-truth labels, predicted labels
and predicted positive-class scores into the metric suite this project
reports (accuracy, precision, recall, F1, ROC-AUC, EER, confusion matrix).
Used both by :mod:`training.trainer` (per-epoch validation) and
:mod:`evaluation.evaluator` (final test-set reporting), so metric
definitions never drift between the two.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Compute the Equal Error Rate: the point where false-accept and false-reject rates match.

    EER is the standard reporting metric in anti-spoofing / audio deepfake
    detection literature — it summarizes the full ROC curve into a single
    threshold-independent number, which is why it is preferred over
    accuracy alone for this task.

    Args:
        y_true: Ground-truth binary labels (1 = fake/positive class).
        y_scores: Predicted probability/score for the positive class.

    Returns:
        Tuple of ``(eer, threshold)`` where ``eer`` is in ``[0, 1]`` and
        ``threshold`` is the score cutoff at which it occurs. Returns
        ``(nan, nan)`` if ``y_true`` contains only one class, since a ROC
        curve is undefined in that case.
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Cannot compute EER: y_true contains only one class.")
        return float("nan"), float("nan")

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_scores)
    false_negative_rate = 1 - true_positive_rate

    eer_index = int(np.nanargmin(np.abs(false_positive_rate - false_negative_rate)))
    eer = float((false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2)
    eer_threshold = float(thresholds[eer_index])

    return eer, eer_threshold


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict[str, object]:
    """Compute the full classification metric suite for one evaluation pass.

    Args:
        y_true: Ground-truth binary labels, shape ``(n,)``.
        y_pred: Predicted binary labels (thresholded), shape ``(n,)``.
        y_scores: Predicted probability for the positive class, shape ``(n,)``.

    Returns:
        Dict with keys ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc``, ``eer``, ``eer_threshold`` (all floats) and
        ``confusion_matrix`` (nested list, rows=true, cols=predicted).
        ``roc_auc``/``eer`` are ``nan`` if only one class is present in
        ``y_true`` (e.g. a very small evaluation batch).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) < 2:
        logger.warning("Cannot compute ROC-AUC: y_true contains only one class.")
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_scores))

    eer, eer_threshold = compute_eer(y_true, y_scores)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "confusion_matrix": conf_matrix,
    }
