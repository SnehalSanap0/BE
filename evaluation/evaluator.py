"""Run a trained model over a dataloader and produce a metrics report + confusion matrix plot.

Single responsibility: model + dataloader in, metrics dict out (plus an
optional saved confusion-matrix figure). Reused by :mod:`training.trainer`
for lightweight per-epoch validation and by :mod:`evaluation.evaluate` for
the final, saved test-set report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.deepfake_dataset import INDEX_TO_LABEL
from evaluation.metrics import compute_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluates a model over a full dataloader and reports classification metrics."""

    def __init__(self, model: nn.Module, dataloader: DataLoader, device: torch.device) -> None:
        """Initialize the evaluator.

        Args:
            model: Trained (or in-training) classification model.
            dataloader: DataLoader over the split to evaluate.
            device: Device to run inference on.
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device

    @torch.no_grad()
    def evaluate(self) -> dict[str, object]:
        """Run inference over every batch and compute the full metric suite.

        Returns:
            The metrics dict produced by :func:`evaluation.metrics.compute_metrics`.
        """
        self.model.eval()

        all_labels: list[int] = []
        all_predictions: list[int] = []
        all_scores: list[float] = []

        for features, labels, _speakers, _languages in self.dataloader:
            features = features.to(self.device)

            logits = self.model(features)
            probabilities = torch.softmax(logits, dim=1)
            positive_scores = probabilities[:, 1]
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_scores.extend(positive_scores.cpu().tolist())

        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_predictions),
            y_scores=np.array(all_scores),
        )

        logger.info(
            "Evaluation: accuracy=%.4f, f1=%.4f, roc_auc=%.4f, eer=%.4f",
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["eer"],
        )
        return metrics

    def save_confusion_matrix(self, metrics: dict[str, object], output_path: Path) -> None:
        """Render and save a confusion matrix figure.

        Args:
            metrics: A metrics dict produced by :meth:`evaluate`, containing
                a ``confusion_matrix`` key.
            output_path: Destination image path. Parent directories are
                created automatically.

        Returns:
            None.
        """
        confusion = np.array(metrics["confusion_matrix"])
        labels = [INDEX_TO_LABEL[index] for index in range(len(confusion))]

        figure, axis = plt.subplots(figsize=(5, 5))
        image = axis.imshow(confusion, cmap="Blues")
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title("Confusion Matrix")

        for row in range(confusion.shape[0]):
            for col in range(confusion.shape[1]):
                axis.text(col, row, str(confusion[row, col]), ha="center", va="center")

        figure.colorbar(image, ax=axis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

        logger.info("Saved confusion matrix to %s", output_path)
