"""Lightweight running-average tracking for the training loop.

Distinct from :mod:`evaluation.metrics`: this module only tracks scalar
running averages (loss, accuracy) for live progress-bar/TensorBoard logging
during a training epoch. The full metric suite (precision/recall/F1/ROC-AUC/
EER/confusion matrix) used for validation and final reporting lives in
:mod:`evaluation.metrics` and is reused as-is by :class:`training.trainer.Trainer`.
"""

from __future__ import annotations

import torch


class AverageMeter:
    """Tracks a running average of a scalar value (e.g. batch loss) over an epoch."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all running statistics to zero.

        Returns:
            None.
        """
        self.total = 0.0
        self.count = 0

    def update(self, value: float, num_samples: int = 1) -> None:
        """Accumulate one batch's value into the running average.

        Args:
            value: The batch's scalar value (e.g. mean loss for the batch).
            num_samples: Number of samples the value was averaged over,
                so the running average is correctly sample-weighted.

        Returns:
            None.
        """
        self.total += value * num_samples
        self.count += num_samples

    @property
    def average(self) -> float:
        """Current running average.

        Returns:
            ``total / count``, or ``0.0`` if no values have been recorded yet.
        """
        return self.total / self.count if self.count > 0 else 0.0


def compute_batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy for a single batch.

    Args:
        logits: Unnormalized model outputs, shape ``(batch, num_classes)``.
        labels: Ground-truth class indices, shape ``(batch,)``.

    Returns:
        Fraction of correctly predicted examples in ``[0, 1]``.
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)
