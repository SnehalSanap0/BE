"""Loss function construction for binary real-vs-fake classification."""

from __future__ import annotations

import torch
from torch import nn

from utils.logger import get_logger

logger = get_logger(__name__)


def build_loss_fn(class_weights: torch.Tensor | None = None) -> nn.Module:
    """Construct the training loss function.

    Args:
        class_weights: Optional per-class weight tensor of shape
            ``(num_classes,)``, ordered per
            :data:`datasets.deepfake_dataset.LABEL_TO_INDEX`. Pass this when
            the real/fake class counts are imbalanced, to avoid the loss
            being dominated by the majority class.

    Returns:
        A :class:`torch.nn.CrossEntropyLoss` configured with the given class weights.
    """
    if class_weights is not None:
        logger.info("Building CrossEntropyLoss with class weights: %s", class_weights.tolist())
    else:
        logger.info("Building CrossEntropyLoss with uniform class weights.")

    return nn.CrossEntropyLoss(weight=class_weights)
