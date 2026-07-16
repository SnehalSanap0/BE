"""Baseline model: EfficientNet-B0 transfer learning for binary classification.

This is the simplest model in the project — a pretrained EfficientNet-B0
backbone with global average pooling (built into ``timm`` when
``num_classes=0``) followed by a small dropout + linear classification head.
It exists as the accuracy floor that :class:`models.efficientnet_bilstm.EfficientNetBiLSTM`
must beat to justify its added complexity.
"""

from __future__ import annotations

import torch
from torch import nn

from configs.config import CONFIG
from models.backbone import EFFICIENTNET_B0_FEATURE_DIM, build_efficientnet_backbone
from utils.logger import get_logger

logger = get_logger(__name__)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 backbone with a linear classification head.

    Attributes:
        backbone: The pretrained EfficientNet-B0 feature extractor.
        classifier: Dropout + linear layer mapping pooled features to class logits.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        dropout: float | None = None,
        pretrained: bool | None = None,
        freeze_backbone: bool | None = None,
    ) -> None:
        """Initialize the baseline classifier.

        Args:
            num_classes: Number of output classes. Defaults to ``CONFIG.model.num_classes``.
            dropout: Dropout probability before the final linear layer.
                Defaults to ``CONFIG.model.dropout``.
            pretrained: Whether to load ImageNet-pretrained backbone weights.
                Defaults to ``CONFIG.model.pretrained``.
            freeze_backbone: Whether to freeze backbone weights. Defaults to
                ``CONFIG.model.freeze_backbone``.
        """
        super().__init__()

        num_classes = num_classes or CONFIG.model.num_classes
        dropout = dropout if dropout is not None else CONFIG.model.dropout

        self.backbone = build_efficientnet_backbone(pretrained=pretrained, freeze=freeze_backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(EFFICIENTNET_B0_FEATURE_DIM, num_classes),
        )

        logger.info("Constructed EfficientNetClassifier with %d output classes.", num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a batch of feature maps through the backbone and classifier head.

        Args:
            features: Batch of single-channel feature maps, shape
                ``(batch, 1, image_size, image_size)``.

        Returns:
            Unnormalized class logits, shape ``(batch, num_classes)``.
        """
        pooled_features = self.backbone(features)
        return self.classifier(pooled_features)
