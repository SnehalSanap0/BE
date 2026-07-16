"""Shared EfficientNet-B0 backbone construction.

Single responsibility: build a correctly-configured ``timm`` EfficientNet-B0
instance. Both :class:`models.baseline_classifier.EfficientNetClassifier`
and :class:`models.efficientnet_bilstm.EfficientNetBiLSTM` call this instead
of constructing ``timm`` models themselves, so backbone configuration (input
channels, pretrained weights, freezing) is defined in exactly one place.
"""

from __future__ import annotations

import timm
from torch import nn

from configs.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

#: EfficientNet-B0's feature dimensionality after global pooling, and the
#: channel depth of its final spatial feature map. Fixed by the architecture.
EFFICIENTNET_B0_FEATURE_DIM = 1280


def build_efficientnet_backbone(
    pretrained: bool | None = None,
    freeze: bool | None = None,
    in_channels: int = 1,
) -> nn.Module:
    """Construct an EfficientNet-B0 backbone with its classifier head removed.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights. Defaults to
            ``CONFIG.model.pretrained``.
        freeze: Whether to freeze all backbone parameters (useful for
            linear-probing or when training data is very limited). Defaults
            to ``CONFIG.model.freeze_backbone``.
        in_channels: Number of input channels. The project's feature maps
            (mel-spectrogram/MFCC/LFCC) are single-channel, so this defaults
            to ``1``; ``timm`` adapts the pretrained stem convolution's
            weights to match automatically.

    Returns:
        A ``timm`` EfficientNet-B0 module whose ``forward`` returns pooled
        features of shape ``(batch, 1280)`` and whose ``forward_features``
        returns spatial features of shape ``(batch, 1280, 7, 7)`` for a
        224x224 input.
    """
    pretrained = CONFIG.model.pretrained if pretrained is None else pretrained
    freeze = CONFIG.model.freeze_backbone if freeze is None else freeze

    backbone = timm.create_model(
        CONFIG.model.backbone,
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=0,
    )

    if freeze:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        logger.info("Backbone '%s' parameters frozen.", CONFIG.model.backbone)

    logger.info(
        "Built backbone '%s' (pretrained=%s, in_channels=%d, frozen=%s).",
        CONFIG.model.backbone,
        pretrained,
        in_channels,
        freeze,
    )
    return backbone
