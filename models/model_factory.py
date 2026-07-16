"""Configuration-driven model construction.

Single responsibility: translate ``CONFIG.model`` into a concrete
``nn.Module``. Training, evaluation and deployment code all call
:func:`build_model` instead of importing a specific model class, so
switching between the baseline and improved architecture is a one-line
config change (``model.use_bilstm``) rather than a code change anywhere
else in the project (Dependency Inversion).
"""

from __future__ import annotations

from torch import nn

from configs.config import CONFIG
from models.baseline_classifier import EfficientNetClassifier
from models.efficientnet_bilstm import EfficientNetBiLSTM
from utils.logger import get_logger

logger = get_logger(__name__)


def build_model(use_bilstm: bool | None = None) -> nn.Module:
    """Construct the classification model specified by project configuration.

    Args:
        use_bilstm: If ``True``, builds :class:`EfficientNetBiLSTM`; if
            ``False``, builds the baseline :class:`EfficientNetClassifier`.
            Defaults to ``CONFIG.model.use_bilstm``.

    Returns:
        An instantiated, uncompiled ``nn.Module`` ready for training or
        loading a checkpoint's state dict into.
    """
    use_bilstm = CONFIG.model.use_bilstm if use_bilstm is None else use_bilstm

    if use_bilstm:
        logger.info("Building improved model: EfficientNet-B0 + BiLSTM.")
        return EfficientNetBiLSTM()

    logger.info("Building baseline model: EfficientNet-B0 transfer learning.")
    return EfficientNetClassifier()
