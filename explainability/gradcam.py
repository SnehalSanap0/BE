"""Grad-CAM: gradient-weighted class activation mapping over a trained model.

Single responsibility: given a model and one input tensor, produce a 2-D
saliency map showing which regions of the spectrogram most influenced the
predicted class. This is what lets you visually confirm the model is
attending to plausible spectral artifacts rather than something spurious.
"""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from configs.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class GradCAMExplainer:
    """Wraps ``pytorch_grad_cam.GradCAM`` bound to one target layer of a model.

    Attributes:
        model: The model being explained.
        target_layer_path: Dotted module path Grad-CAM hooks into (e.g.
            ``"backbone.conv_head"``).
    """

    def __init__(self, model: nn.Module, target_layer_path: str | None = None) -> None:
        """Initialize the explainer by resolving and hooking the target layer.

        Args:
            model: The (already trained/loaded) model to explain.
            target_layer_path: Dotted path to the layer whose activations and
                gradients Grad-CAM will use. Defaults to
                ``CONFIG.explainability.target_layer``. Must name a
                convolutional layer late enough in the network to carry
                class-discriminative spatial information (EfficientNet-B0's
                final ``conv_head`` is the default).

        Raises:
            ValueError: If ``target_layer_path`` does not name a module that
                exists on ``model``.
        """
        self.model = model
        self.target_layer_path = target_layer_path or CONFIG.explainability.target_layer

        modules_by_name = dict(model.named_modules())
        if self.target_layer_path not in modules_by_name:
            raise ValueError(
                f"Target layer '{self.target_layer_path}' not found on model. "
                f"Available leaf modules include: {list(modules_by_name)[-10:]}"
            )

        target_layer = modules_by_name[self.target_layer_path]
        self._cam = GradCAM(model=model, target_layers=[target_layer])

        logger.info("GradCAMExplainer bound to layer '%s'.", self.target_layer_path)

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Compute a Grad-CAM saliency map for one input.

        Args:
            input_tensor: A single-example batch, shape ``(1, 1, H, W)``.
                Must be produced outside any ``torch.no_grad()`` context,
                since Grad-CAM needs to backpropagate through the model.
            target_class: Class index to explain. Defaults to the model's
                own top prediction for this input.

        Returns:
            2-D array of shape ``(H, W)`` with values in ``[0, 1]``, where
            higher values indicate greater influence on the target class.
        """
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        grayscale_cam = self._cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0]
