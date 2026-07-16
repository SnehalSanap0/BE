"""Overlay a Grad-CAM saliency map onto a mel-spectrogram feature image and save it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

from configs.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

#: Maps the config's colormap name to an OpenCV colormap constant.
_COLORMAPS: dict[str, int] = {
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
}


def _normalize_to_unit_range(feature: np.ndarray) -> np.ndarray:
    """Min-max normalize a feature array to ``[0, 1]`` for display.

    Args:
        feature: 2-D feature array (typically z-score normalized, so values
            are not already in a displayable range).

    Returns:
        2-D array with values in ``[0, 1]``. Returns all-zeros if the input
        is constant, to avoid a division by zero.
    """
    min_value, max_value = feature.min(), feature.max()
    if max_value - min_value < 1e-8:
        return np.zeros_like(feature)
    return (feature - min_value) / (max_value - min_value)


def overlay_cam_on_feature(
    feature: np.ndarray,
    cam: np.ndarray,
    colormap: str | None = None,
    overlay_alpha: float | None = None,
) -> np.ndarray:
    """Blend a Grad-CAM saliency map onto a feature image as a heatmap overlay.

    Args:
        feature: 2-D feature array (e.g. the normalized mel-spectrogram fed
            to the model), shape ``(H, W)``.
        cam: 2-D Grad-CAM saliency map in ``[0, 1]``, shape ``(H, W)``,
            matching ``feature``'s spatial size.
        colormap: Name of the heatmap colormap. Defaults to
            ``CONFIG.explainability.colormap``.
        overlay_alpha: Blend weight of the heatmap (``0`` = only the
            original spectrogram visible, ``1`` = only the heatmap visible).
            Defaults to ``CONFIG.explainability.overlay_alpha``.

    Returns:
        ``uint8`` RGB image of shape ``(H, W, 3)`` with the heatmap blended in.
    """
    colormap_name = (colormap or CONFIG.explainability.colormap).lower()
    overlay_alpha = overlay_alpha if overlay_alpha is not None else CONFIG.explainability.overlay_alpha
    cv2_colormap = _COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)

    normalized_feature = _normalize_to_unit_range(feature).astype(np.float32)
    rgb_image = np.stack([normalized_feature] * 3, axis=-1)

    overlay = show_cam_on_image(
        rgb_image,
        cam,
        use_rgb=True,
        colormap=cv2_colormap,
        image_weight=1.0 - overlay_alpha,
    )
    return overlay


def save_overlay(overlay: np.ndarray, output_path: Path) -> None:
    """Save an RGB overlay image to disk.

    Args:
        overlay: ``uint8`` RGB image, shape ``(H, W, 3)``.
        output_path: Destination path. Parent directories are created automatically.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_image = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)
    logger.info("Saved Grad-CAM overlay to %s", output_path)
