"""Overlay a Grad-CAM saliency map onto a mel-spectrogram feature image and save it."""

from __future__ import annotations

from pathlib import Path

import cv2
import librosa
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


def render_feature_as_image(feature: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Render the raw model-input feature array (e.g. a mel-spectrogram) as a viewable image.

    This is the plain spectrogram the model actually sees -- shown alongside
    the Grad-CAM overlay so a viewer can compare "what the input looked like"
    against "what the model attended to," rather than only seeing the overlay
    on its own. Uses a different colormap than the Grad-CAM overlay by
    default (viridis vs. the overlay's jet) so the two are visually distinct
    even when shown side by side.

    Args:
        feature: 2-D feature array as fed to the model (e.g. the z-score
            normalized mel-spectrogram from
            :meth:`feature_extraction.extractor.FeatureExtractor.extract_array`).
        colormap: Name of the colormap to render with.

    Returns:
        ``uint8`` RGB image of shape ``(H, W, 3)``.
    """
    cv2_colormap = _COLORMAPS.get(colormap.lower(), cv2.COLORMAP_VIRIDIS)
    normalized = (_normalize_to_unit_range(feature) * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(normalized, cv2_colormap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


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


def render_colormap_legend(
    colormap: str | None = None,
    bar_width: int = 500,
    bar_height: int = 34,
) -> np.ndarray:
    """Render a horizontal colorbar legend for a Grad-CAM overlay's colormap.

    Horizontal (not vertical) so it can span the available width of a wide
    UI row instead of sitting as a narrow strip -- a wide, short bar reads
    more naturally as "a legend for the row above/below it" than a tall,
    thin one does.

    Args:
        colormap: Name of the heatmap colormap. Defaults to
            ``CONFIG.explainability.colormap``, matching the overlay it labels.
        bar_width: Width in pixels of the color gradient strip. Gradio scales
            the returned image to fit its container, so this mainly sets the
            aspect ratio, not the on-screen size.
        bar_height: Height in pixels of the color gradient strip.

    Returns:
        ``uint8`` RGB image with the gradient strip plus "Low"/"High" text
        labels at each end, on a white background.
    """
    colormap_name = (colormap or CONFIG.explainability.colormap).lower()
    cv2_colormap = _COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)

    gradient_row = np.linspace(0, 255, bar_width, dtype=np.uint8).reshape(1, bar_width)
    gradient = np.repeat(gradient_row, bar_height, axis=0)
    colored_bgr = cv2.applyColorMap(gradient, cv2_colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    label_height = 22
    canvas = np.full((bar_height + label_height, bar_width, 3), 255, dtype=np.uint8)
    canvas[:bar_height, :] = colored_rgb

    text_y = bar_height + 16
    cv2.putText(canvas, "Low influence", (4, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    high_text = "High influence"
    text_size, _ = cv2.getTextSize(high_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        canvas, high_text, (bar_width - text_size[0] - 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
    return canvas


def describe_peak_region(cam: np.ndarray, top_fraction: float = 0.1) -> str:
    """Describe the time/frequency region of a Grad-CAM map with the highest influence.

    Maps the highest-activation pixels back to approximate frequency (via
    ``CONFIG.features``' mel range) and time (via
    ``CONFIG.audio.target_duration_seconds``) coordinates, for a
    human-readable caption. This is approximate: the feature map is resized
    to a square before being fed to the model (see
    :func:`feature_extraction.extractor.FeatureExtractor.extract_array`), so
    row/column indices are re-mapped onto the mel/time axes proportionally
    rather than read off the original (pre-resize) spectrogram bins.

    Args:
        cam: 2-D Grad-CAM saliency map in ``[0, 1]``, shape ``(H, W)``.
        top_fraction: Fraction of highest-valued pixels to treat as the
            "peak" region (also floored to at least half the map's max value,
            so a near-uniform map doesn't report a spurious wide region).

    Returns:
        A caption like ``"2.1-3.4 kHz, 0.40-0.70s"``, or a fallback message
        if no clear peak region exists (e.g. an all-zero map).
    """
    if cam.max() <= 0:
        return "no dominant region (flat activation)"

    image_size = cam.shape[0]
    threshold = max(np.quantile(cam, 1.0 - top_fraction), cam.max() * 0.5)
    rows, cols = np.nonzero(cam >= threshold)
    if rows.size == 0:
        return "no dominant region (flat activation)"

    cfg = CONFIG.features
    mel_frequencies = librosa.mel_frequencies(n_mels=image_size, fmin=cfg.fmin, fmax=cfg.fmax)
    time_axis = np.linspace(0.0, CONFIG.audio.target_duration_seconds, image_size)

    freq_low, freq_high = mel_frequencies[rows.min()], mel_frequencies[rows.max()]
    time_low, time_high = time_axis[cols.min()], time_axis[cols.max()]

    return f"{freq_low / 1000:.1f}-{freq_high / 1000:.1f} kHz, {time_low:.2f}-{time_high:.2f}s"


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
