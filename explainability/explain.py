"""End-to-end explanation: raw audio file -> prediction + Grad-CAM overlay.

The core function here, :func:`explain_audio_file`, is deliberately the same
entrypoint both this module's CLI and :mod:`deployment.app` (Phase 8) call —
it reuses the exact preprocessing (:func:`preprocessing.transforms.preprocess_audio`)
and feature extraction (:meth:`feature_extraction.extractor.FeatureExtractor.extract_array`)
code paths as training, entirely in-memory (no intermediate files), so
what the model explains is guaranteed identical to what it was trained on.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from configs.config import CONFIG
from dataset.deepfake_dataset import INDEX_TO_LABEL
from explainability.gradcam import GradCAMExplainer
from explainability.visualizer import describe_peak_region, overlay_cam_on_feature, render_feature_as_image, save_overlay
from feature_extraction.extractor import FeatureExtractor
from models.model_factory import build_model
from preprocessing.audio_loader import load_audio
from preprocessing.transforms import preprocess_audio
from preprocessing.validator import AudioValidator
from training.checkpoint import BEST_CHECKPOINT_FILENAME, CheckpointManager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExplanationResult:
    """Outcome of explaining one audio file.

    Attributes:
        predicted_label: ``"real"`` or ``"fake"``.
        confidence: Model's softmax probability for ``predicted_label``, in ``[0, 1]``.
        processed_audio: The mono/resampled/silence-trimmed/normalized/padded
            waveform actually fed to feature extraction (see
            :func:`preprocessing.transforms.preprocess_audio`) -- shown ahead
            of the spectrogram so a viewer can hear what preprocessing did to
            their raw upload before the model ever sees it.
        processed_sample_rate: Sample rate of ``processed_audio``, in Hz
            (``CONFIG.audio.sample_rate``).
        spectrogram_image: The plain input feature (e.g. mel-spectrogram) the
            model was fed, rendered as a viewable ``uint8`` RGB array -- shown
            alongside ``overlay_image`` so a viewer can compare input vs.
            attention.
        overlay_image: The Grad-CAM heatmap overlay, ``uint8`` RGB array.
        output_path: Path the overlay image was saved to.
        peak_region_caption: Human-readable time/frequency description of the
            spectrogram region that most influenced ``predicted_label`` (see
            :func:`explainability.visualizer.describe_peak_region`).
    """

    predicted_label: str
    confidence: float
    processed_audio: np.ndarray
    processed_sample_rate: int
    spectrogram_image: np.ndarray
    overlay_image: np.ndarray
    output_path: Path
    peak_region_caption: str


def explain_audio_file(
    audio_path: Path,
    model: nn.Module,
    device: torch.device,
    feature_extractor: FeatureExtractor | None = None,
    explainer: GradCAMExplainer | None = None,
    output_dir: Path | None = None,
) -> ExplanationResult:
    """Validate, preprocess, predict, and Grad-CAM-explain a single audio file.

    Args:
        audio_path: Path to a raw (arbitrary-length, arbitrary-format) audio file.
        model: A loaded classification model (baseline or BiLSTM), on ``device``.
        device: Device to run inference on.
        feature_extractor: Feature extractor to use. Defaults to a new
            :class:`FeatureExtractor` built from project configuration.
        explainer: Grad-CAM explainer bound to ``model``. Defaults to a new
            :class:`GradCAMExplainer`.
        output_dir: Directory to save the overlay image under. Defaults to
            ``CONFIG.paths.explanation_dir``.

    Returns:
        An :class:`ExplanationResult` with the prediction and saved overlay.

    Raises:
        ValueError: If ``audio_path`` fails validation (corrupt, unsupported
            format, empty, etc.).
    """
    validation = AudioValidator().validate(audio_path)
    if not validation.is_valid:
        raise ValueError(f"Invalid audio file '{audio_path.name}': {validation.reason}")

    raw_audio, sample_rate = load_audio(audio_path)
    return explain_waveform(
        raw_audio,
        sample_rate,
        model=model,
        device=device,
        output_stem=audio_path.stem,
        feature_extractor=feature_extractor,
        explainer=explainer,
        output_dir=output_dir,
    )


def explain_waveform(
    raw_audio: np.ndarray,
    sample_rate: int,
    model: nn.Module,
    device: torch.device,
    output_stem: str,
    feature_extractor: FeatureExtractor | None = None,
    explainer: GradCAMExplainer | None = None,
    output_dir: Path | None = None,
) -> ExplanationResult:
    """Preprocess, predict, and Grad-CAM-explain an in-memory waveform.

    Used directly to explain one window of a longer recording (see
    :mod:`forensics.segments`); :func:`explain_audio_file` wraps it for files.

    Args:
        raw_audio: Raw samples, mono or ``(frames, channels)``.
        sample_rate: Sample rate of ``raw_audio``, in Hz.
        model: A loaded classification model, on ``device``.
        device: Device to run inference on.
        output_stem: File-name stem for the saved overlay image.
        feature_extractor: Feature extractor; defaults to a new :class:`FeatureExtractor`.
        explainer: Grad-CAM explainer bound to ``model``; defaults to a new one.
        output_dir: Overlay directory; defaults to ``CONFIG.paths.explanation_dir``.

    Returns:
        An :class:`ExplanationResult`.
    """
    processed_audio = preprocess_audio(raw_audio, sample_rate)

    feature_extractor = feature_extractor or FeatureExtractor()
    feature = feature_extractor.extract_array(processed_audio, CONFIG.audio.sample_rate)

    explainer = explainer or GradCAMExplainer(model)

    input_tensor = torch.from_numpy(feature).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_index = int(torch.argmax(logits, dim=1).item())
    confidence = float(probabilities[0, predicted_index].item())

    grayscale_cam = explainer.generate(input_tensor, target_class=predicted_index)
    spectrogram_image = render_feature_as_image(feature)
    overlay = overlay_cam_on_feature(feature, grayscale_cam)
    peak_region_caption = describe_peak_region(grayscale_cam)

    output_dir = output_dir or CONFIG.paths.explanation_dir
    output_path = output_dir / f"{output_stem}_gradcam.png"
    save_overlay(overlay, output_path)

    predicted_label = INDEX_TO_LABEL[predicted_index]
    logger.info(
        "Explained %s: predicted='%s' confidence=%.4f peak_region='%s' -> %s",
        output_stem,
        predicted_label,
        confidence,
        peak_region_caption,
        output_path,
    )

    return ExplanationResult(
        predicted_label=predicted_label,
        confidence=confidence,
        processed_audio=processed_audio,
        processed_sample_rate=CONFIG.audio.sample_rate,
        spectrogram_image=spectrogram_image,
        overlay_image=overlay,
        output_path=output_path,
        peak_region_caption=peak_region_caption,
    )


def main() -> None:
    """CLI: explain a single audio file using the best saved checkpoint.

    Usage:
        python -m explainability.explain --audio path/to/clip.wav

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Explain a single audio clip's real/fake prediction via Grad-CAM.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to a raw audio file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_manager = CheckpointManager(CONFIG.paths.checkpoint_dir)
    checkpoint_path = CONFIG.paths.checkpoint_dir / BEST_CHECKPOINT_FILENAME
    state = checkpoint_manager.load(checkpoint_path, map_location=device)

    use_bilstm = state.get("extra", {}).get("use_bilstm", CONFIG.model.use_bilstm)
    model = build_model(use_bilstm=use_bilstm)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    result = explain_audio_file(args.audio, model, device)

    print(f"Prediction: {result.predicted_label} (confidence={result.confidence:.4f})")
    print(f"Explanation saved to: {result.output_path}")


if __name__ == "__main__":
    main()
