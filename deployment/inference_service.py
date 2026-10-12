"""Loads a trained checkpoint once and serves predictions for individual audio files.

Single responsibility: own the model/device/explainer lifecycle. Kept
separate from :mod:`deployment.app` so the Flask layer is pure HTTP routing
— the model is loaded exactly once at service construction, not per request,
and this class is equally usable from a script or test without Flask at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from configs.config import CONFIG
from dataset.deepfake_dataset import LABEL_TO_INDEX
from explainability.explain import ExplanationResult, explain_audio_file, explain_waveform
from explainability.gradcam import GradCAMExplainer
from feature_extraction.extractor import FeatureExtractor
from forensics.evidence import sha256_file
from models.model_factory import build_model
from training.checkpoint import CheckpointManager
from utils.logger import get_logger

logger = get_logger(__name__)


class InferenceService:
    """Holds one loaded model in memory and predicts+explains audio files against it."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: torch.device | None = None,
        feature_image_size: int | None = None,
    ) -> None:
        """Load the best checkpoint and construct the model, feature extractor and explainer.

        Args:
            checkpoint_path: Path to a checkpoint saved by
                :class:`training.checkpoint.CheckpointManager`. Defaults to
                ``CONFIG.deployment.model_checkpoint``.
            device: Device to run inference on. Defaults to CUDA if available, else CPU.
            feature_image_size: Override for the spatial size features are
                resized to (see :class:`feature_extraction.extractor.FeatureExtractor`).
                Defaults to ``CONFIG.features.image_size``. Needed when
                serving a checkpoint trained under a different config than
                the process-wide ``CONFIG`` singleton (e.g. a second
                language's smoke config with a different ``image_size`,
                since a checkpoint must be fed features shaped exactly as it
                was trained on, and ``CONFIG`` is fixed per process).

        Raises:
            FileNotFoundError: If no checkpoint exists at the resolved path
                (train a model first, e.g. ``python -m scripts.run_paired_seaspoof_experiment``).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_manager = CheckpointManager(CONFIG.paths.checkpoint_dir)
        resolved_checkpoint_path = checkpoint_path or CONFIG.deployment.model_checkpoint
        state = checkpoint_manager.load(resolved_checkpoint_path, map_location=self.device)

        # Model identity, so a forensic report can state exactly which weights produced it.
        self.checkpoint_path = Path(resolved_checkpoint_path)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.checkpoint_epoch = int(state["epoch"])

        use_bilstm = state.get("extra", {}).get("use_bilstm", CONFIG.model.use_bilstm)
        self.model = build_model(use_bilstm=use_bilstm)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.feature_extractor = FeatureExtractor(image_size=feature_image_size)
        self.explainer = GradCAMExplainer(self.model)

        logger.info(
            "InferenceService ready: device=%s, use_bilstm=%s, checkpoint_epoch=%d.",
            self.device,
            use_bilstm,
            state["epoch"],
        )

    def predict(self, audio_path: Path) -> ExplanationResult:
        """Predict and explain a single audio file.

        Args:
            audio_path: Path to a raw audio file on disk (any supported format/length).

        Returns:
            An :class:`explainability.explain.ExplanationResult`.

        Raises:
            ValueError: If the audio file fails validation.
        """
        return explain_audio_file(
            audio_path,
            model=self.model,
            device=self.device,
            feature_extractor=self.feature_extractor,
            explainer=self.explainer,
        )

    def explain_waveform(self, raw_audio: np.ndarray, sample_rate: int, output_stem: str) -> ExplanationResult:
        """Predict and explain an in-memory waveform (e.g. one window of a long recording).

        Args:
            raw_audio: Raw samples, mono or ``(frames, channels)``.
            sample_rate: Sample rate of ``raw_audio``, in Hz.
            output_stem: File-name stem for the saved Grad-CAM overlay.

        Returns:
            An :class:`explainability.explain.ExplanationResult`.
        """
        return explain_waveform(
            raw_audio,
            sample_rate,
            model=self.model,
            device=self.device,
            output_stem=output_stem,
            feature_extractor=self.feature_extractor,
            explainer=self.explainer,
        )

    @torch.no_grad()
    def fake_scores(self, processed_windows: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Score many already-preprocessed windows in batches, without Grad-CAM.

        Args:
            processed_windows: Waveforms already passed through
                :func:`preprocessing.transforms.preprocess_audio` (fixed length,
                ``CONFIG.audio.sample_rate``).
            batch_size: Windows per forward pass.

        Returns:
            Array of shape ``(len(processed_windows),)`` holding each window's
            probability of being synthetic.
        """
        self.model.eval()
        fake_index = LABEL_TO_INDEX["fake"]
        scores: list[np.ndarray] = []
        for start in range(0, len(processed_windows), batch_size):
            features = np.stack(
                [
                    self.feature_extractor.extract_array(window, CONFIG.audio.sample_rate)
                    for window in processed_windows[start : start + batch_size]
                ]
            )
            batch = torch.from_numpy(features).float().unsqueeze(1).to(self.device)
            probabilities = torch.softmax(self.model(batch), dim=1)[:, fake_index]
            scores.append(probabilities.cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
