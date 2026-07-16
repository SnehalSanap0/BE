"""Loads a trained checkpoint once and serves predictions for individual audio files.

Single responsibility: own the model/device/explainer lifecycle. Kept
separate from :mod:`deployment.app` so the Flask layer is pure HTTP routing
— the model is loaded exactly once at service construction, not per request,
and this class is equally usable from a script or test without Flask at all.
"""

from __future__ import annotations

from pathlib import Path

import torch

from configs.config import CONFIG
from explainability.explain import ExplanationResult, explain_audio_file
from explainability.gradcam import GradCAMExplainer
from feature_extraction.extractor import FeatureExtractor
from models.model_factory import build_model
from training.checkpoint import BEST_CHECKPOINT_FILENAME, CheckpointManager
from utils.logger import get_logger

logger = get_logger(__name__)


class InferenceService:
    """Holds one loaded model in memory and predicts+explains audio files against it."""

    def __init__(self, checkpoint_path: Path | None = None, device: torch.device | None = None) -> None:
        """Load the best checkpoint and construct the model, feature extractor and explainer.

        Args:
            checkpoint_path: Path to a checkpoint saved by
                :class:`training.checkpoint.CheckpointManager`. Defaults to
                ``CONFIG.paths.checkpoint_dir / "best_model.pt"``.
            device: Device to run inference on. Defaults to CUDA if available, else CPU.

        Raises:
            FileNotFoundError: If no checkpoint exists at the resolved path
                (train a model first via ``python -m training.train``).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_manager = CheckpointManager(CONFIG.paths.checkpoint_dir)
        resolved_checkpoint_path = checkpoint_path or (CONFIG.paths.checkpoint_dir / BEST_CHECKPOINT_FILENAME)
        state = checkpoint_manager.load(resolved_checkpoint_path, map_location=self.device)

        use_bilstm = state.get("extra", {}).get("use_bilstm", CONFIG.model.use_bilstm)
        self.model = build_model(use_bilstm=use_bilstm)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.feature_extractor = FeatureExtractor()
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
