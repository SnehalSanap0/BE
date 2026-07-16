"""Model/optimizer/scheduler checkpointing, with resume support.

Single responsibility: serialize and restore everything needed to resume
training exactly where it left off (model weights, optimizer and scheduler
state, epoch number, and the best monitored metric seen so far), and
separately track the single best checkpoint for downstream evaluation/deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.logger import get_logger

logger = get_logger(__name__)

LAST_CHECKPOINT_FILENAME = "last_checkpoint.pt"
BEST_CHECKPOINT_FILENAME = "best_model.pt"


class CheckpointManager:
    """Saves and loads training checkpoints under a fixed directory.

    Attributes:
        checkpoint_dir: Directory checkpoints are written to and read from.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints in. Created if missing.
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        epoch: int,
        best_metric_value: float,
        is_best: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save a checkpoint, always as "last", and additionally as "best" if requested.

        Args:
            model: Model whose ``state_dict`` will be saved.
            optimizer: Optimizer whose ``state_dict`` will be saved.
            scheduler: Scheduler whose ``state_dict`` will be saved, if present.
            epoch: Index of the epoch that just completed.
            best_metric_value: Best monitored metric value seen so far
                (across all epochs, not just this one).
            is_best: Whether this epoch produced a new best metric value;
                if ``True``, an additional copy is saved as the best checkpoint.
            extra: Additional JSON/tensor-serializable state to persist
                (e.g. label mapping, feature config snapshot).

        Returns:
            None.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_metric_value": best_metric_value,
            "extra": extra or {},
        }

        last_path = self.checkpoint_dir / LAST_CHECKPOINT_FILENAME
        torch.save(state, last_path)
        logger.info("Saved checkpoint at epoch %d to %s", epoch, last_path)

        if is_best:
            best_path = self.checkpoint_dir / BEST_CHECKPOINT_FILENAME
            torch.save(state, best_path)
            logger.info("New best checkpoint (metric=%.6f) saved to %s", best_metric_value, best_path)

    def load(self, path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """Load a checkpoint file.

        Args:
            path: Path to a checkpoint file previously written by :meth:`save`.
            map_location: Device to map tensors to on load.

        Returns:
            The raw checkpoint state dict.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        state = torch.load(path, map_location=map_location, weights_only=False)
        logger.info("Loaded checkpoint from %s (epoch %d).", path, state["epoch"])
        return state

    def load_latest(self, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
        """Load the most recent checkpoint, if one exists.

        Args:
            map_location: Device to map tensors to on load.

        Returns:
            The checkpoint state dict, or ``None`` if no checkpoint exists yet
            (i.e. this is a fresh training run).
        """
        last_path = self.checkpoint_dir / LAST_CHECKPOINT_FILENAME
        if not last_path.exists():
            return None
        return self.load(last_path, map_location=map_location)
