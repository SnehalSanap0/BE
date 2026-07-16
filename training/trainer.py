"""Training loop orchestration: the Trainer class ties every Phase 5 module together.

Single responsibility: run epochs. Everything it needs (optimizer/scheduler
construction, loss, batch metrics, early stopping, checkpointing, full
validation metrics) is delegated to the corresponding single-purpose module;
this class only sequences them and owns epoch-level state (current epoch,
best metric value).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import CONFIG, TrainingConfig
from evaluation.evaluator import Evaluator
from models.efficientnet_bilstm import EfficientNetBiLSTM
from training.callbacks import EarlyStopping
from training.checkpoint import CheckpointManager
from training.loss import build_loss_fn
from training.metrics import AverageMeter, compute_batch_accuracy
from training.optimizer import build_optimizer, build_scheduler
from utils.logger import get_logger

logger = get_logger(__name__)


def _monitored_metric_key(checkpoint_metric: str) -> str:
    """Strip a ``val_`` prefix (if present) to get the raw metrics-dict key.

    Args:
        checkpoint_metric: Config value such as ``"val_f1"`` or ``"f1"``.

    Returns:
        The bare metric key as produced by :func:`evaluation.metrics.compute_metrics`.
    """
    prefix = "val_"
    return checkpoint_metric[len(prefix) :] if checkpoint_metric.startswith(prefix) else checkpoint_metric


class Trainer:
    """Runs the full train/validate/checkpoint/early-stop loop for a classification model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config: TrainingConfig | None = None,
        checkpoint_dir: Path | None = None,
        log_dir: Path | None = None,
    ) -> None:
        """Initialize the trainer and optionally resume from a previous checkpoint.

        Args:
            model: The classification model to train (from :func:`models.model_factory.build_model`).
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
            training_config: Training hyperparameters. Defaults to ``CONFIG.training``.
            checkpoint_dir: Directory to save/load checkpoints. Defaults to
                ``CONFIG.paths.checkpoint_dir``.
            log_dir: Directory for TensorBoard event files. Defaults to
                ``CONFIG.paths.log_dir / "tensorboard"``.
        """
        self.config = training_config or CONFIG.training
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = build_optimizer(self.model.parameters(), self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config, self.config.num_epochs)
        self.loss_fn = build_loss_fn()

        self.checkpoint_manager = CheckpointManager(checkpoint_dir or CONFIG.paths.checkpoint_dir)
        self.monitored_metric_key = _monitored_metric_key(self.config.checkpoint_metric)
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode="max",
        )

        self.writer = SummaryWriter(log_dir=str(log_dir or CONFIG.paths.log_dir / "tensorboard"))

        self.use_amp = self.config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)

        self.start_epoch = 0
        self.best_metric_value = float("-inf")

        if self.config.resume:
            self._resume_if_available()

        logger.info(
            "Trainer initialized on device=%s, mixed_precision=%s, monitored_metric=%s.",
            self.device,
            self.use_amp,
            self.monitored_metric_key,
        )

    def _resume_if_available(self) -> None:
        """Load the latest checkpoint into model/optimizer/scheduler state, if one exists.

        Returns:
            None.
        """
        state = self.checkpoint_manager.load_latest(map_location=self.device)
        if state is None:
            logger.info("No existing checkpoint found; starting training from scratch.")
            return

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        self.start_epoch = state["epoch"] + 1
        self.best_metric_value = state["best_metric_value"]
        logger.info("Resumed training from epoch %d (best %s so far: %.6f).", self.start_epoch, self.monitored_metric_key, self.best_metric_value)

    def _train_one_epoch(self) -> dict[str, float]:
        """Run one full pass over the training DataLoader with backpropagation.

        Returns:
            Dict with keys ``loss`` and ``accuracy``, averaged over the epoch.
        """
        self.model.train()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        progress = tqdm(self.train_loader, desc="Train", leave=False)
        for features, labels, _speakers, _languages in progress:
            features = features.to(self.device)
            labels = labels.to(self.device)
            batch_size = labels.size(0)

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(features)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(compute_batch_accuracy(logits, labels), batch_size)
            progress.set_postfix(loss=loss_meter.average, accuracy=accuracy_meter.average)

        return {"loss": loss_meter.average, "accuracy": accuracy_meter.average}

    def _validate(self) -> dict[str, object]:
        """Run one full evaluation pass over the validation DataLoader.

        Returns:
            The metrics dict produced by :func:`evaluation.metrics.compute_metrics`.
        """
        evaluator = Evaluator(self.model, self.val_loader, self.device)
        return evaluator.evaluate()

    def fit(self, num_epochs: int | None = None) -> None:
        """Run the full training loop: train, validate, checkpoint, and possibly early-stop each epoch.

        Args:
            num_epochs: Total epochs to train for (counting from epoch 0,
                not from any resumed epoch). Defaults to ``self.config.num_epochs``.

        Returns:
            None.
        """
        num_epochs = num_epochs or self.config.num_epochs

        if self.start_epoch >= num_epochs:
            logger.info("Resumed epoch %d already reached target %d; nothing to do.", self.start_epoch, num_epochs)
            self.writer.close()
            return

        for epoch in range(self.start_epoch, num_epochs):
            train_metrics = self._train_one_epoch()
            val_metrics = self._validate()

            if self.scheduler is not None:
                self.scheduler.step()

            self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            for key, value in val_metrics.items():
                if key != "confusion_matrix":
                    self.writer.add_scalar(f"val/{key}", value, epoch)

            monitored_value = val_metrics[self.monitored_metric_key]
            is_best = monitored_value > self.best_metric_value
            if is_best:
                self.best_metric_value = monitored_value

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_metric_value=self.best_metric_value,
                is_best=is_best,
                extra={"use_bilstm": isinstance(self.model, EfficientNetBiLSTM)},
            )

            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_%s=%.4f%s",
                epoch + 1,
                num_epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                self.monitored_metric_key,
                monitored_value,
                " (best)" if is_best else "",
            )

            if self.early_stopping.step(monitored_value):
                break

        self.writer.close()
