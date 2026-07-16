"""Optimizer and learning-rate scheduler construction from project configuration.

Single responsibility: translate ``CONFIG.training`` (optimizer name,
learning rate, weight decay, scheduler name) into constructed PyTorch
objects. Kept separate from :mod:`training.trainer` so the dispatch logic
is independently testable and reusable.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR

from configs.config import TrainingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def build_optimizer(parameters: Iterator[nn.Parameter], training_config: TrainingConfig) -> Optimizer:
    """Construct the optimizer specified by ``training_config.optimizer``.

    Args:
        parameters: Model parameters to optimize (typically ``model.parameters()``).
        training_config: Training configuration providing ``optimizer``,
            ``learning_rate`` and ``weight_decay``.

    Returns:
        A constructed :class:`torch.optim.Optimizer`.

    Raises:
        ValueError: If ``training_config.optimizer`` is not recognized.
    """
    optimizer_name = training_config.optimizer.lower()

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=training_config.learning_rate, weight_decay=training_config.weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            parameters, lr=training_config.learning_rate, weight_decay=training_config.weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer '{training_config.optimizer}'. Expected adamw, adam, or sgd.")

    logger.info("Built optimizer '%s' (lr=%g, weight_decay=%g).", optimizer_name, training_config.learning_rate, training_config.weight_decay)
    return optimizer


def build_scheduler(
    optimizer: Optimizer, training_config: TrainingConfig, num_epochs: int
) -> LRScheduler | None:
    """Construct the learning-rate scheduler specified by ``training_config.scheduler``.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        training_config: Training configuration providing ``scheduler``.
        num_epochs: Total planned training epochs, used to size the schedule.

    Returns:
        A constructed learning-rate scheduler, or ``None`` if
        ``training_config.scheduler`` is ``"none"``.

    Raises:
        ValueError: If ``training_config.scheduler`` is not recognized.
    """
    scheduler_name = training_config.scheduler.lower()

    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler '{training_config.scheduler}'. Expected cosine, step, or none.")

    logger.info("Built scheduler '%s' for %d planned epochs.", scheduler_name, num_epochs)
    return scheduler
