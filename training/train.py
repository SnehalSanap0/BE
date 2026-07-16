"""CLI entrypoint: train the configured model on the configured dataset splits.

Usage:
    python -m training.train
"""

from __future__ import annotations

from dataset.dataloader import build_dataloader
from models.model_factory import build_model
from training.trainer import Trainer
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Build dataloaders and model per project configuration, then run training.

    Returns:
        None.
    """
    train_loader = build_dataloader("train")
    val_loader = build_dataloader("val")

    model = build_model()
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()
