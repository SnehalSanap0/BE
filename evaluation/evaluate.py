"""CLI entrypoint: evaluate a trained checkpoint on the test split.

Usage:
    python -m evaluation.evaluate
"""

from __future__ import annotations

import json

import torch

from configs.config import CONFIG
from dataset.dataloader import build_dataloader
from evaluation.evaluator import Evaluator
from models.model_factory import build_model
from training.checkpoint import BEST_CHECKPOINT_FILENAME, CheckpointManager
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Load the best checkpoint, run it over the test split, and save a metrics report.

    Returns:
        None.

    Raises:
        FileNotFoundError: If no best checkpoint exists yet (train a model first).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_manager = CheckpointManager(CONFIG.paths.checkpoint_dir)
    checkpoint_path = CONFIG.paths.checkpoint_dir / BEST_CHECKPOINT_FILENAME
    state = checkpoint_manager.load(checkpoint_path, map_location=device)

    # The checkpoint records which architecture it was trained as (Trainer
    # persists this in `extra`), so evaluation always rebuilds the matching
    # model even if CONFIG.model.use_bilstm has since changed.
    use_bilstm = state.get("extra", {}).get("use_bilstm", CONFIG.model.use_bilstm)
    model = build_model(use_bilstm=use_bilstm)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    test_loader = build_dataloader("test")
    evaluator = Evaluator(model, test_loader, device)
    metrics = evaluator.evaluate()

    report_path = CONFIG.paths.output_dir / "test_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)
    logger.info("Saved test metrics report to %s", report_path)

    confusion_matrix_path = CONFIG.paths.output_dir / "confusion_matrix.png"
    evaluator.save_confusion_matrix(metrics, confusion_matrix_path)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
