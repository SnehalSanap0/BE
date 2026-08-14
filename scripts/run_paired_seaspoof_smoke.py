"""Run a fast, isolated paired Hindi SEA-Spoof smoke test.

This is a pipeline check, not a research run. It uses
``configs/config.smoke_seaspoof.yaml`` by default, which writes all smoke
artifacts into dedicated ``smoke_seaspoof_paired`` subfolders and trains on a
tiny transcript-matched SEA-Spoof Hindi bonafide/spoof subset.

Usage:
    python -m scripts.run_paired_seaspoof_smoke
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = ROOT_DIR / "configs" / "config.smoke_seaspoof.yaml"
os.environ.setdefault("DEEPFAKE_CONFIG_PATH", str(SMOKE_CONFIG_PATH))

import torch  # noqa: E402

from configs.config import CONFIG  # noqa: E402
from dataset.dataloader import build_dataloader  # noqa: E402
from dataset.split_assigner import assign_grouped_splits  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from feature_extraction.pipeline import FeatureExtractionPipeline  # noqa: E402
from models.model_factory import build_model  # noqa: E402
from preprocessing.hf_ingestion import ingest_all  # noqa: E402
from preprocessing.metadata_builder import build_initial_metadata  # noqa: E402
from preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.metadata_manager import MetadataManager  # noqa: E402

logger = get_logger(__name__)


def _source_count(label: str, source_tag: str) -> int:
    """Count locally available raw clips for one smoke source."""
    source_dir = CONFIG.paths.raw_dir / CONFIG.ingestion.language / label
    if not source_dir.exists():
        return 0
    return sum(1 for path in source_dir.iterdir() if path.is_file() and path.name.startswith(f"{source_tag}_"))


def _target_pairs() -> int:
    """Return the configured paired sample count."""
    return min(CONFIG.ingestion.real[0].target_count, CONFIG.ingestion.fake[0].target_count)


def _ensure_raw_pairs() -> None:
    """Use existing smoke raw clips if present, otherwise ingest paired SEA-Spoof."""
    target_pairs = _target_pairs()
    real_count = _source_count("real", "real_sea")
    fake_count = _source_count("fake", "fake_sea")
    existing_pairs = min(real_count, fake_count)

    if existing_pairs >= target_pairs:
        logger.info("Using %d existing local paired SEA-Spoof smoke clip(s).", existing_pairs)
        return

    logger.info(
        "Need %d paired clip(s), found %d. Starting paired SEA-Spoof ingestion.",
        target_pairs,
        existing_pairs,
    )
    ingest_all()


def main() -> None:
    """Run ingestion, preprocessing, features, split assignment, training, and test evaluation."""
    logger.info("Smoke config: %s", os.environ["DEEPFAKE_CONFIG_PATH"])
    logger.info("Smoke artifacts root: %s", CONFIG.paths.data_dir)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    _ensure_raw_pairs()

    manager = MetadataManager(CONFIG.paths.metadata_file)
    manager = build_initial_metadata(manager)
    PreprocessingPipeline(metadata_manager=manager).run(bootstrap_if_empty=False)

    manager = MetadataManager(CONFIG.paths.metadata_file)
    FeatureExtractionPipeline(metadata_manager=manager).run()

    manager = MetadataManager(CONFIG.paths.metadata_file)
    # Rebuild old row-stratified smoke splits: paired members must never be
    # separated across train/validation/test.
    manager = assign_grouped_splits(manager, group_column="transcript", force_reassign=True)

    train_loader = build_dataloader("train", metadata_manager=manager)
    val_loader = build_dataloader("val", metadata_manager=manager)
    test_loader = build_dataloader("test", metadata_manager=manager)

    model = build_model()
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader)
    trainer.fit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_state = trainer.checkpoint_manager.load(CONFIG.paths.checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    evaluator = Evaluator(model, test_loader, device)
    metrics = evaluator.evaluate()

    CONFIG.paths.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = CONFIG.paths.output_dir / "smoke_test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    print(json.dumps(metrics, indent=2))
    logger.info("Saved smoke metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
