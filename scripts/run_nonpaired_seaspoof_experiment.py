"""Run the E1 control: ordinary (non-paired), count-matched SEA-Spoof training.

Mirrors ``scripts/run_paired_seaspoof_experiment.py`` (the E2 transcript-paired
system) with matching sources, counts, model, and training budget — the only
difference is ``ingestion.allow_pairing: false`` in
``configs/config.paper_seaspoof_e1.yaml``, so real/fake clips are sampled
independently instead of matched by transcript, and splits are assigned by
ordinary label-stratified sampling (:func:`dataset.split_assigner.assign_splits`)
instead of transcript grouping. This is the control condition needed to
isolate whether transcript pairing itself improves cross-source
generalization, versus sample count or pipeline differences alone.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("DEEPFAKE_CONFIG_PATH", str(ROOT_DIR / "configs" / "config.paper_seaspoof_e1.yaml"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configs.config import CONFIG  # noqa: E402
from dataset.dataloader import build_dataloader  # noqa: E402
from dataset.split_assigner import assign_splits  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from feature_extraction.pipeline import FeatureExtractionPipeline  # noqa: E402
from models.model_factory import build_model  # noqa: E402
from preprocessing.hf_ingestion import ingest_all  # noqa: E402
from preprocessing.metadata_builder import build_initial_metadata  # noqa: E402
from preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from utils.metadata_manager import MetadataManager  # noqa: E402


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=CONFIG.dataset.random_seed)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-ingestion", action="store_true")
    args = parser.parse_args()
    set_reproducible_seed(args.seed)

    if not args.skip_ingestion:
        ingest_all()
    manager = build_initial_metadata(MetadataManager(CONFIG.paths.metadata_file))
    PreprocessingPipeline(metadata_manager=manager).run(bootstrap_if_empty=False)
    manager = MetadataManager(CONFIG.paths.metadata_file)
    FeatureExtractionPipeline(metadata_manager=manager).run()
    manager = MetadataManager(CONFIG.paths.metadata_file)
    manager = assign_splits(manager)

    train_loader = build_dataloader("train", metadata_manager=manager)
    val_loader = build_dataloader("val", metadata_manager=manager)
    test_loader = build_dataloader("test", metadata_manager=manager)
    run_checkpoint_dir = CONFIG.paths.checkpoint_dir / f"seed_{args.seed}"
    run_log_dir = CONFIG.paths.log_dir / f"seed_{args.seed}"
    model = build_model()
    trainer = Trainer(
        model, train_loader, val_loader,
        checkpoint_dir=run_checkpoint_dir, log_dir=run_log_dir,
    )
    trainer.fit(num_epochs=args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = trainer.checkpoint_manager.load(run_checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    metrics = Evaluator(model, test_loader, device).evaluate()
    result = {
        "status": "completed",
        "seed": args.seed,
        "config_path": os.environ["DEEPFAKE_CONFIG_PATH"],
        "checkpoint": str(run_checkpoint_dir / "best_model.pt"),
        "test_metrics": metrics,
    }
    result_path = CONFIG.paths.output_dir / f"in_domain_seed_{args.seed}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
