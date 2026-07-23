"""Cross-source generalization check: does the model learn spoofing artifacts,
or just the recording/channel fingerprint of each source dataset?

Design rationale:
    ``dataset.split_assigner`` splits train/val/test *randomly across the
    pooled clips*, so every source dataset (Kathbath, Common Voice,
    IndicSynth, SEA-Spoof) appears in all three splits. A model can then hit
    near-perfect metrics by learning "which recording chain is this," which
    generalizes fine within this benchmark but says nothing about detecting
    spoofed speech in general.

    This script builds an alternate split instead, held out *by source*. The
    project has exactly two source datasets per class — real: Kathbath (A)
    vs Common Voice (B); fake: IndicSynth (A) vs SEA-Spoof (B) — so the four
    rotations below form a complete 2x2 factorial over which real-source and
    which fake-source the model trains on, always testing on the
    complementary pair. This isolates whether an asymmetry in cross-source
    generalization comes from the real side, the fake side, or both:

      - ``original``:        train real=A(Kathbath)     fake=A(IndicSynth)  | test real=B(CV)       fake=B(SEA-Spoof)
      - ``swapped``:         train real=B(CV)            fake=B(SEA-Spoof)  | test real=A(Kathbath)  fake=A(IndicSynth)
      - ``real_swap_only``:  train real=B(CV)            fake=A(IndicSynth) | test real=A(Kathbath)  fake=B(SEA-Spoof)
      - ``fake_swap_only``:  train real=A(Kathbath)      fake=B(SEA-Spoof)  | test real=B(CV)        fake=A(IndicSynth)

    ``original`` and ``swapped`` were run first (see docs/research_log.md):
    the reverse pairing collapsed to near-chance accuracy at the default
    threshold despite a still-reasonable ROC-AUC, suggesting a domain-shift
    calibration failure rather than zero learned signal. ``real_swap_only``
    and ``fake_swap_only`` hold one side fixed at a time to find out whether
    that collapse traces back to the real-source pairing (Kathbath vs
    Common Voice's differing channel/noise characteristics), the
    fake-source pairing (IndicSynth's single TTS system vs SEA-Spoof's many
    systems), or requires both to flip at once.

    If test EER stays near the pooled-split result, the model is learning
    genuine spoof-vs-real cues. If it jumps substantially, the pooled
    near-zero EER was a channel/source shortcut.

    Reuses every existing pipeline component (MetadataManager, build_model,
    build_dataloader, Trainer, Evaluator) exactly as they're designed to be
    reused — no preprocessing or feature extraction is repeated, since this
    only re-labels the ``split`` column of a *copy* of metadata.csv pointing
    at the same already-processed/-featured files. The real
    ``data/metadata/metadata.csv`` and ``checkpoints/`` are never touched;
    each rotation writes to its own
    ``data/metadata/metadata_cross_source_holdout_<rotation>.csv``,
    ``checkpoints/cross_source_holdout_<rotation>/`` and
    ``outputs/cross_source_holdout_<rotation>_metrics.json`` so no rotation's
    results ever overwrite another's.

Usage:
    python -m scripts.cross_source_holdout_check --rotation original --epochs 15
    python -m scripts.cross_source_holdout_check --rotation swapped --epochs 15
    python -m scripts.cross_source_holdout_check --rotation real_swap_only --epochs 15
    python -m scripts.cross_source_holdout_check --rotation fake_swap_only --epochs 15
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs.config import CONFIG
from dataset.dataloader import build_dataloader
from evaluation.evaluator import Evaluator
from models.model_factory import build_model
from training.trainer import Trainer
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)

SOURCE_TAG_PATTERN = re.compile(r"([a-z_]+?)_\d{6}")

ALL_SOURCES = {"real", "fake", "real_cv_m", "real_cv_f", "fake_sea"}

#: The two real-source and two fake-source options this project has.
REAL_A = {"real"}  # Kathbath
REAL_B = {"real_cv_m", "real_cv_f"}  # Common Voice
FAKE_A = {"fake"}  # IndicSynth
FAKE_B = {"fake_sea"}  # SEA-Spoof

#: Named train/test source pairings. Each value is (train_sources, holdout_sources).
#: The four rotations form a complete 2x2 factorial over (real source, fake
#: source) used for training, always testing on the complementary pair.
ROTATIONS: dict[str, tuple[set[str], set[str]]] = {
    "original": (REAL_A | FAKE_A, REAL_B | FAKE_B),
    "swapped": (REAL_B | FAKE_B, REAL_A | FAKE_A),
    "real_swap_only": (REAL_B | FAKE_A, REAL_A | FAKE_B),
    "fake_swap_only": (REAL_A | FAKE_B, REAL_B | FAKE_A),
}

VAL_FRACTION_OF_TRAIN_SOURCES = 0.1


def infer_source_tag(filepath: str) -> str:
    """Recover a row's ingestion source_tag from its raw filename prefix.

    Args:
        filepath: The row's ``filepath`` column (path to the raw clip).

    Returns:
        The source_tag prefix (e.g. ``"real_cv_f"``, ``"fake_sea"``), or the
        full stem if the filename doesn't match the expected pattern.
    """
    name = Path(filepath).stem
    match = SOURCE_TAG_PATTERN.match(name)
    return match.group(1) if match else name


def build_holdout_metadata(
    train_sources: set[str], holdout_sources: set[str], metadata_path: Path
) -> MetadataManager:
    """Create a source-held-out metadata copy, re-splitting by source_tag.

    Args:
        train_sources: source_tags allowed into train/val.
        holdout_sources: source_tags reserved entirely for test.
        metadata_path: Destination path for this rotation's metadata copy.

    Returns:
        A :class:`MetadataManager` bound to ``metadata_path`` with every
        feature-extracted row reassigned to ``train``/``val``/``test`` by
        which source dataset it came from, rather than at random.
    """
    source_manager = MetadataManager(CONFIG.paths.metadata_file)
    df = source_manager.dataframe
    df = df[df["feature_extracted"].astype(bool)].copy()
    df["source_tag"] = df["filepath"].apply(infer_source_tag)

    unknown_sources = set(df["source_tag"].unique()) - train_sources - holdout_sources
    if unknown_sources:
        raise ValueError(
            f"Unrecognized source_tag(s) {unknown_sources} found in metadata.csv; "
            "update ROTATIONS in scripts/cross_source_holdout_check.py to account "
            "for every ingestion source before running this check."
        )

    rng = np.random.default_rng(CONFIG.dataset.random_seed)
    split_assignments = pd.Series(index=df.index, dtype=object)

    train_source_mask = df["source_tag"].isin(train_sources)
    for label_value in df.loc[train_source_mask, "label"].unique():
        class_index = df.loc[train_source_mask & (df["label"] == label_value)].index.to_numpy().copy()
        rng.shuffle(class_index)
        num_val = max(1, int(round(len(class_index) * VAL_FRACTION_OF_TRAIN_SOURCES)))
        split_assignments.loc[class_index[:num_val]] = "val"
        split_assignments.loc[class_index[num_val:]] = "train"

    split_assignments.loc[df["source_tag"].isin(holdout_sources)] = "test"
    df["split"] = split_assignments

    counts = df.groupby(["source_tag", "split"]).size().to_dict()
    logger.info("Cross-source holdout split assignment (source_tag, split) -> count: %s", counts)

    holdout_manager = MetadataManager(metadata_path)
    holdout_manager.upsert_records(df.drop(columns=["source_tag"]).to_dict("records"))
    holdout_manager.save()
    return holdout_manager


def main() -> None:
    """Build the held-out split, train a fresh model on it, and report test metrics.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rotation", choices=sorted(ROTATIONS), default="original",
        help="Which train/test source pairing to run. Default 'original'.",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Max epochs for this diagnostic run (early stopping still applies). Default 15.",
    )
    parser.add_argument(
        "--rebuild-split", action="store_true",
        help="Recompute the holdout split even if this rotation's metadata copy already exists.",
    )
    args = parser.parse_args()

    train_sources, holdout_sources = ROTATIONS[args.rotation]
    metadata_path = CONFIG.paths.metadata_dir / f"metadata_cross_source_holdout_{args.rotation}.csv"
    checkpoint_dir = CONFIG.paths.checkpoint_dir / f"cross_source_holdout_{args.rotation}"
    metrics_path = CONFIG.paths.output_dir / f"cross_source_holdout_{args.rotation}_metrics.json"

    if args.rebuild_split or not metadata_path.exists():
        holdout_manager = build_holdout_metadata(train_sources, holdout_sources, metadata_path)
    else:
        logger.info("Reusing existing holdout metadata at %s (pass --rebuild-split to recompute).", metadata_path)
        holdout_manager = MetadataManager(metadata_path)

    train_loader = build_dataloader("train", metadata_manager=holdout_manager)
    val_loader = build_dataloader("val", metadata_manager=holdout_manager)
    test_loader = build_dataloader("test", metadata_manager=holdout_manager)

    model = build_model()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.fit(num_epochs=args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_state = trainer.checkpoint_manager.load(
        checkpoint_dir / "best_model.pt", map_location=device
    )
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    evaluator = Evaluator(model, test_loader, device)
    metrics = evaluator.evaluate()

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    logger.info(
        "Cross-source holdout test [%s] (unseen sources %s): accuracy=%.4f f1=%.4f roc_auc=%.4f eer=%.4f",
        args.rotation,
        sorted(holdout_sources),
        metrics["accuracy"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["eer"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
