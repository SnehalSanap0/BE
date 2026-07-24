"""Diagnose *why* cross-source generalization varies by rotation: does the
model's learned embedding space separate clips by source (a shortcut) more
than it separates them by label (the intended task)?

Design rationale:
    ``scripts/cross_source_holdout_check.py`` showed generalization is a
    real/fake *source-pairing interaction* (EER 3.7%-40.5% across the 2x2
    factorial), not a fixed property of the model or a single "genuine vs
    shortcut" split. This script inspects *why*, directly: for each rotation's
    trained model, every feature-extracted clip in the full dataset (all four
    sources, not just that rotation's own held-out split) is pushed through
    the backbone to get its 1280-d pooled embedding. If those embeddings
    cluster by ``source_tag`` more tightly than by ``label``, the model is
    keying on recording/channel fingerprints rather than spoofing artifacts,
    and t-SNE/UMAP should visibly show source clusters that straddle both
    labels. Silhouette score quantifies this without relying on eyeballing
    the plot.

    Reuses :func:`scripts.cross_source_holdout_check.infer_source_tag` for
    the same filename -> source_tag recovery used to build the holdout
    splits, so "source" here means exactly the same four groups (Kathbath,
    Common Voice, IndicSynth, SEA-Spoof) analyzed there.

Usage:
    python -m scripts.embedding_analysis --rotation original
    python -m scripts.embedding_analysis --rotation fake_swap_only
    python -m scripts.embedding_analysis --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from configs.config import CONFIG
from dataset.deepfake_dataset import LABEL_TO_INDEX
from models.model_factory import build_model
from scripts.cross_source_holdout_check import infer_source_tag
from training.checkpoint import CheckpointManager
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)

#: Rotation name -> checkpoint directory, matching cross_source_holdout_check.py's layout.
#: "main" is the original pooled-random-split model (checkpoints/best_model.pt).
CHECKPOINTS: dict[str, Path] = {
    "main": CONFIG.paths.checkpoint_dir / "best_model.pt",
    "original": CONFIG.paths.checkpoint_dir / "cross_source_holdout_original" / "best_model.pt",
    "swapped": CONFIG.paths.checkpoint_dir / "cross_source_holdout_swapped" / "best_model.pt",
    "real_swap_only": CONFIG.paths.checkpoint_dir / "cross_source_holdout_real_swap_only" / "best_model.pt",
    "fake_swap_only": CONFIG.paths.checkpoint_dir / "cross_source_holdout_fake_swap_only" / "best_model.pt",
}

#: Cap per source_tag so t-SNE stays fast and classes/sources stay balanced
#: in the plot regardless of each source's raw clip count.
MAX_CLIPS_PER_SOURCE = 400

SOURCE_COLORS: dict[str, str] = {
    "real": "#1f77b4",
    "real_cv_m": "#17becf",
    "real_cv_f": "#9edae5",
    "fake": "#d62728",
    "fake_sea": "#ff9896",
}
LABEL_COLORS: dict[str, str] = {"real": "#1f77b4", "fake": "#d62728"}


def load_embedding_backbone(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained model and return just its backbone (pooled 1280-d features, pre-classifier).

    Args:
        checkpoint_path: Path to a checkpoint written by :class:`training.checkpoint.CheckpointManager`.
        device: Device to load the model onto.

    Returns:
        The checkpoint's ``backbone`` submodule in eval mode, on ``device``.
    """
    checkpoint_manager = CheckpointManager(checkpoint_path.parent)
    state = checkpoint_manager.load(checkpoint_path, map_location=device)
    use_bilstm = state.get("extra", {}).get("use_bilstm", CONFIG.model.use_bilstm)

    model = build_model(use_bilstm=use_bilstm)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    if not hasattr(model, "backbone"):
        raise AttributeError(
            f"Model built with use_bilstm={use_bilstm} has no top-level 'backbone' attribute; "
            "update embedding extraction for this architecture."
        )
    return model.backbone


def sample_balanced_by_source(df, max_per_source: int, seed: int):
    """Cap each source_tag group at ``max_per_source`` rows, shuffled.

    Args:
        df: Metadata DataFrame with a ``source_tag`` column already assigned.
        max_per_source: Maximum rows to keep per distinct ``source_tag``.
        seed: RNG seed for reproducible sampling.

    Returns:
        A row-sampled copy of ``df``, index reset.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for source_tag, group in df.groupby("source_tag"):
        if len(group) > max_per_source:
            group = group.sample(n=max_per_source, random_state=rng.integers(0, 2**31 - 1))
        parts.append(group)
    import pandas as pd

    return pd.concat(parts, ignore_index=True)


@torch.no_grad()
def extract_embeddings(
    backbone: torch.nn.Module, df, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """Forward every row's saved feature array through the backbone.

    Args:
        backbone: Backbone module (from :func:`load_embedding_backbone`).
        df: Metadata rows to embed; must have a ``feature_path`` column.
        device: Device to run inference on.
        batch_size: Rows per forward pass.

    Returns:
        Array of shape ``(len(df), 1280)`` -- one pooled embedding per row, in ``df`` row order.
    """
    embeddings = []
    feature_paths = df["feature_path"].tolist()
    for start in range(0, len(feature_paths), batch_size):
        batch_paths = feature_paths[start : start + batch_size]
        batch = np.stack([np.load(p) for p in batch_paths])
        tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device)
        pooled = backbone(tensor)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def plot_embeddings(coords: np.ndarray, df, rotation: str, output_dir: Path) -> None:
    """Save side-by-side t-SNE scatter plots colored by source_tag and by label.

    Args:
        coords: 2-D t-SNE coordinates, shape ``(n, 2)``.
        df: Metadata rows aligned with ``coords`` (same order), with
            ``source_tag`` and ``label`` columns.
        rotation: Rotation name, used in the title and output filename.
        output_dir: Directory to save the figure under.

    Returns:
        None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for source_tag, color in SOURCE_COLORS.items():
        mask = (df["source_tag"] == source_tag).to_numpy()
        if mask.any():
            axes[0].scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6, color=color, label=source_tag)
    axes[0].set_title(f"[{rotation}] embeddings colored by source")
    axes[0].legend(markerscale=2, fontsize=8)

    for label, color in LABEL_COLORS.items():
        mask = (df["label"] == label).to_numpy()
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6, color=color, label=label)
    axes[1].set_title(f"[{rotation}] embeddings colored by label")
    axes[1].legend(markerscale=2, fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{rotation}_tsne.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved embedding plot to %s", output_path)


def analyze_rotation(rotation: str, device: torch.device, output_dir: Path) -> dict:
    """Run the full embedding diagnostic for one trained checkpoint.

    Args:
        rotation: Key into :data:`CHECKPOINTS`.
        device: Device to run inference on.
        output_dir: Directory to save plots and metrics under.

    Returns:
        Dict of silhouette scores and sample counts for this rotation, also
        written to ``<output_dir>/<rotation>_silhouette.json``.
    """
    checkpoint_path = CHECKPOINTS[rotation]
    logger.info("[%s] loading checkpoint from %s", rotation, checkpoint_path)
    backbone = load_embedding_backbone(checkpoint_path, device)

    manager = MetadataManager(CONFIG.paths.metadata_file)
    df = manager.dataframe
    df = df[df["feature_extracted"].astype(bool)].copy()
    df["source_tag"] = df["filepath"].apply(infer_source_tag)

    df = sample_balanced_by_source(df, MAX_CLIPS_PER_SOURCE, seed=CONFIG.dataset.random_seed)
    logger.info("[%s] embedding %d clips (source_tag counts: %s)", rotation, len(df), df["source_tag"].value_counts().to_dict())

    embeddings = extract_embeddings(backbone, df, device)

    source_labels = df["source_tag"].to_numpy()
    class_labels = df["label"].to_numpy()

    silhouette_by_source = float(silhouette_score(embeddings, source_labels))
    silhouette_by_label = float(silhouette_score(embeddings, class_labels))

    coords = TSNE(
        n_components=2, perplexity=30, init="pca", random_state=CONFIG.dataset.random_seed
    ).fit_transform(embeddings)
    plot_embeddings(coords, df, rotation, output_dir)

    result = {
        "rotation": rotation,
        "num_clips": len(df),
        "silhouette_by_source_tag": silhouette_by_source,
        "silhouette_by_label": silhouette_by_label,
        "shortcut_ratio": silhouette_by_source / silhouette_by_label if silhouette_by_label != 0 else None,
    }
    metrics_path = output_dir / f"{rotation}_silhouette.json"
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(result, file_handle, indent=2)

    logger.info(
        "[%s] silhouette(source)=%.4f silhouette(label)=%.4f (higher source-silhouette relative to "
        "label-silhouette suggests the embedding space separates recording source more cleanly than "
        "real-vs-fake).",
        rotation,
        silhouette_by_source,
        silhouette_by_label,
    )
    return result


def main() -> None:
    """CLI: run the embedding diagnostic for one or all trained checkpoints.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", choices=sorted(CHECKPOINTS), help="Single rotation to analyze.")
    parser.add_argument("--all", action="store_true", help="Analyze every checkpoint in CHECKPOINTS.")
    args = parser.parse_args()

    if not args.rotation and not args.all:
        parser.error("Pass --rotation <name> or --all.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = CONFIG.paths.output_dir / "embedding_analysis"

    rotations = sorted(CHECKPOINTS) if args.all else [args.rotation]
    results = [analyze_rotation(rotation, device, output_dir) for rotation in rotations]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
