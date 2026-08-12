"""Leakage-aware train/val/test split assignment, written to metadata.csv.

Single responsibility: decide which split (``train``/``val``/``test``) each
metadata row belongs to, and persist that decision. This runs once after
feature extraction and before training. It is idempotent: rows that already
have a non-empty ``split`` are left untouched, so re-running after ingesting
additional clips only assigns splits to the new arrivals instead of
reshuffling everything already trained/evaluated on.

For transcript-paired corpora, :func:`assign_grouped_splits` must be used so
all clips sharing a transcript stay in the same partition.  The older
label-stratified function remains available for unpaired datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from configs.config import CONFIG
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)


def _stratified_assign(labels: pd.Series, train_ratio: float, val_ratio: float, test_ratio: float, random_seed: int) -> pd.Series:
    """Assign a split label to each row, preserving each label's class proportions.

    Args:
        labels: Series of class labels (e.g. ``"real"``/``"fake"``) indexed
            the same as the rows being split.
        train_ratio: Fraction of each class assigned to ``"train"``.
        val_ratio: Fraction of each class assigned to ``"val"``.
        test_ratio: Fraction of each class assigned to ``"test"``.
        random_seed: Seed for the shuffle, for reproducibility.

    Returns:
        Series of ``"train"``/``"val"``/``"test"`` strings, same index as ``labels``.
    """
    rng = np.random.default_rng(random_seed)
    assignments = pd.Series(index=labels.index, dtype=object)

    for label_value in labels.unique():
        class_index = labels[labels == label_value].index.to_numpy().copy()
        rng.shuffle(class_index)

        num_train = int(round(len(class_index) * train_ratio))
        num_val = int(round(len(class_index) * val_ratio))

        train_ids = class_index[:num_train]
        val_ids = class_index[num_train : num_train + num_val]
        test_ids = class_index[num_train + num_val :]

        assignments.loc[train_ids] = "train"
        assignments.loc[val_ids] = "val"
        assignments.loc[test_ids] = "test"

        logger.debug(
            "Class '%s': %d train, %d val, %d test.",
            label_value,
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )

    return assignments


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Reject invalid split ratios before mutating metadata."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must be non-negative and sum to 1.0; got {ratios}.")


def _grouped_assign(
    groups: pd.Series,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> pd.Series:
    """Assign whole groups to partitions, preventing group leakage.

    The unit being shuffled is a transcript (or another caller-provided
    group), never an individual clip.  Empty group values are converted to a
    unique row group so unrelated unpaired clips are not collapsed together.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    normalized = groups.fillna("").astype(str).str.strip()
    normalized = normalized.where(normalized != "", "__row_" + groups.index.astype(str))
    unique_groups = normalized.drop_duplicates().to_numpy().copy()
    np.random.default_rng(random_seed).shuffle(unique_groups)

    num_train = int(round(len(unique_groups) * train_ratio))
    num_val = int(round(len(unique_groups) * val_ratio))
    group_splits: dict[str, str] = {}
    for group in unique_groups[:num_train]:
        group_splits[str(group)] = "train"
    for group in unique_groups[num_train : num_train + num_val]:
        group_splits[str(group)] = "val"
    for group in unique_groups[num_train + num_val :]:
        group_splits[str(group)] = "test"
    return normalized.map(group_splits)


def assign_grouped_splits(
    metadata_manager: MetadataManager | None = None,
    group_column: str = "transcript",
    force_reassign: bool = False,
    random_seed: int | None = None,
) -> MetadataManager:
    """Assign leakage-safe splits while keeping every group intact.

    Args:
        metadata_manager: Metadata store to update.
        group_column: Column defining an indivisible group. For paired
            SEA-Spoof this is ``transcript``.
        force_reassign: Rebuild all eligible assignments. This is required
            when replacing a previously row-stratified paired split.
        random_seed: Optional experiment seed; defaults to the config seed.
    """
    manager = metadata_manager or MetadataManager(CONFIG.paths.metadata_file)
    df = manager.dataframe
    if group_column not in df.columns:
        raise ValueError(f"Cannot group splits by missing metadata column '{group_column}'.")

    featured_mask = df["feature_extracted"].astype(bool)
    eligible_mask = featured_mask if force_reassign else featured_mask & (df["split"] == "")
    eligible = df[eligible_mask]
    if eligible.empty:
        logger.info("No feature-extracted rows require grouped split assignment.")
        return manager

    if not force_reassign:
        assigned = df[featured_mask & (df["split"] != "")]
        assigned_groups = set(assigned[group_column].fillna("").astype(str).str.strip()) - {""}
        incoming_groups = set(eligible[group_column].fillna("").astype(str).str.strip()) - {""}
        overlap = assigned_groups & incoming_groups
        if overlap:
            raise ValueError(
                f"{len(overlap)} {group_column} group(s) already have a split while new rows in the same "
                "groups are unassigned; rerun with force_reassign=True."
            )

    assignments = _grouped_assign(
        groups=eligible[group_column],
        train_ratio=CONFIG.dataset.train_ratio,
        val_ratio=CONFIG.dataset.val_ratio,
        test_ratio=CONFIG.dataset.test_ratio,
        random_seed=CONFIG.dataset.random_seed if random_seed is None else random_seed,
    )
    for row_index, split_value in assignments.items():
        manager.update_field(str(df.loc[row_index, "audio_id"]), {"split": split_value})
    manager.save()

    audit = manager.dataframe[featured_mask].groupby(group_column)["split"].nunique()
    leaking_groups = int((audit > 1).sum())
    if leaking_groups:
        raise RuntimeError(f"Grouped split invariant failed: {leaking_groups} group(s) span partitions.")
    logger.info(
        "Assigned %d row(s) by %d intact '%s' group(s): %s.",
        len(assignments), eligible[group_column].nunique(), group_column,
        assignments.value_counts().to_dict(),
    )
    return manager


def assign_splits(metadata_manager: MetadataManager | None = None) -> MetadataManager:
    """Assign train/val/test splits to every feature-extracted row missing one.

    Args:
        metadata_manager: Metadata store to read/update. Defaults to a
            manager bound to ``CONFIG.paths.metadata_file``.

    Returns:
        The :class:`MetadataManager` with splits assigned and persisted.
    """
    manager = metadata_manager or MetadataManager(CONFIG.paths.metadata_file)
    df = manager.dataframe

    eligible_mask = df["feature_extracted"].astype(bool) & (df["split"] == "")
    eligible = df[eligible_mask]

    if eligible.empty:
        logger.info("No feature-extracted rows are missing a split assignment.")
        return manager

    assignments = _stratified_assign(
        labels=eligible["label"],
        train_ratio=CONFIG.dataset.train_ratio,
        val_ratio=CONFIG.dataset.val_ratio,
        test_ratio=CONFIG.dataset.test_ratio,
        random_seed=CONFIG.dataset.random_seed,
    )

    for row_index, split_value in assignments.items():
        audio_id = str(df.loc[row_index, "audio_id"])
        manager.update_field(audio_id, {"split": split_value})

    manager.save()

    counts = assignments.value_counts().to_dict()
    logger.info("Assigned splits to %d row(s): %s", len(assignments), counts)
    return manager


if __name__ == "__main__":
    assign_splits()
