"""Stratified train/val/test split assignment, written back to metadata.csv.

Single responsibility: decide which split (``train``/``val``/``test``) each
metadata row belongs to, and persist that decision. This runs once after
feature extraction and before training. It is idempotent: rows that already
have a non-empty ``split`` are left untouched, so re-running after ingesting
additional clips only assigns splits to the new arrivals instead of
reshuffling everything already trained/evaluated on.

Known limitation:
    Splits are stratified by ``label`` only, not by ``speaker``. The current
    ingestion path (:mod:`preprocessing.hf_ingestion`) does not capture the
    upstream speaker identifiers, so every row's ``speaker`` field is
    ``"unknown"`` and cannot be used to guarantee the same speaker never
    appears in both train and test. For a stronger anti-leakage guarantee,
    ingestion would need to persist ``speaker_id`` and this module would
    group by it before splitting.
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
