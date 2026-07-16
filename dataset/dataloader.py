"""DataLoader factory: wires DeepfakeAudioDataset into batched, shuffled iteration.

Single responsibility: translate project configuration
(``CONFIG.training.batch_size``/``num_workers``) plus a split name into a
ready-to-iterate :class:`torch.utils.data.DataLoader`. Kept separate from
:class:`dataset.deepfake_dataset.DeepfakeAudioDataset` so the dataset class
stays a pure data-access object and all batching policy lives in one place.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from configs.config import CONFIG
from dataset.deepfake_dataset import DeepfakeAudioDataset
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)


def build_dataloader(
    split: str,
    metadata_manager: MetadataManager | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    """Construct a DataLoader for one dataset split.

    Args:
        split: Which split to serve: ``"train"``, ``"val"``, or ``"test"``.
        metadata_manager: Metadata store to read from. Defaults to a
            manager bound to ``CONFIG.paths.metadata_file``.
        batch_size: Examples per batch. Defaults to ``CONFIG.training.batch_size``.
        num_workers: Subprocesses used for data loading. Defaults to
            ``CONFIG.training.num_workers``.

    Returns:
        A :class:`torch.utils.data.DataLoader` over the requested split.
        Shuffling and dropping the last incomplete batch are both enabled
        only for the ``"train"`` split; validation/test iterate in order
        over every example exactly once.
    """
    dataset = DeepfakeAudioDataset(split=split, metadata_manager=metadata_manager)

    is_train = split == "train"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size or CONFIG.training.batch_size,
        shuffle=is_train,
        num_workers=num_workers if num_workers is not None else CONFIG.training.num_workers,
        drop_last=is_train,
        pin_memory=True,
    )

    logger.info(
        "Built DataLoader for split '%s': %d examples, batch_size=%d, shuffle=%s.",
        split,
        len(dataset),
        dataloader.batch_size,
        is_train,
    )
    return dataloader
