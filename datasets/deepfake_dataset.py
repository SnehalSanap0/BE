"""PyTorch Dataset over metadata.csv — never touches the filesystem tree directly.

Single responsibility: given a split name, expose its feature-extracted rows
as ``(feature_tensor, label, speaker, language)`` tuples. Every other Phase
1/2 module treated metadata.csv as authoritative; this class is where that
discipline pays off — construction is a single filtered read of the table,
and every ``__getitem__`` call loads exactly one ``.npy`` file named by the
row's ``feature_path`` column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import CONFIG
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)

#: Canonical label encoding for the binary real-vs-fake classification task.
LABEL_TO_INDEX: dict[str, int] = {"real": 0, "fake": 1}
INDEX_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_INDEX.items()}


class DeepfakeAudioDataset(Dataset):
    """Serves feature-extracted clips belonging to one dataset split.

    Attributes:
        split: The split this dataset instance serves (``"train"``, ``"val"``, or ``"test"``).
    """

    def __init__(
        self,
        split: str,
        metadata_manager: MetadataManager | None = None,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        """Initialize the dataset by filtering metadata down to this split.

        Args:
            split: Which split to serve: ``"train"``, ``"val"``, or ``"test"``.
            metadata_manager: Metadata store to read from. Defaults to a
                manager bound to ``CONFIG.paths.metadata_file``.
            label_to_index: Mapping from string label to integer class index.
                Defaults to :data:`LABEL_TO_INDEX`.

        Raises:
            ValueError: If no feature-extracted rows exist for ``split``.
        """
        self.split = split
        self.label_to_index = label_to_index or LABEL_TO_INDEX

        manager = metadata_manager or MetadataManager(CONFIG.paths.metadata_file)
        split_df = manager.get_by_split(split)
        self._records = split_df[split_df["feature_extracted"].astype(bool)].reset_index(drop=True)

        if len(self._records) == 0:
            raise ValueError(
                f"No feature-extracted records found for split '{split}'. "
                "Run preprocessing.pipeline, feature_extraction.pipeline and "
                "datasets.split_assigner before constructing this dataset."
            )

        logger.info("Loaded %d record(s) for split '%s'.", len(self._records), split)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        """Load one feature array and its associated labels.

        Args:
            index: Row index within this split.

        Returns:
            Tuple of ``(feature_tensor, label_index, speaker, language)`` where
            ``feature_tensor`` has shape ``(1, image_size, image_size)``.

        Raises:
            FileNotFoundError: If the row's ``feature_path`` no longer exists on disk.
        """
        record = self._records.iloc[index]
        feature_path = Path(record["feature_path"])

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file for audio_id '{record['audio_id']}' not found at {feature_path}."
            )

        feature = np.load(feature_path)
        feature_tensor = torch.from_numpy(feature).float().unsqueeze(0)

        label_index = self.label_to_index[str(record["label"])]

        return feature_tensor, label_index, str(record["speaker"]), str(record["language"])
