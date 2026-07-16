"""Single source of truth for dataset state: ``data/metadata/metadata.csv``.

Design rationale:
    Every stage of this project (preprocessing, feature extraction, dataset
    loading, training) reads and writes exactly one artifact to learn what
    work remains: ``metadata.csv``. No stage is allowed to ``rglob`` a data
    folder to discover files — that responsibility belongs solely to
    :mod:`preprocessing.metadata_builder`, which performs the one-time
    bootstrap scan that populates this file in the first place. Centralizing
    reads/writes in :class:`MetadataManager` means every other module gets
    resume-safety and schema consistency for free.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

#: Canonical column schema for metadata.csv, in column order.
METADATA_COLUMNS: tuple[str, ...] = (
    "audio_id",
    "filepath",
    "processed_path",
    "feature_path",
    "label",
    "speaker",
    "language",
    "sample_rate",
    "duration",
    "split",
    "processed",
    "feature_extracted",
)

def _to_bool(value: object) -> bool:
    """Coerce a CSV-round-tripped value (bool, or 'True'/'False' string) to bool.

    Args:
        value: A Python ``bool``, or a string as read back from CSV.

    Returns:
        The equivalent boolean value.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


_DEFAULTS: dict[str, object] = {
    "processed_path": "",
    "feature_path": "",
    "speaker": "unknown",
    "split": "",
    "processed": False,
    "feature_extracted": False,
}


class MetadataManager:
    """Reads, mutates and persists ``metadata.csv``.

    This class owns the in-memory :class:`pandas.DataFrame` representation
    of the metadata table for its lifetime. Callers should create one
    instance per script/process, call :meth:`load` (or rely on the
    constructor auto-loading), mutate it via the provided methods, and call
    :meth:`save` to flush changes to disk.

    Attributes:
        metadata_path: Absolute path to the CSV file this instance manages.
    """

    def __init__(self, metadata_path: Path) -> None:
        """Initialize the manager and load an existing CSV if present.

        Args:
            metadata_path: Absolute path to ``metadata.csv``.
        """
        self.metadata_path = metadata_path
        self._df: pd.DataFrame = self._load_or_create()

    def _load_or_create(self) -> pd.DataFrame:
        """Load the metadata CSV from disk, or create an empty schema.

        Returns:
            A DataFrame conforming to :data:`METADATA_COLUMNS`.
        """
        if self.metadata_path.exists():
            try:
                df = pd.read_csv(
                    self.metadata_path,
                    dtype={
                        "audio_id": str,
                        "filepath": str,
                        "processed_path": str,
                        "feature_path": str,
                        "label": str,
                        "speaker": str,
                        "language": str,
                        "split": str,
                    },
                    keep_default_na=False,
                    na_values={"sample_rate": [""], "duration": [""]},
                )
            except pd.errors.EmptyDataError:
                logger.warning("Metadata file %s is empty; creating fresh schema.", self.metadata_path)
                return pd.DataFrame(columns=METADATA_COLUMNS)

            missing_columns = set(METADATA_COLUMNS) - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Metadata file {self.metadata_path} is missing required columns: {missing_columns}"
                )

            df["processed"] = df["processed"].apply(_to_bool)
            df["feature_extracted"] = df["feature_extracted"].apply(_to_bool)

            logger.info("Loaded metadata with %d records from %s", len(df), self.metadata_path)
            return df[list(METADATA_COLUMNS)]

        logger.info("No metadata file found at %s; starting with an empty schema.", self.metadata_path)
        return pd.DataFrame(columns=METADATA_COLUMNS)

    def save(self) -> None:
        """Persist the current in-memory table to ``metadata_path``.

        Returns:
            None.
        """
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.metadata_path, index=False)
        logger.debug("Saved metadata (%d records) to %s", len(self._df), self.metadata_path)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying metadata table.

        Returns:
            A :class:`pandas.DataFrame` copy so callers cannot mutate
            internal state without going through this class's methods.
        """
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)

    def upsert_records(self, records: list[dict[str, object]]) -> None:
        """Insert new rows or update existing ones, keyed by ``audio_id``.

        Args:
            records: List of row dicts. Missing optional columns are filled
                with schema defaults; ``audio_id`` is required in every record.

        Returns:
            None.

        Raises:
            KeyError: If any record is missing the required ``audio_id`` key.
        """
        if not records:
            return

        for record in records:
            if "audio_id" not in record:
                raise KeyError("Each metadata record must include 'audio_id'.")
            for column, default in _DEFAULTS.items():
                record.setdefault(column, default)

        new_df = pd.DataFrame(records)[list(METADATA_COLUMNS)]

        if self._df.empty:
            self._df = new_df
        else:
            self._df = (
                pd.concat([self._df, new_df], ignore_index=True)
                .drop_duplicates(subset="audio_id", keep="last")
                .reset_index(drop=True)
            )

        logger.info("Upserted %d record(s); table now has %d rows.", len(records), len(self._df))

    def update_field(self, audio_id: str, field_updates: dict[str, object]) -> None:
        """Update one or more fields for a single existing record.

        Args:
            audio_id: Identifier of the record to update.
            field_updates: Mapping of column name to new value.

        Returns:
            None.

        Raises:
            KeyError: If ``audio_id`` does not exist in the table.
        """
        mask = self._df["audio_id"] == audio_id
        if not mask.any():
            raise KeyError(f"audio_id '{audio_id}' not found in metadata.")

        for column, value in field_updates.items():
            self._df.loc[mask, column] = value

    def get_unprocessed(self) -> pd.DataFrame:
        """Return records that have not yet been through preprocessing.

        Returns:
            Subset of the metadata table where ``processed`` is falsy.
        """
        return self._df[self._df["processed"].astype(bool) == False]  # noqa: E712

    def get_unfeatured(self) -> pd.DataFrame:
        """Return processed records that have not yet had features extracted.

        Returns:
            Subset of the metadata table where ``processed`` is true and
            ``feature_extracted`` is falsy.
        """
        processed_mask = self._df["processed"].astype(bool) == True  # noqa: E712
        unfeatured_mask = self._df["feature_extracted"].astype(bool) == False  # noqa: E712
        return self._df[processed_mask & unfeatured_mask]

    def get_by_split(self, split: str) -> pd.DataFrame:
        """Return records belonging to a given dataset split.

        Args:
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            Subset of the metadata table matching the requested split.
        """
        return self._df[self._df["split"] == split]
