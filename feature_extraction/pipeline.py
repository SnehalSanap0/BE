"""Metadata-driven, resumable batch feature extraction orchestrator.

Mirrors :mod:`preprocessing.pipeline`: reads metadata.csv to find which
processed audio files still need features extracted (``processed`` is true
and ``feature_extracted`` is false), drives
:class:`feature_extraction.extractor.FeatureExtractor` over exactly those
rows, and persists progress after every file so an interrupted run resumes
without recomputing completed work.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from configs.config import CONFIG
from feature_extraction.extractor import ExtractionResult, FeatureExtractor
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)


class FeatureExtractionPipeline:
    """Drives batch feature extraction over every unfeatured record in metadata."""

    def __init__(
        self,
        metadata_manager: MetadataManager | None = None,
        extractor: FeatureExtractor | None = None,
        feature_dir: Path | None = None,
        save_every: int = 1,
    ) -> None:
        """Initialize the pipeline.

        Args:
            metadata_manager: Metadata store to read/update. Defaults to a
                manager bound to ``CONFIG.paths.metadata_file``.
            extractor: Single-file extractor to drive. Defaults to a new
                :class:`FeatureExtractor` built from project configuration.
            feature_dir: Root directory feature arrays are written under.
                Defaults to ``CONFIG.paths.feature_dir``.
            save_every: Flush metadata to disk after this many extracted
                files. ``1`` maximizes resume-safety.
        """
        self.metadata_manager = metadata_manager or MetadataManager(CONFIG.paths.metadata_file)
        self.extractor = extractor or FeatureExtractor()
        self.feature_dir = feature_dir or CONFIG.paths.feature_dir
        self.save_every = max(1, save_every)

    def _destination_for(self, audio_id: str, language: str, label: str) -> Path:
        """Compute the feature output path for a given record.

        Args:
            audio_id: Unique identifier of the record.
            language: Language the clip belongs to, used for folder layout.
            label: ``real``/``fake`` label, used for folder layout.

        Returns:
            Destination path under
            ``feature_dir/<feature_type>/<language>/<label>/<audio_id>.npy``.
        """
        return self.feature_dir / self.extractor.feature_type / language / label / f"{audio_id}.npy"

    def run(self) -> dict[str, int]:
        """Extract features for every processed-but-unfeatured record in metadata.

        Returns:
            Summary counts: ``{"extracted": int, "failed": int}``.
        """
        pending_records = self.metadata_manager.get_unfeatured()
        total_pending = len(pending_records)
        logger.info(
            "Feature extraction pipeline starting: %d record(s) pending (%s).",
            total_pending,
            self.extractor.feature_type,
        )

        summary = {"extracted": 0, "failed": 0}

        for offset, (_, record) in enumerate(
            tqdm(pending_records.iterrows(), total=total_pending, desc="Extracting features"), start=1
        ):
            audio_id = str(record["audio_id"])
            input_path = Path(record["processed_path"])
            output_path = self._destination_for(audio_id, str(record["language"]), str(record["label"]))

            result: ExtractionResult = self.extractor.extract_file(input_path, output_path)

            if result.success:
                self.metadata_manager.update_field(
                    audio_id,
                    {
                        "feature_path": str(result.output_path),
                        "feature_extracted": True,
                    },
                )
                summary["extracted"] += 1
            else:
                summary["failed"] += 1
                logger.warning("Record %s failed feature extraction: %s", audio_id, result.error)

            if offset % self.save_every == 0:
                self.metadata_manager.save()

        self.metadata_manager.save()

        logger.info(
            "Feature extraction pipeline finished: %d extracted, %d failed (of %d pending).",
            summary["extracted"],
            summary["failed"],
            total_pending,
        )
        return summary


if __name__ == "__main__":
    pipeline = FeatureExtractionPipeline()
    pipeline.run()
