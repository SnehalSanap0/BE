"""Metadata-driven, resumable batch preprocessing orchestrator.

Single responsibility: decide *which* files still need processing (by
reading ``metadata.csv``, never the filesystem) and drive
:class:`preprocessing.processor.AudioProcessor` over exactly those files,
persisting progress after every file so an interrupted run can resume
without reprocessing completed work.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from configs.config import CONFIG
from preprocessing.metadata_builder import build_initial_metadata
from preprocessing.processor import AudioProcessor, ProcessingResult
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Drives batch preprocessing of every unprocessed file listed in metadata.

    The pipeline is resumable by construction: it only ever asks the
    :class:`MetadataManager` for rows where ``processed`` is falsy, and it
    flushes the metadata table to disk after each file. If the process is
    killed mid-run, restarting simply re-queries for the (now smaller) set
    of unprocessed rows.
    """

    def __init__(
        self,
        metadata_manager: MetadataManager | None = None,
        processor: AudioProcessor | None = None,
        processed_dir: Path | None = None,
        save_every: int = 1,
    ) -> None:
        """Initialize the pipeline.

        Args:
            metadata_manager: Metadata store to read/update. Defaults to a
                manager bound to ``CONFIG.paths.metadata_file``.
            processor: Single-file processor to drive. Defaults to a new
                :class:`AudioProcessor` built from project configuration.
            processed_dir: Root directory processed files are written under.
                Defaults to ``CONFIG.paths.processed_dir``.
            save_every: Flush metadata to disk after this many processed
                files. ``1`` maximizes resume-safety; higher values trade
                some resume granularity for fewer disk writes on large runs.
        """
        self.metadata_manager = metadata_manager or MetadataManager(CONFIG.paths.metadata_file)
        self.processor = processor or AudioProcessor()
        self.processed_dir = processed_dir or CONFIG.paths.processed_dir
        self.save_every = max(1, save_every)

    def _destination_for(self, audio_id: str, language: str, label: str) -> Path:
        """Compute the processed-audio output path for a given record.

        Args:
            audio_id: Unique identifier of the record.
            language: Language the clip belongs to, used for folder layout.
            label: ``real``/``fake`` label, used for folder layout.

        Returns:
            Destination path under ``processed_dir/<language>/<label>/<audio_id>.wav``.
        """
        return self.processed_dir / language / label / f"{audio_id}.wav"

    def run(self, bootstrap_if_empty: bool = True) -> dict[str, int]:
        """Process every unprocessed record listed in metadata.

        Args:
            bootstrap_if_empty: If the metadata table is empty, run
                :func:`preprocessing.metadata_builder.build_initial_metadata`
                first to discover raw files before processing.

        Returns:
            Summary counts: ``{"processed": int, "failed": int, "skipped": int}``.
        """
        if bootstrap_if_empty and len(self.metadata_manager) == 0:
            logger.info("Metadata table is empty; running initial bootstrap scan.")
            self.metadata_manager = build_initial_metadata(self.metadata_manager)

        pending_records = self.metadata_manager.get_unprocessed()

        total_pending = len(pending_records)
        logger.info("Preprocessing pipeline starting: %d unprocessed record(s).", total_pending)

        summary = {"processed": 0, "failed": 0, "skipped": 0}

        for offset, (_, record) in enumerate(tqdm(pending_records.iterrows(), total=total_pending, desc="Preprocessing"), start=1):
            audio_id = str(record["audio_id"])
            input_path = Path(record["filepath"])
            output_path = self._destination_for(audio_id, str(record["language"]), str(record["label"]))

            result: ProcessingResult = self.processor.process_file(input_path, output_path)

            if result.success:
                self.metadata_manager.update_field(
                    audio_id,
                    {
                        "processed_path": str(result.output_path),
                        "sample_rate": result.sample_rate,
                        "duration": result.duration_seconds,
                        "processed": True,
                    },
                )
                summary["processed"] += 1
            else:
                summary["failed"] += 1
                logger.warning("Record %s failed preprocessing: %s", audio_id, result.error)

            if offset % self.save_every == 0:
                self.metadata_manager.save()

        self.metadata_manager.save()

        logger.info(
            "Preprocessing pipeline finished: %d processed, %d failed (of %d pending).",
            summary["processed"],
            summary["failed"],
            total_pending,
        )
        return summary


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
