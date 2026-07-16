"""Single-file audio processing: validate, transform, persist.

Single responsibility: given one raw audio file and a destination path,
produce one processed audio file on disk. This module knows nothing about
metadata.csv, resuming, or iterating over datasets — that orchestration
belongs to :mod:`preprocessing.pipeline`. Keeping this class metadata-unaware
makes it directly reusable from the Flask inference path in Phase 8, where
there is no metadata table at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from configs.config import CONFIG
from preprocessing.transforms import preprocess_audio
from preprocessing.validator import AudioValidator, ValidationResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProcessingResult:
    """Outcome of processing a single audio file.

    Attributes:
        success: Whether processing completed and the output was written.
        output_path: Path the processed audio was written to, if successful.
        sample_rate: Sample rate of the processed output, if successful.
        duration_seconds: Duration of the processed output, if successful.
        error: Failure reason when ``success`` is ``False``.
    """

    success: bool
    output_path: Path | None = None
    sample_rate: int | None = None
    duration_seconds: float | None = None
    error: str = ""


class AudioProcessor:
    """Validates, transforms and persists a single audio file at a time.

    Dependencies (the validator and the transform function) are injected
    through the constructor so this class can be unit-tested with a fake
    validator, and so alternate validation policies can be swapped in
    without modifying this class (Dependency Inversion).
    """

    def __init__(self, validator: AudioValidator | None = None) -> None:
        """Initialize the processor.

        Args:
            validator: Validator used to reject corrupt/unsupported files
                before any processing is attempted. Defaults to a new
                :class:`AudioValidator` built from project configuration.
        """
        self.validator = validator or AudioValidator()

    def process_file(self, input_path: Path, output_path: Path) -> ProcessingResult:
        """Validate, preprocess and save a single audio file.

        Args:
            input_path: Path to the raw audio file to process.
            output_path: Destination path for the processed ``.wav`` file.
                Parent directories are created automatically.

        Returns:
            A :class:`ProcessingResult` describing success or the reason
            for failure. Failures never raise — they are reported so batch
            callers (:mod:`preprocessing.pipeline`) can continue past a
            single bad file.
        """
        validation: ValidationResult = self.validator.validate(input_path)

        if not validation.is_valid:
            logger.warning("Skipping %s: %s", input_path, validation.reason)
            return ProcessingResult(success=False, error=validation.reason)

        try:
            audio, sample_rate = sf.read(str(input_path), dtype="float32", always_2d=False)
            audio = np.asarray(audio).T if audio.ndim > 1 else audio

            processed_audio = preprocess_audio(audio, sample_rate)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), processed_audio, CONFIG.audio.sample_rate)

        except Exception as exc:  # noqa: BLE001 - any failure here is a processing failure, not a crash
            logger.error("Failed to process %s: %s", input_path, exc, exc_info=True)
            return ProcessingResult(success=False, error=str(exc))

        duration_seconds = processed_audio.shape[0] / float(CONFIG.audio.sample_rate)
        logger.info("Processed %s -> %s (%.2fs)", input_path.name, output_path, duration_seconds)

        return ProcessingResult(
            success=True,
            output_path=output_path,
            sample_rate=CONFIG.audio.sample_rate,
            duration_seconds=duration_seconds,
        )
