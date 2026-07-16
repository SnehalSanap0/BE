"""Audio-file validation prior to preprocessing.

Single responsibility: decide whether a raw audio file is safe to feed into
the preprocessing pipeline. This module never mutates audio or metadata —
it only inspects a file and reports a structured verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from configs.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a single audio file.

    Attributes:
        is_valid: Whether the file passed every validation check.
        reason: Human-readable explanation when ``is_valid`` is ``False``,
            empty string otherwise.
        sample_rate: Detected sample rate in Hz, if the file could be read.
        duration_seconds: Detected duration in seconds, if the file could be read.
    """

    is_valid: bool
    reason: str = ""
    sample_rate: int | None = None
    duration_seconds: float | None = None


class AudioValidator:
    """Validates raw audio files against format, integrity and content checks.

    Each check is a small, independently testable method. :meth:`validate`
    composes them in cheapest-first order so that expensive full-file reads
    are skipped once a file has already failed a cheaper check.
    """

    def __init__(
        self,
        supported_formats: tuple[str, ...] | None = None,
        min_duration_seconds: float = 0.1,
    ) -> None:
        """Initialize the validator.

        Args:
            supported_formats: Allowed file extensions. Defaults to
                ``CONFIG.audio.supported_formats``.
            min_duration_seconds: Minimum acceptable clip duration, in seconds.
                Clips shorter than this are rejected as unusable.
        """
        self.supported_formats = supported_formats or CONFIG.audio.supported_formats
        self.min_duration_seconds = min_duration_seconds

    def is_supported_format(self, filepath: Path) -> bool:
        """Check whether a file's extension is one this project supports.

        Args:
            filepath: Path to the audio file.

        Returns:
            ``True`` if the extension is supported.
        """
        return filepath.suffix.lower() in self.supported_formats

    def exists(self, filepath: Path) -> bool:
        """Check whether a file exists on disk.

        Args:
            filepath: Path to the audio file.

        Returns:
            ``True`` if the file exists.
        """
        return filepath.exists() and filepath.is_file()

    def validate(self, filepath: Path) -> ValidationResult:
        """Run every validation check against a single audio file.

        Args:
            filepath: Path to the audio file to validate.

        Returns:
            A :class:`ValidationResult` describing the outcome. Checks run
            cheapest-first: existence, extension, then a full decode to
            check for corruption, NaN/Inf samples and minimum duration.
        """
        if not self.exists(filepath):
            logger.warning("Validation failed for %s: file does not exist.", filepath)
            return ValidationResult(is_valid=False, reason="File does not exist")

        if not self.is_supported_format(filepath):
            logger.warning("Validation failed for %s: unsupported format.", filepath)
            return ValidationResult(is_valid=False, reason=f"Unsupported format: {filepath.suffix}")

        try:
            audio, sample_rate = sf.read(str(filepath), dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001 - any backend failure means "corrupted/unreadable"
            logger.warning("Validation failed for %s: unreadable (%s).", filepath, exc)
            return ValidationResult(is_valid=False, reason=f"Unreadable or corrupted audio: {exc}")

        if audio.size == 0:
            logger.warning("Validation failed for %s: empty audio.", filepath)
            return ValidationResult(is_valid=False, reason="Empty audio signal")

        if np.isnan(audio).any():
            logger.warning("Validation failed for %s: contains NaN values.", filepath)
            return ValidationResult(is_valid=False, reason="Audio contains NaN values")

        if np.isinf(audio).any():
            logger.warning("Validation failed for %s: contains Inf values.", filepath)
            return ValidationResult(is_valid=False, reason="Audio contains infinite values")

        num_frames = audio.shape[0] if audio.ndim > 0 else 0
        duration_seconds = num_frames / float(sample_rate)

        if duration_seconds < self.min_duration_seconds:
            logger.warning(
                "Validation failed for %s: duration %.3fs below minimum %.3fs.",
                filepath,
                duration_seconds,
                self.min_duration_seconds,
            )
            return ValidationResult(is_valid=False, reason="Duration below minimum threshold")

        return ValidationResult(
            is_valid=True,
            sample_rate=int(sample_rate),
            duration_seconds=duration_seconds,
        )
