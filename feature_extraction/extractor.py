"""Single-file feature extraction: load processed audio, extract, resize, save.

Single responsibility: given one processed ``.wav`` file and a destination
``.npy`` path, produce one saved feature array. Mirrors the shape of
:class:`preprocessing.processor.AudioProcessor` — this class is unaware of
metadata.csv or batch iteration, which keeps it directly reusable from the
Flask inference path in Phase 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from configs.config import CONFIG
from feature_extraction.transforms import FEATURE_EXTRACTORS
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of extracting features from a single processed audio file.

    Attributes:
        success: Whether extraction completed and the output was written.
        output_path: Path the feature array was written to, if successful.
        feature_shape: Shape of the saved array, if successful.
        error: Failure reason when ``success`` is ``False``.
    """

    success: bool
    output_path: Path | None = None
    feature_shape: tuple[int, ...] | None = None
    error: str = ""


def _resize_to_square(feature: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a 2-D feature map to ``(image_size, image_size)`` via interpolation.

    Args:
        feature: 2-D array of shape ``(n_coeffs, num_frames)``.
        image_size: Target spatial size for both dimensions.

    Returns:
        2-D array of shape ``(image_size, image_size)``, dtype ``float32``.
    """
    resized = cv2.resize(feature, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def _normalize(feature: np.ndarray) -> np.ndarray:
    """Standardize a feature array to zero mean and unit variance.

    Args:
        feature: Feature array of any shape.

    Returns:
        Standardized array. Returned unchanged if the array has zero variance.
    """
    std = feature.std()
    if std == 0:
        return feature - feature.mean()
    return (feature - feature.mean()) / std


class FeatureExtractor:
    """Extracts a configured feature type from a single processed audio file."""

    def __init__(self, feature_type: str | None = None, image_size: int | None = None) -> None:
        """Initialize the extractor.

        Args:
            feature_type: One of ``"mel_spectrogram"``, ``"mfcc"``, ``"lfcc"``.
                Defaults to ``CONFIG.features.type``.
            image_size: Spatial size features are resized to. Defaults to
                ``CONFIG.features.image_size``.

        Raises:
            ValueError: If ``feature_type`` is not a recognized extractor.
        """
        self.feature_type = feature_type or CONFIG.features.type
        self.image_size = image_size or CONFIG.features.image_size

        if self.feature_type not in FEATURE_EXTRACTORS:
            raise ValueError(
                f"Unknown feature type '{self.feature_type}'. "
                f"Expected one of {list(FEATURE_EXTRACTORS)}."
            )

        self._extract_fn = FEATURE_EXTRACTORS[self.feature_type]

    def extract_file(self, input_path: Path, output_path: Path) -> ExtractionResult:
        """Load a processed audio file, extract features, and save them as ``.npy``.

        Args:
            input_path: Path to a processed (mono, 16kHz, fixed-length) ``.wav`` file.
            output_path: Destination path for the ``.npy`` feature array.
                Parent directories are created automatically.

        Returns:
            An :class:`ExtractionResult` describing success or failure.
            Failures never raise — they are reported so batch callers
            (:mod:`feature_extraction.pipeline`) can continue past a single
            bad file.
        """
        try:
            audio, sample_rate = sf.read(str(input_path), dtype="float32", always_2d=False)
            feature = self.extract_array(audio, sample_rate)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, feature)

        except Exception as exc:  # noqa: BLE001 - any failure here is an extraction failure, not a crash
            logger.error("Failed to extract features from %s: %s", input_path, exc, exc_info=True)
            return ExtractionResult(success=False, error=str(exc))

        logger.info("Extracted %s features %s -> %s", self.feature_type, input_path.name, output_path)

        return ExtractionResult(success=True, output_path=output_path, feature_shape=feature.shape)

    def extract_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract, resize and normalize features from an in-memory waveform.

        Factored out of :meth:`extract_file` so single-inference callers
        (Grad-CAM explanation, the Flask inference endpoint) can reuse the
        exact same feature computation without a round trip through disk.

        Args:
            audio: 1-D mono waveform, already preprocessed (fixed length,
                target sample rate).
            sample_rate: Sample rate of ``audio``, in Hz.

        Returns:
            2-D array of shape ``(image_size, image_size)``, z-score normalized.
        """
        feature = self._extract_fn(audio, sample_rate)
        feature = _resize_to_square(feature, self.image_size)
        return _normalize(feature)
