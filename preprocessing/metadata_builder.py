"""One-time bootstrap: discover raw audio files and register them in metadata.csv.

Design rationale:
    The project rule is "never scan folders during training" — but *something*
    has to look at the filesystem exactly once to learn what raw files exist
    in the first place. This module is that single, explicitly-sanctioned
    exception. It is run standalone (or once at the start of
    :mod:`preprocessing.pipeline`) to populate ``metadata.csv`` with one row
    per raw audio file discovered under ``CONFIG.paths.raw_dir``. After this
    runs, every other module treats ``metadata.csv`` as authoritative and
    never touches the raw directory tree directly.

Expected raw layout:
    ``data/raw/<language>/<label>/<speaker_or_file>...``

    where ``<label>`` is one of ``CONFIG.dataset.labels`` (e.g. ``real`` /
    ``fake``). The speaker is taken from the immediate parent directory of
    the audio file when it differs from the label directory, otherwise
    ``"unknown"``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import soundfile as sf

from configs.config import CONFIG
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)


def _discover_audio_files(raw_dir: Path, supported_formats: tuple[str, ...]) -> list[Path]:
    """Recursively find every audio file with a supported extension.

    Args:
        raw_dir: Root directory to search under.
        supported_formats: Allowed file extensions, e.g. ``(".wav", ".flac")``.

    Returns:
        Sorted list of discovered file paths.
    """
    files: list[Path] = [
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_formats
    ]
    return sorted(files)


def _infer_label(filepath: Path, raw_dir: Path, valid_labels: tuple[str, ...]) -> str | None:
    """Infer the real/fake label from a file's path relative to ``raw_dir``.

    Args:
        filepath: Absolute path to the audio file.
        raw_dir: Root raw-data directory the file lives under.
        valid_labels: Labels considered valid, e.g. ``("real", "fake")``.

    Returns:
        The matching label string, or ``None`` if no ancestor directory
        matches a valid label.
    """
    relative_parts = {part.lower() for part in filepath.relative_to(raw_dir).parts}
    for label in valid_labels:
        if label.lower() in relative_parts:
            return label
    return None


def _infer_language(filepath: Path, raw_dir: Path, known_languages: tuple[str, ...]) -> str:
    """Infer the language from a file's path relative to ``raw_dir``.

    Args:
        filepath: Absolute path to the audio file.
        raw_dir: Root raw-data directory the file lives under.
        known_languages: Languages configured for this project.

    Returns:
        The matching language string, or ``"unknown"`` if none match.
    """
    relative_parts = {part.lower() for part in filepath.relative_to(raw_dir).parts}
    for language in known_languages:
        if language.lower() in relative_parts:
            return language
    return "unknown"


def _infer_speaker(filepath: Path, raw_dir: Path, valid_labels: tuple[str, ...], known_languages: tuple[str, ...]) -> str:
    """Infer a speaker identifier from the file's immediate parent directory.

    Only treats the parent directory as a speaker folder if it is neither
    the raw-data root, a label directory (``real``/``fake``), nor a language
    directory — those are structural, not per-speaker. In the standard
    ``raw/<language>/<label>/<file>`` layout this always resolves to
    ``"unknown"``, since no per-speaker subfolder exists.

    Args:
        filepath: Absolute path to the audio file.
        raw_dir: Root raw-data directory the file lives under.
        valid_labels: Labels considered structural, e.g. ``("real", "fake")``.
        known_languages: Languages considered structural, e.g. ``("hindi",)``.

    Returns:
        The parent directory name if it looks like a genuine per-speaker
        folder, otherwise ``"unknown"``.
    """
    parent_name = filepath.parent.name
    structural_names = {raw_dir.name.lower()}
    structural_names.update(label.lower() for label in valid_labels)
    structural_names.update(language.lower() for language in known_languages)

    if parent_name and parent_name.lower() not in structural_names:
        return parent_name
    return "unknown"


def _probe_audio(filepath: Path) -> tuple[int, float] | None:
    """Read sample rate and duration from a file's header without full decode.

    Args:
        filepath: Absolute path to the audio file.

    Returns:
        ``(sample_rate, duration_seconds)`` tuple, or ``None`` if the file
        could not be probed (e.g. corrupted header).
    """
    try:
        info = sf.info(str(filepath))
        return info.samplerate, info.frames / float(info.samplerate)
    except Exception as exc:  # noqa: BLE001 - any backend failure means "unreadable"
        logger.warning("Could not probe audio file %s: %s", filepath, exc)
        return None


def build_initial_metadata(metadata_manager: MetadataManager | None = None) -> MetadataManager:
    """Scan the raw data directory once and register discovered files.

    Existing rows (matched by ``audio_id`` derived from the file's relative
    path) are left untouched, so this function is safe to re-run after new
    raw files are added — it only appends new discoveries.

    Args:
        metadata_manager: Optional pre-constructed manager to add records to.
            Defaults to a new manager bound to ``CONFIG.paths.metadata_file``.

    Returns:
        The :class:`MetadataManager` with newly discovered files upserted
        and persisted.
    """
    manager = metadata_manager or MetadataManager(CONFIG.paths.metadata_file)

    raw_dir = CONFIG.paths.raw_dir
    if not raw_dir.exists():
        logger.warning("Raw directory %s does not exist; nothing to register.", raw_dir)
        return manager

    existing_ids = set(manager.dataframe["audio_id"]) if len(manager) else set()

    discovered_files = _discover_audio_files(raw_dir, CONFIG.audio.supported_formats)
    logger.info("Discovered %d candidate audio file(s) under %s", len(discovered_files), raw_dir)

    new_records: list[dict[str, object]] = []
    skipped_unlabeled = 0

    for filepath in discovered_files:
        audio_id = uuid.uuid5(uuid.NAMESPACE_URL, str(filepath)).hex

        if audio_id in existing_ids:
            continue

        label = _infer_label(filepath, raw_dir, CONFIG.dataset.labels)
        if label is None:
            skipped_unlabeled += 1
            logger.debug("Skipping %s: no recognizable label in path.", filepath)
            continue

        probe_result = _probe_audio(filepath)
        sample_rate, duration = probe_result if probe_result is not None else (0, 0.0)

        new_records.append(
            {
                "audio_id": audio_id,
                "filepath": str(filepath),
                "label": label,
                "speaker": _infer_speaker(filepath, raw_dir, CONFIG.dataset.labels, CONFIG.dataset.languages),
                "language": _infer_language(filepath, raw_dir, CONFIG.dataset.languages),
                "sample_rate": sample_rate,
                "duration": duration,
            }
        )

    manager.upsert_records(new_records)
    manager.save()

    logger.info(
        "Metadata bootstrap complete: %d new record(s) added, %d file(s) skipped (unlabeled).",
        len(new_records),
        skipped_unlabeled,
    )
    return manager


if __name__ == "__main__":
    build_initial_metadata()
