"""Bounded streaming ingestion of raw audio from HuggingFace source datasets.

Design rationale:
    The upstream sources for this project — ``ai4bharat/Kathbath`` (real
    Hindi speech, ~174GB) and ``vdivyasharma/IndicSynth`` (synthetic Hindi
    speech, part of an 845GB multilingual set) — are far too large to
    download in full for a research project. This module streams examples
    from each source over HTTP via the HuggingFace ``datasets`` library
    (``streaming=True``, so nothing beyond the current example is buffered),
    writes at most ``CONFIG.ingestion.samples_per_class`` raw clips per label
    to ``data/raw/<language>/<label>/``, and stops. It performs no audio
    decoding or transformation of its own — clips are written byte-for-byte
    in their original container format, and the existing
    :mod:`preprocessing.metadata_builder` / :mod:`preprocessing.pipeline`
    modules take over from there exactly as if the files had always been
    local. This module is the *only* one in the project allowed to make
    network calls to fetch dataset content.

Authentication:
    ``ai4bharat/Kathbath`` is a gated dataset. Before running this module
    you must (1) accept its terms at
    https://huggingface.co/datasets/ai4bharat/Kathbath while logged into a
    HuggingFace account, (2) create a read-scoped access token at
    https://huggingface.co/settings/tokens, and (3) set it as the
    environment variable named by ``CONFIG.ingestion.hf_token_env_var``
    (default ``HF_TOKEN``). ``vdivyasharma/IndicSynth`` requires no auth.
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import Audio, load_dataset
from datasets.exceptions import DatasetNotFoundError
from tqdm import tqdm

from configs.config import CONFIG, HFSourceConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetAccessError(RuntimeError):
    """Raised when a source dataset cannot be accessed, typically due to auth."""


def _existing_clip_count(destination_dir: Path) -> int:
    """Count clips already ingested for a given language/label directory.

    Used to make ingestion resumable: if a previous run already wrote N
    clips, this run only fetches enough additional clips to reach the
    configured target instead of re-fetching from the start.

    Args:
        destination_dir: Directory clips for one language/label are written to.

    Returns:
        Number of files already present in ``destination_dir``.
    """
    if not destination_dir.exists():
        return 0
    return sum(1 for entry in destination_dir.iterdir() if entry.is_file())


def _resolve_hf_token(token_env_var: str) -> str | None:
    """Read the HuggingFace access token from the configured environment variable.

    Args:
        token_env_var: Name of the environment variable holding the token.

    Returns:
        The token string, or ``None`` if it is not set (sufficient for
        ungated datasets).
    """
    token = os.environ.get(token_env_var)
    if not token:
        logger.debug("Environment variable %s is not set; proceeding unauthenticated.", token_env_var)
    return token


def ingest_source(
    source: HFSourceConfig,
    language: str,
    target_count: int,
    raw_dir: Path,
    hf_token: str | None,
) -> dict[str, int]:
    """Stream up to ``target_count`` clips from one HuggingFace source dataset.

    Args:
        source: Description of the dataset, config, split and label to pull.
        language: Language subdirectory to write clips under.
        target_count: Maximum number of clips to have on disk for this label
            after this call returns (existing clips count toward the total).
        raw_dir: Root raw-data directory (``CONFIG.paths.raw_dir``).
        hf_token: HuggingFace access token, or ``None`` for ungated datasets.

    Returns:
        Summary dict: ``{"already_present": int, "newly_written": int, "skipped": int}``.

    Raises:
        DatasetAccessError: If the dataset is gated/private and cannot be
            accessed with the provided token.
    """
    destination_dir = raw_dir / language / source.label
    destination_dir.mkdir(parents=True, exist_ok=True)

    already_present = _existing_clip_count(destination_dir)
    remaining = max(0, target_count - already_present)

    if remaining == 0:
        logger.info(
            "Skipping %s/%s: already have %d/%d clips.",
            language,
            source.label,
            already_present,
            target_count,
        )
        return {"already_present": already_present, "newly_written": 0, "skipped": 0}

    logger.info(
        "Streaming %s (%s/%s) for label '%s': need %d more clip(s).",
        source.dataset,
        source.config_name,
        source.split,
        source.label,
        remaining,
    )

    try:
        dataset = load_dataset(
            source.dataset,
            source.config_name,
            split=source.split,
            streaming=True,
            token=hf_token,
        )
        dataset = dataset.cast_column(source.audio_field, Audio(decode=False))
    except DatasetNotFoundError as exc:
        raise DatasetAccessError(
            f"Cannot access '{source.dataset}'. If this is a gated dataset, accept its terms on "
            f"huggingface.co while logged in, create a read token at "
            f"huggingface.co/settings/tokens, and set it in the "
            f"'{CONFIG.ingestion.hf_token_env_var}' environment variable."
        ) from exc

    newly_written = 0
    skipped = 0

    progress = tqdm(total=remaining, desc=f"{language}/{source.label}")

    for example in dataset:
        if newly_written >= remaining:
            break

        audio_entry = example.get(source.audio_field)
        if not audio_entry or not audio_entry.get("bytes"):
            skipped += 1
            continue

        source_name = audio_entry.get("path") or f"clip_{already_present + newly_written}.wav"
        extension = Path(source_name).suffix or ".wav"

        clip_id = f"{source.label}_{already_present + newly_written:06d}{extension}"
        output_path = destination_dir / clip_id

        try:
            output_path.write_bytes(audio_entry["bytes"])
        except OSError as exc:
            logger.warning("Failed to write clip %s: %s", output_path, exc)
            skipped += 1
            continue

        newly_written += 1
        progress.update(1)

    progress.close()

    logger.info(
        "Finished %s/%s: %d already present, %d newly written, %d skipped.",
        language,
        source.label,
        already_present,
        newly_written,
        skipped,
    )
    return {"already_present": already_present, "newly_written": newly_written, "skipped": skipped}


def ingest_all() -> dict[str, dict[str, int]]:
    """Ingest both the real and fake source datasets per ``CONFIG.ingestion``.

    Returns:
        Mapping of label -> summary dict, as returned by :func:`ingest_source`.
    """
    hf_token = _resolve_hf_token(CONFIG.ingestion.hf_token_env_var)

    results: dict[str, dict[str, int]] = {}
    for source in (CONFIG.ingestion.real, CONFIG.ingestion.fake):
        results[source.label] = ingest_source(
            source=source,
            language=CONFIG.ingestion.language,
            target_count=CONFIG.ingestion.samples_per_class,
            raw_dir=CONFIG.paths.raw_dir,
            hf_token=hf_token,
        )

    return results


if __name__ == "__main__":
    ingest_all()
