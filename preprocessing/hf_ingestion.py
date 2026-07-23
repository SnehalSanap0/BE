"""Bounded streaming ingestion of raw audio from HuggingFace source datasets.

Design rationale:
    The upstream sources for this project are far too large to download in
    full for a research project. This module streams examples from each
    configured source over HTTP via the HuggingFace ``datasets`` library
    (``streaming=True``, so nothing beyond the current example is buffered),
    writes at most that source's ``target_count`` raw clips to
    ``data/raw/<language>/<label>/``, and stops. It performs no audio
    decoding or transformation of its own — clips are written byte-for-byte
    in their original container format, and the existing
    :mod:`preprocessing.metadata_builder` / :mod:`preprocessing.pipeline`
    modules take over from there exactly as if the files had always been
    local. This module is the *only* one in the project allowed to make
    network calls to fetch dataset content.

    Each label (``real``/``fake``) is backed by *multiple* sources
    (``CONFIG.ingestion.real`` / ``.fake``), deliberately: a single source
    per label means "real vs fake" is perfectly confounded with "which
    dataset," so a detector can score well by recognizing recording/dataset
    fingerprints rather than genuine spoofing artifacts. Current sources:
      - real: ``ai4bharat/Kathbath`` + Common Voice Hindi (via a Parquet
        mirror, since the official Mozilla repo uses a loading-script format
        newer ``datasets`` versions no longer support)
      - fake: ``vdivyasharma/IndicSynth`` + ``Jack-ppkdczgx/SEA-Spoof``
        (filtered to Hindi), which spans multiple distinct TTS/VC systems
        rather than just IndicSynth's, so the fake class isn't defined by
        one synthesis technique's artifacts.

Authentication:
    ``ai4bharat/Kathbath`` and ``Jack-ppkdczgx/SEA-Spoof`` are gated
    datasets. Before running this module you must (1) accept each one's
    terms/request access on huggingface.co while logged into a HuggingFace
    account, (2) create a read-scoped access token at
    https://huggingface.co/settings/tokens, and (3) set it as the
    environment variable named by ``CONFIG.ingestion.hf_token_env_var``
    (default ``HF_TOKEN``). The other sources require no auth.
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


def _existing_clip_count(destination_dir: Path, source_tag: str) -> int:
    """Count clips already ingested by one specific source into a label directory.

    Used to make ingestion resumable per-source: multiple sources can share
    the same ``destination_dir`` (e.g. two real sources both writing into
    ``data/raw/hindi/real/``), so counting must be scoped to files this
    source itself wrote — identified by its ``source_tag`` filename prefix —
    not every file in the directory.

    Args:
        destination_dir: Directory clips for one language/label are written to.
        source_tag: This source's filename prefix (``HFSourceConfig.source_tag``).

    Returns:
        Number of files already present in ``destination_dir`` belonging to this source.
    """
    if not destination_dir.exists():
        return 0
    prefix = f"{source_tag}_"
    return sum(1 for entry in destination_dir.iterdir() if entry.is_file() and entry.name.startswith(prefix))


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


def _matches_filters(example: dict, filters: tuple[tuple[str, str], ...]) -> bool:
    """Check whether a streamed example satisfies every configured filter.

    Args:
        example: One row from a streamed HuggingFace dataset.
        filters: ``(column, required_value)`` pairs. Empty means "no filtering."

    Returns:
        ``True`` if every filter's column equals its required value.
    """
    return all(str(example.get(field)) == value for field, value in filters)


def ingest_source(
    source: HFSourceConfig,
    language: str,
    raw_dir: Path,
    hf_token: str | None,
) -> dict[str, int]:
    """Stream up to ``source.target_count`` clips from one HuggingFace source dataset.

    Args:
        source: Description of the dataset, config, split, label and target
            count to pull. Multiple sources may target the same label/directory —
            see :class:`configs.config.HFSourceConfig` for how that stays resumable.
        language: Language subdirectory to write clips under.
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

    already_present = _existing_clip_count(destination_dir, source.source_tag)
    remaining = max(0, source.target_count - already_present)

    if remaining == 0:
        logger.info(
            "Skipping %s (%s/%s): already have %d/%d clips.",
            source.source_tag,
            language,
            source.label,
            already_present,
            source.target_count,
        )
        return {"already_present": already_present, "newly_written": 0, "skipped": 0}

    logger.info(
        "Streaming %s (%s/%s) for label '%s' [%s]: need %d more clip(s).",
        source.dataset,
        source.config_name,
        source.split,
        source.label,
        source.source_tag,
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

    progress = tqdm(total=remaining, desc=f"{language}/{source.label}/{source.source_tag}")

    for example in dataset:
        if newly_written >= remaining:
            break

        if source.filters and not _matches_filters(example, source.filters):
            continue

        audio_entry = example.get(source.audio_field)
        if not audio_entry or not audio_entry.get("bytes"):
            skipped += 1
            continue

        source_name = audio_entry.get("path") or f"clip_{already_present + newly_written}.wav"
        extension = Path(source_name).suffix or ".wav"

        clip_id = f"{source.source_tag}_{already_present + newly_written:06d}{extension}"
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
        "Finished %s (%s/%s): %d already present, %d newly written, %d skipped.",
        source.source_tag,
        language,
        source.label,
        already_present,
        newly_written,
        skipped,
    )
    return {"already_present": already_present, "newly_written": newly_written, "skipped": skipped}


def ingest_all() -> dict[str, dict[str, int]]:
    """Ingest every real and fake source dataset per ``CONFIG.ingestion``.

    Returns:
        Mapping of ``source_tag`` -> summary dict, as returned by :func:`ingest_source`.
    """
    hf_token = _resolve_hf_token(CONFIG.ingestion.hf_token_env_var)

    results: dict[str, dict[str, int]] = {}
    for source in (*CONFIG.ingestion.real, *CONFIG.ingestion.fake):
        results[source.source_tag] = ingest_source(
            source=source,
            language=CONFIG.ingestion.language,
            raw_dir=CONFIG.paths.raw_dir,
            hf_token=hf_token,
        )

    return results


if __name__ == "__main__":
    ingest_all()
