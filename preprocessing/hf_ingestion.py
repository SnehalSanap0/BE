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

    SEA-Spoof's paired bonafide/spoof ingestion (:func:`ingest_paired_sources`)
    does not use ``datasets``' ``streaming=True`` mode. SEA-Spoof's parquet
    shards live behind HuggingFace's Xet storage backend, and streaming mode
    cannot use the fast Xet transfer path for it — it falls back to slow,
    single-connection, timeout-prone sequential HTTP range requests (confirmed
    by direct measurement: it could not finish downloading a single ~1.5GB
    shard in over an hour). Instead, each needed shard is downloaded once via
    :func:`huggingface_hub.hf_hub_download`, which does use the fast Xet path
    (measured ~4-5 minutes per shard, vs. never completing one via streaming),
    and queried locally with ``pyarrow`` for matched pairs.

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

import csv
import os
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Audio, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi, hf_hub_download
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


def _filter_value(source: HFSourceConfig, field: str) -> str | None:
    """Return one configured filter value for ``field``, if present."""
    for filter_field, value in source.filters:
        if filter_field == field:
            return value
    return None


def _non_label_filters(source: HFSourceConfig) -> tuple[tuple[str, str], ...]:
    """Return filters that should apply to both sides of a paired source."""
    return tuple((field, value) for field, value in source.filters if field != "label")


def _pair_key(example: dict) -> str | None:
    """Pick a stable transcript-pairing key from a SEA-Spoof example.

    ``utterance_id`` encodes the synthesis system for spoof rows (e.g.
    ``hindi_indic_indic-tts_0001581_hindi_offline``), so it never matches a
    bonafide row's id. ``text`` — the transcript spoof clips were synthesized
    from — is the only field that actually links a bonafide clip to its spoof
    counterpart(s); confirmed against real SEA-Spoof rows before relying on it.
    """
    text = example.get("text")
    if text is None:
        return None
    normalized = str(text).strip()
    return normalized or None


def _audio_extension(audio_entry: dict, fallback: str = ".wav") -> str:
    """Infer an output extension from a HuggingFace audio entry."""
    source_name = audio_entry.get("path") or f"clip{fallback}"
    return Path(source_name).suffix or fallback


def _write_audio_bytes(output_path: Path, audio_entry: dict) -> None:
    """Write one HuggingFace audio entry's raw bytes to disk."""
    output_path.write_bytes(audio_entry["bytes"])


#: Sidecar filename (one per language directory) mapping a written clip's
#: filename to the transcript it was paired/selected by. Written here rather
#: than into metadata.csv directly, since raw ingestion never touches
#: metadata.csv -- :mod:`preprocessing.metadata_builder` is the sole writer of
#: that file. ``metadata_builder`` reads this sidecar when it discovers a raw
#: file with a matching filename and copies the transcript into its
#: ``transcript`` column, so pairing can be verified from metadata.csv alone
#: (matching ``real_sea_NNNNNN`` / ``fake_sea_NNNNNN`` transcripts) instead of
#: only being implied by filename-index matching.
PAIR_TRANSCRIPTS_FILENAME = "pair_transcripts.csv"


def _append_pair_transcripts(language_dir: Path, records: list[tuple[str, str]]) -> None:
    """Append ``(filename, transcript)`` rows to this language's transcript sidecar.

    Args:
        language_dir: ``raw_dir / language`` directory (parent of the
            ``real``/``fake`` label subdirectories).
        records: ``(filename, transcript)`` pairs to append. Filenames are
            basenames only (e.g. ``real_sea_001500.flac``), not full paths.
    """
    if not records:
        return

    sidecar_path = language_dir / PAIR_TRANSCRIPTS_FILENAME
    is_new_file = not sidecar_path.exists()
    language_dir.mkdir(parents=True, exist_ok=True)

    with sidecar_path.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        if is_new_file:
            writer.writerow(["filename", "transcript"])
        writer.writerows(records)


def _list_train_shards(dataset: str, split: str, hf_token: str | None) -> list[str]:
    """List a split's parquet shard filenames in a HuggingFace dataset repo, in order.

    Args:
        dataset: HuggingFace dataset repo id (e.g. ``Jack-ppkdczgx/SEA-Spoof``).
        split: Split name (e.g. ``train``).
        hf_token: HuggingFace access token, or ``None`` for ungated datasets.

    Returns:
        Sorted list of repo-relative parquet filenames for the split.
    """
    prefix = f"data/{split}/"
    files = HfApi(token=hf_token).list_repo_files(repo_id=dataset, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def _ordered_pairing_shards(shard_filenames: list[str]) -> list[str]:
    """Apply optional shard-order overrides for quick SEA-Spoof smoke runs."""
    ordered = list(shard_filenames)

    start_raw = os.environ.get("SEA_SPOOF_SHARD_START_INDEX")
    if start_raw:
        try:
            start_index = int(start_raw)
        except ValueError:
            logger.warning("Ignoring invalid SEA_SPOOF_SHARD_START_INDEX=%r.", start_raw)
        else:
            ordered = ordered[start_index:]
            logger.info("Starting SEA-Spoof paired scan at shard index %d.", start_index)

    if os.environ.get("SEA_SPOOF_SHARD_REVERSE", "").strip().lower() in {"1", "true", "yes"}:
        ordered = list(reversed(ordered))
        logger.info("Scanning SEA-Spoof paired shards in reverse order.")

    return ordered


def _download_shard(dataset: str, filename: str, hf_token: str | None) -> Path:
    """Download one parquet shard via the fast Xet transfer path, or reuse the local cache.

    Args:
        dataset: HuggingFace dataset repo id.
        filename: Repo-relative parquet filename, as returned by :func:`_list_train_shards`.
        hf_token: HuggingFace access token, or ``None`` for ungated datasets.

    Returns:
        Local filesystem path to the downloaded (or already-cached) shard.
    """
    return Path(hf_hub_download(repo_id=dataset, repo_type="dataset", filename=filename, token=hf_token))


def _sources_form_pair(real_source: HFSourceConfig, fake_source: HFSourceConfig) -> bool:
    """Check whether two configured sources can be streamed as matched pairs."""
    return (
        real_source.dataset == fake_source.dataset
        and real_source.config_name == fake_source.config_name
        and real_source.split == fake_source.split
        and real_source.audio_field == fake_source.audio_field
        and _filter_value(real_source, "label") == "bonafide"
        and _filter_value(fake_source, "label") == "spoof"
        and _non_label_filters(real_source) == _non_label_filters(fake_source)
    )


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

        extension = _audio_extension(audio_entry)

        clip_id = f"{source.source_tag}_{already_present + newly_written:06d}{extension}"
        output_path = destination_dir / clip_id

        try:
            _write_audio_bytes(output_path, audio_entry)
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


def ingest_paired_sources(
    real_source: HFSourceConfig,
    fake_source: HFSourceConfig,
    language: str,
    raw_dir: Path,
    hf_token: str | None,
) -> dict[str, dict[str, int]]:
    """Ingest SEA-Spoof as transcript-matched bonafide/spoof pairs.

    The ordinary ``ingest_source`` path samples each label independently. For
    SEA-Spoof, the useful training unit is a pair: one bonafide clip and one
    spoof clip sharing the same transcript. This function keeps the
    downstream folder layout unchanged while ensuring both classes receive
    the same number of matched examples.

    Shards are downloaded once each (see :func:`_download_shard`) and queried
    locally with pyarrow: first for the metadata columns needed to pick
    matched transcript keys, then again (same cached files, no re-download)
    for the audio bytes of just those keys. Shards are only fetched as
    needed, in order, stopping as soon as enough matched pairs are found.
    """
    real_dir = raw_dir / language / real_source.label
    fake_dir = raw_dir / language / fake_source.label
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_existing = _existing_clip_count(real_dir, real_source.source_tag)
    fake_existing = _existing_clip_count(fake_dir, fake_source.source_tag)
    complete_existing_pairs = min(real_existing, fake_existing)
    next_pair_index = max(real_existing, fake_existing)
    target_pairs = min(real_source.target_count, fake_source.target_count)
    remaining = max(0, target_pairs - complete_existing_pairs)

    if real_source.target_count != fake_source.target_count:
        logger.warning(
            "Paired sources %s/%s have different target_count values (%d vs %d); "
            "using %d matched pair(s).",
            real_source.source_tag,
            fake_source.source_tag,
            real_source.target_count,
            fake_source.target_count,
            target_pairs,
        )

    if real_existing != fake_existing:
        logger.warning(
            "Existing paired source counts are imbalanced (%s=%d, %s=%d). "
            "New pair numbering will continue at %d.",
            real_source.source_tag,
            real_existing,
            fake_source.source_tag,
            fake_existing,
            next_pair_index,
        )

    if remaining == 0:
        logger.info(
            "Skipping paired %s/%s (%s): already have %d/%d matched pair(s).",
            real_source.source_tag,
            fake_source.source_tag,
            language,
            complete_existing_pairs,
            target_pairs,
        )
        summary = {"already_present": complete_existing_pairs, "newly_written": 0, "skipped": 0}
        return {real_source.source_tag: summary, fake_source.source_tag: summary.copy()}

    label_to_source = {
        _filter_value(real_source, "label"): real_source,
        _filter_value(fake_source, "label"): fake_source,
    }
    common_filters = _non_label_filters(real_source)
    real_label_value = _filter_value(real_source, "label")
    fake_label_value = _filter_value(fake_source, "label")
    if real_label_value is None or fake_label_value is None:
        raise ValueError("Paired SEA-Spoof sources must both define a label filter.")

    try:
        shard_filenames = _list_train_shards(real_source.dataset, real_source.split, hf_token)
    except Exception as exc:
        raise DatasetAccessError(
            f"Cannot list shards for '{real_source.dataset}'. If this is a gated dataset, accept its terms on "
            f"huggingface.co while logged in, create a read token at "
            f"huggingface.co/settings/tokens, and set it in the "
            f"'{CONFIG.ingestion.hf_token_env_var}' environment variable."
        ) from exc

    if not shard_filenames:
        raise DatasetAccessError(
            f"No parquet shards found for split '{real_source.split}' in '{real_source.dataset}'."
        )
    shard_filenames = _ordered_pairing_shards(shard_filenames)

    logger.info(
        "Pairing %s (%s/%s): need %d more matched bonafide/spoof pair(s); %d shard(s) available.",
        real_source.dataset,
        real_source.config_name,
        real_source.split,
        remaining,
        len(shard_filenames),
    )
    logger.info(
        "First pass downloads shards once each (Xet-accelerated) and scans metadata only "
        "to choose %d transcript-matched key(s); audio bytes are fetched afterward from the same cached shards.",
        remaining,
    )

    pending_labels_by_key: dict[str, set[str]] = {}
    selected_keys: list[str] = []
    selected_key_set: set[str] = set()
    rows_seen = 0
    common_filter_matches = 0
    bonafide_seen = 0
    spoof_seen = 0
    duplicate_sides = 0
    skipped = 0
    scanned_shards: list[str] = []

    for shard_filename in shard_filenames:
        if len(selected_keys) >= remaining:
            break

        shard_path = _download_shard(real_source.dataset, shard_filename, hf_token)
        scanned_shards.append(shard_filename)
        shard_rows_seen = 0
        shard_matched = 0

        for batch in pq.ParquetFile(shard_path).iter_batches(columns=["text", "language", "label"]):
            if len(selected_keys) >= remaining:
                break
            for row in batch.to_pylist():
                rows_seen += 1
                shard_rows_seen += 1

                if common_filters and not _matches_filters(row, common_filters):
                    continue
                common_filter_matches += 1

                label_value = str(row.get("label"))
                if label_value not in label_to_source:
                    continue

                if label_value == real_label_value:
                    bonafide_seen += 1
                elif label_value == fake_label_value:
                    spoof_seen += 1

                key = _pair_key(row)
                if key is None:
                    skipped += 1
                    continue

                labels_for_key = pending_labels_by_key.setdefault(key, set())
                if label_value in labels_for_key:
                    duplicate_sides += 1
                    continue
                labels_for_key.add(label_value)

                if real_label_value not in labels_for_key or fake_label_value not in labels_for_key:
                    continue

                if key not in selected_key_set:
                    selected_keys.append(key)
                    selected_key_set.add(key)
                    shard_matched += 1
                pending_labels_by_key.pop(key, None)

                if len(selected_keys) >= remaining:
                    break

        logger.info(
            "Pair metadata scan: shard %s done (%d row(s) in shard, %d newly matched key(s)); "
            "%d row(s) scanned overall, %d matched language/filter, %d bonafide, %d spoof, "
            "%d pending key(s), %d/%d key(s) selected.",
            shard_filename,
            shard_rows_seen,
            shard_matched,
            rows_seen,
            common_filter_matches,
            bonafide_seen,
            spoof_seen,
            len(pending_labels_by_key),
            len(selected_keys),
            remaining,
        )

    if len(selected_keys) < remaining:
        logger.warning(
            "Exhausted all %d available shard(s) after scanning %d row(s): only found %d/%d matched key(s).",
            len(shard_filenames),
            rows_seen,
            len(selected_keys),
            remaining,
        )

    logger.info(
        "Metadata pass selected %d/%d matched key(s) after scanning %d row(s) across %d shard(s): "
        "%d matched language/filter, %d bonafide, %d spoof, %d skipped, %d duplicate side(s).",
        len(selected_keys),
        remaining,
        rows_seen,
        len(scanned_shards),
        common_filter_matches,
        bonafide_seen,
        spoof_seen,
        skipped,
        duplicate_sides,
    )

    if not selected_keys:
        logger.warning("No matched bonafide/spoof transcript keys found; no audio will be written.")
        summary = {"already_present": complete_existing_pairs, "newly_written": 0, "skipped": skipped}
        return {real_source.source_tag: summary, fake_source.source_tag: summary.copy()}

    logger.info(
        "Second pass re-reads the same %d cached shard(s) for audio bytes of %d selected key(s).",
        len(scanned_shards),
        len(selected_keys),
    )

    audio_by_key: dict[str, dict[str, dict]] = {}
    audio_rows_seen = 0
    audio_skipped = 0

    for shard_filename in scanned_shards:
        complete_audio_pairs = sum(
            1 for pair in audio_by_key.values() if real_label_value in pair and fake_label_value in pair
        )
        if complete_audio_pairs >= len(selected_keys):
            break

        shard_path = _download_shard(real_source.dataset, shard_filename, hf_token)
        parquet_file = pq.ParquetFile(shard_path)

        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_index,
                columns=["text", "label", real_source.audio_field],
                use_threads=False,
            )
            for row in table.to_pylist():
                audio_rows_seen += 1
                key = _pair_key(row)
                label_value = str(row.get("label"))
                if key not in selected_key_set or label_value not in label_to_source:
                    continue

                audio_entry = row.get(real_source.audio_field)
                if not audio_entry or not audio_entry.get("bytes"):
                    audio_skipped += 1
                    continue

                audio_by_key.setdefault(key, {})[label_value] = audio_entry

            complete_audio_pairs = sum(
                1 for pair in audio_by_key.values() if real_label_value in pair and fake_label_value in pair
            )
            if complete_audio_pairs >= len(selected_keys):
                break

        logger.info(
            "Pair audio fetch: shard %s done; %d row(s) seen so far, %d/%d complete audio pair(s), %d skipped.",
            shard_filename,
            audio_rows_seen,
            complete_audio_pairs,
            len(selected_keys),
            audio_skipped,
        )

    newly_written_pairs = 0
    write_skipped = 0
    new_transcript_records: list[tuple[str, str]] = []
    progress = tqdm(total=len(selected_keys), desc=f"{language}/paired/{real_source.source_tag}+{fake_source.source_tag}")

    for key in selected_keys:
        pair = audio_by_key.get(key, {})
        if real_label_value not in pair or fake_label_value not in pair:
            write_skipped += 1
            continue

        pair_index = next_pair_index + newly_written_pairs
        real_audio = pair[real_label_value]
        fake_audio = pair[fake_label_value]
        real_path = real_dir / f"{real_source.source_tag}_{pair_index:06d}{_audio_extension(real_audio)}"
        fake_path = fake_dir / f"{fake_source.source_tag}_{pair_index:06d}{_audio_extension(fake_audio)}"

        try:
            _write_audio_bytes(real_path, real_audio)
            _write_audio_bytes(fake_path, fake_audio)
        except OSError as exc:
            logger.warning("Failed to write paired clips %s / %s: %s", real_path, fake_path, exc)
            real_path.unlink(missing_ok=True)
            fake_path.unlink(missing_ok=True)
            write_skipped += 1
            continue

        new_transcript_records.append((real_path.name, key))
        new_transcript_records.append((fake_path.name, key))
        newly_written_pairs += 1
        progress.update(1)

    progress.close()
    _append_pair_transcripts(raw_dir / language, new_transcript_records)

    logger.info(
        "Finished paired %s/%s (%s): %d already present, %d newly written pair(s), "
        "%d metadata skipped, %d audio skipped, %d write skipped, %d metadata row(s) scanned "
        "across %d shard(s), %d matched language/filter, %d bonafide, %d spoof, "
        "%d duplicate side(s), %d selected audio row(s) seen.",
        real_source.source_tag,
        fake_source.source_tag,
        language,
        complete_existing_pairs,
        newly_written_pairs,
        skipped,
        audio_skipped,
        write_skipped,
        rows_seen,
        len(scanned_shards),
        common_filter_matches,
        bonafide_seen,
        spoof_seen,
        duplicate_sides,
        audio_rows_seen,
    )

    real_summary = {
        "already_present": complete_existing_pairs,
        "newly_written": newly_written_pairs,
        "skipped": skipped + audio_skipped + write_skipped,
    }
    fake_summary = real_summary.copy()
    return {real_source.source_tag: real_summary, fake_source.source_tag: fake_summary}


def ingest_cosourced_independent(
    real_source: HFSourceConfig,
    fake_source: HFSourceConfig,
    language: str,
    raw_dir: Path,
    hf_token: str | None,
) -> dict[str, dict[str, int]]:
    """Ingest bonafide/spoof clips independently from one shared shard pool.

    Used for the E1 (non-paired) control: ``real_source``/``fake_source``
    point at the same dataset/split (e.g. SEA-Spoof ``train``), same as
    :func:`ingest_paired_sources`, but clips are *not* matched by transcript
    — each label is simply filled up to its own ``target_count`` from
    whichever rows match its filters, in scan order.

    Shares :func:`_download_shard`'s local, cache-aware download with the
    paired path so each shard is fetched from HuggingFace once total, not
    once per label — using :func:`ingest_source`'s HF-streaming path here
    instead would silently re-stream the same shard bytes twice (once per
    label), roughly doubling transfer time for no benefit.
    """
    real_dir = raw_dir / language / real_source.label
    fake_dir = raw_dir / language / fake_source.label
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_existing = _existing_clip_count(real_dir, real_source.source_tag)
    fake_existing = _existing_clip_count(fake_dir, fake_source.source_tag)
    real_remaining = max(0, real_source.target_count - real_existing)
    fake_remaining = max(0, fake_source.target_count - fake_existing)

    if real_remaining == 0 and fake_remaining == 0:
        logger.info(
            "Skipping co-sourced %s/%s (%s): already have %d/%d and %d/%d clip(s).",
            real_source.source_tag, fake_source.source_tag, language,
            real_existing, real_source.target_count, fake_existing, fake_source.target_count,
        )
        return {
            real_source.source_tag: {"already_present": real_existing, "newly_written": 0, "skipped": 0},
            fake_source.source_tag: {"already_present": fake_existing, "newly_written": 0, "skipped": 0},
        }

    try:
        shard_filenames = _list_train_shards(real_source.dataset, real_source.split, hf_token)
    except Exception as exc:
        raise DatasetAccessError(
            f"Cannot list shards for '{real_source.dataset}'. If this is a gated dataset, accept its terms on "
            f"huggingface.co while logged in, create a read token at "
            f"huggingface.co/settings/tokens, and set it in the "
            f"'{CONFIG.ingestion.hf_token_env_var}' environment variable."
        ) from exc
    if not shard_filenames:
        raise DatasetAccessError(
            f"No parquet shards found for split '{real_source.split}' in '{real_source.dataset}'."
        )
    shard_filenames = _ordered_pairing_shards(shard_filenames)

    columns = sorted({real_source.audio_field}
                      | {field for field, _ in real_source.filters}
                      | {field for field, _ in fake_source.filters})

    logger.info(
        "Co-sourced independent ingest %s/%s (%s): need %d real + %d fake more clip(s); %d shard(s) available.",
        real_source.source_tag, fake_source.source_tag, language,
        real_remaining, fake_remaining, len(shard_filenames),
    )

    real_written = 0
    fake_written = 0
    skipped = 0
    rows_seen = 0
    real_progress = tqdm(total=real_remaining, desc=f"{language}/{real_source.label}/{real_source.source_tag}")
    fake_progress = tqdm(total=fake_remaining, desc=f"{language}/{fake_source.label}/{fake_source.source_tag}")

    for shard_filename in shard_filenames:
        if real_written >= real_remaining and fake_written >= fake_remaining:
            break

        shard_path = _download_shard(real_source.dataset, shard_filename, hf_token)
        parquet_file = pq.ParquetFile(shard_path)

        for row_group_index in range(parquet_file.num_row_groups):
            if real_written >= real_remaining and fake_written >= fake_remaining:
                break
            table = parquet_file.read_row_group(row_group_index, columns=columns, use_threads=False)
            for row in table.to_pylist():
                if real_written >= real_remaining and fake_written >= fake_remaining:
                    break
                rows_seen += 1

                if real_written < real_remaining and _matches_filters(row, real_source.filters):
                    source, dest_dir, index = real_source, real_dir, real_existing + real_written
                elif fake_written < fake_remaining and _matches_filters(row, fake_source.filters):
                    source, dest_dir, index = fake_source, fake_dir, fake_existing + fake_written
                else:
                    continue

                audio_entry = row.get(source.audio_field)
                if not audio_entry or not audio_entry.get("bytes"):
                    skipped += 1
                    continue

                output_path = dest_dir / f"{source.source_tag}_{index:06d}{_audio_extension(audio_entry)}"
                try:
                    _write_audio_bytes(output_path, audio_entry)
                except OSError as exc:
                    logger.warning("Failed to write clip %s: %s", output_path, exc)
                    skipped += 1
                    continue

                if source is real_source:
                    real_written += 1
                    real_progress.update(1)
                else:
                    fake_written += 1
                    fake_progress.update(1)

        logger.info(
            "Co-sourced scan: shard %s done; %d row(s) seen so far, %d/%d real, %d/%d fake, %d skipped.",
            shard_filename, rows_seen, real_written, real_remaining, fake_written, fake_remaining, skipped,
        )

    real_progress.close()
    fake_progress.close()

    if real_written < real_remaining or fake_written < fake_remaining:
        logger.warning(
            "Exhausted all %d available shard(s) after scanning %d row(s): only found %d/%d real, %d/%d fake.",
            len(shard_filenames), rows_seen, real_written, real_remaining, fake_written, fake_remaining,
        )

    logger.info(
        "Finished co-sourced %s/%s (%s): %d already present real, %d newly written real, "
        "%d already present fake, %d newly written fake, %d skipped, %d row(s) scanned.",
        real_source.source_tag, fake_source.source_tag, language,
        real_existing, real_written, fake_existing, fake_written, skipped, rows_seen,
    )

    return {
        real_source.source_tag: {"already_present": real_existing, "newly_written": real_written, "skipped": skipped},
        fake_source.source_tag: {"already_present": fake_existing, "newly_written": fake_written, "skipped": 0},
    }


def ingest_wavefake_paired(
    dataset: str,
    split: str,
    id_field: str,
    audio_field: str,
    real_value: str,
    fake_values: tuple[str, ...],
    target_pairs: int,
    language: str,
    raw_dir: Path,
    hf_token: str | None,
    real_tag: str = "real_wf",
    fake_tag: str = "fake_wf",
) -> dict[str, dict[str, int]]:
    """Ingest WaveFake-style real/fake clips paired by shared utterance id.

    Unlike SEA-Spoof (one bonafide row + one spoof row per transcript),
    WaveFake exposes one real clip alongside *several* fake variants (one per
    vocoder, ``fake_values``) sharing the same utterance id (``id_field``,
    e.g. ``audio_id``). To keep pairing 1:1 -- so this project's existing
    paired-training machinery (``pair_transcripts.csv`` sidecar,
    :func:`dataset.split_assigner.assign_grouped_splits`) works unmodified --
    each utterance id is deterministically assigned exactly one vocoder
    (``hash(id) % len(fake_values)``), so across many utterances every
    vocoder appears roughly equally often, but each individual written pair
    is exactly one real clip + one fake clip sharing an id, exactly like a
    SEA-Spoof pair.

    Streamed in a single pass, unlike :func:`ingest_source`'s
    one-restart-per-source approach: scanning once for whichever label each
    row happens to carry is far cheaper here than restarting the HTTP stream
    from the beginning once per vocoder (this dataset is not gated and
    streams directly, unlike SEA-Spoof's Xet-only shards).

    Args:
        dataset: HuggingFace dataset repo id.
        split: Split name.
        id_field: Column giving the shared utterance id across real/fake rows.
        audio_field: Column holding the audio bytes.
        real_value: The label value (e.g. ``"R"``) identifying real rows.
        fake_values: All possible fake label values (e.g. ``("WF1", ..., "WF7")``).
        target_pairs: Number of matched real/fake pairs to ingest.
        language: Language subdirectory to write clips under.
        raw_dir: Root raw-data directory.
        hf_token: HuggingFace access token, or ``None`` for ungated datasets.
        real_tag: Filename prefix for real clips.
        fake_tag: Filename prefix for fake clips.

    Returns:
        ``{real_tag: summary, fake_tag: summary}``, each a dict with
        ``already_present``/``newly_written``/``skipped`` counts.
    """
    real_dir = raw_dir / language / "real"
    fake_dir = raw_dir / language / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing_pairs = min(_existing_clip_count(real_dir, real_tag), _existing_clip_count(fake_dir, fake_tag))
    remaining = max(0, target_pairs - existing_pairs)
    if remaining == 0:
        logger.info("Skipping WaveFake pairing (%s): already have %d/%d pair(s).", language, existing_pairs, target_pairs)
        summary = {"already_present": existing_pairs, "newly_written": 0, "skipped": 0}
        return {real_tag: summary, fake_tag: summary.copy()}

    logger.info(
        "Streaming %s (%s) for WaveFake-style pairing: need %d more pair(s), %d already present.",
        dataset, split, remaining, existing_pairs,
    )

    hf_dataset = load_dataset(dataset, split=split, streaming=True, token=hf_token)
    hf_dataset = hf_dataset.cast_column(audio_field, Audio(decode=False))

    def _assigned_fake_value(utterance_id: str) -> str:
        return fake_values[hash(utterance_id) % len(fake_values)]

    pending: dict[str, dict[str, dict]] = {}
    newly_written = 0
    rows_seen = 0
    skipped = 0
    progress = tqdm(total=remaining, desc=f"{language}/paired/{real_tag}+{fake_tag}")

    for row in hf_dataset:
        if newly_written >= remaining:
            break
        rows_seen += 1

        utterance_id = row.get(id_field)
        label_value = row.get("real_or_fake")
        if utterance_id is None or label_value is None:
            skipped += 1
            continue

        utterance_key = str(utterance_id)
        assigned_fake = _assigned_fake_value(utterance_key)
        if label_value != real_value and label_value != assigned_fake:
            continue  # Not this utterance's assigned vocoder -- skip without buffering.

        audio_entry = row.get(audio_field)
        if not audio_entry or not audio_entry.get("bytes"):
            skipped += 1
            continue

        entry = pending.setdefault(utterance_key, {})
        entry[label_value] = audio_entry

        if real_value not in entry or assigned_fake not in entry:
            continue

        pair_index = existing_pairs + newly_written
        real_path = real_dir / f"{real_tag}_{pair_index:06d}{_audio_extension(entry[real_value])}"
        fake_path = fake_dir / f"{fake_tag}_{pair_index:06d}{_audio_extension(entry[assigned_fake])}"

        try:
            _write_audio_bytes(real_path, entry[real_value])
            _write_audio_bytes(fake_path, entry[assigned_fake])
        except OSError as exc:
            logger.warning("Failed to write WaveFake pair %s / %s: %s", real_path, fake_path, exc)
            real_path.unlink(missing_ok=True)
            fake_path.unlink(missing_ok=True)
            skipped += 1
            pending.pop(utterance_key, None)
            continue

        # Flushed immediately (not batched to the end of the loop) so an
        # interrupted run never leaves audio files on disk whose pairing
        # can't be recovered -- an earlier batched version of this function
        # lost 100 pairs' transcript records this way when the process was
        # killed mid-run (see docs/research_log.md).
        _append_pair_transcripts(raw_dir / language, [(real_path.name, utterance_key), (fake_path.name, utterance_key)])
        newly_written += 1
        progress.update(1)
        pending.pop(utterance_key, None)

    progress.close()

    if newly_written < remaining:
        logger.warning(
            "Stream exhausted after scanning %d row(s): only found %d/%d WaveFake pair(s).",
            rows_seen, newly_written, remaining,
        )

    logger.info(
        "Finished WaveFake pairing %s/%s (%s): %d already present, %d newly written pair(s), "
        "%d skipped, %d row(s) scanned.",
        real_tag, fake_tag, language, existing_pairs, newly_written, skipped, rows_seen,
    )

    real_summary = {"already_present": existing_pairs, "newly_written": newly_written, "skipped": skipped}
    fake_summary = real_summary.copy()
    return {real_tag: real_summary, fake_tag: fake_summary}


def ingest_all() -> dict[str, dict[str, int]]:
    """Ingest every real and fake source dataset per ``CONFIG.ingestion``.

    Returns:
        Mapping of ``source_tag`` -> summary dict, as returned by :func:`ingest_source`.
    """
    hf_token = _resolve_hf_token(CONFIG.ingestion.hf_token_env_var)

    results: dict[str, dict[str, int]] = {}
    paired_fake_tags: set[str] = set()

    for real_source in CONFIG.ingestion.real:
        cosourced_fake = next(
            (
                fake_source
                for fake_source in CONFIG.ingestion.fake
                if fake_source.source_tag not in paired_fake_tags and _sources_form_pair(real_source, fake_source)
            ),
            None,
        )
        if cosourced_fake is None:
            results[real_source.source_tag] = ingest_source(
                source=real_source,
                language=CONFIG.ingestion.language,
                raw_dir=CONFIG.paths.raw_dir,
                hf_token=hf_token,
            )
            continue

        ingest_fn = ingest_paired_sources if CONFIG.ingestion.allow_pairing else ingest_cosourced_independent
        results.update(
            ingest_fn(
                real_source=real_source,
                fake_source=cosourced_fake,
                language=CONFIG.ingestion.language,
                raw_dir=CONFIG.paths.raw_dir,
                hf_token=hf_token,
            )
        )
        paired_fake_tags.add(cosourced_fake.source_tag)

    for source in CONFIG.ingestion.fake:
        if source.source_tag in paired_fake_tags:
            continue
        results[source.source_tag] = ingest_source(
            source=source,
            language=CONFIG.ingestion.language,
            raw_dir=CONFIG.paths.raw_dir,
            hf_token=hf_token,
        )

    return results


if __name__ == "__main__":
    ingest_all()
