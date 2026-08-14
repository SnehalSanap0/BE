"""Verify that transcript-paired real/fake clips actually share the same transcript.

Design rationale:
    ``preprocessing.hf_ingestion.ingest_paired_sources`` pairs real/fake clips
    by matching filename index (``real_sea_000042.*`` / ``fake_sea_000042.*``)
    and, as of the ``transcript`` metadata column, also records the transcript
    each clip was selected/paired by (see ``pair_transcripts.csv`` sidecar and
    ``preprocessing.metadata_builder``). This script is the direct check: for
    every matched-index pair under two given source tags, confirm both sides'
    ``transcript`` values in metadata.csv are identical and non-empty, rather
    than only trusting that the ingestion code paired them correctly.

    Requires metadata.csv to have been (re)built via
    ``preprocessing.metadata_builder.build_initial_metadata`` *after* the
    transcript sidecar was written -- rows registered before that will show
    an empty transcript and are reported separately, not as mismatches.

Usage:
    python -m scripts.verify_seaspoof_pairing
    python -m scripts.verify_seaspoof_pairing --real-tag real_sea --fake-tag fake_sea
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from configs.config import CONFIG
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)


def _index_by_pair_number(df: pd.DataFrame, source_tag: str) -> dict[str, tuple[str, str]]:
    """Map pair index -> (transcript, filepath) for one source_tag's rows.

    Args:
        df: Full metadata table.
        source_tag: Filename prefix identifying one side of a paired source
            (e.g. ``"real_sea"``).

    Returns:
        Mapping of the zero-padded numeric suffix (e.g. ``"001500"``) to
        that row's ``(transcript, filepath)``.
    """
    pattern = re.compile(rf"^{re.escape(source_tag)}_(\d+)\.")
    indexed: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        name = Path(str(row["filepath"])).name
        match = pattern.match(name)
        if match:
            indexed[match.group(1)] = (str(row["transcript"]), str(row["filepath"]))
    return indexed


def verify_pairing(real_tag: str, fake_tag: str) -> None:
    """Check every matched-index pair between two source tags for transcript agreement.

    Args:
        real_tag: Source tag for the real/bonafide side (e.g. ``"real_sea"``).
        fake_tag: Source tag for the fake/spoof side (e.g. ``"fake_sea"``).
    """
    manager = MetadataManager(CONFIG.paths.metadata_file)
    df = manager.dataframe

    real_index = _index_by_pair_number(df, real_tag)
    fake_index = _index_by_pair_number(df, fake_tag)

    common_indices = sorted(set(real_index) & set(fake_index))
    if not common_indices:
        logger.warning(
            "No matched-index pairs found between '%s' and '%s' in %s. "
            "Has ingestion run for these source tags?",
            real_tag, fake_tag, CONFIG.paths.metadata_file,
        )
        return

    matched = 0
    empty_transcript = 0
    mismatched: list[tuple[str, str, str]] = []

    for index in common_indices:
        real_transcript, real_path = real_index[index]
        fake_transcript, fake_path = fake_index[index]

        if not real_transcript or not fake_transcript:
            empty_transcript += 1
            continue

        if real_transcript == fake_transcript:
            matched += 1
        else:
            mismatched.append((index, real_transcript, fake_transcript))

    logger.info(
        "Checked %d matched-index pair(s) between '%s' and '%s': "
        "%d confirmed same transcript, %d missing transcript (rebuild metadata to fix), %d MISMATCHED.",
        len(common_indices), real_tag, fake_tag, matched, empty_transcript, len(mismatched),
    )

    if empty_transcript:
        logger.info(
            "%d pair(s) have an empty transcript field -- these were likely registered in "
            "metadata.csv before the 'transcript' column existed. Delete their rows (or the "
            "whole metadata.csv, if you can afford a full rebuild) and rerun "
            "preprocessing.metadata_builder to backfill from pair_transcripts.csv.",
            empty_transcript,
        )

    if mismatched:
        logger.error("Found %d pair(s) with DIFFERENT transcripts -- pairing is broken for these:", len(mismatched))
        for index, real_transcript, fake_transcript in mismatched[:10]:
            logger.error("  index %s: real=%r vs fake=%r", index, real_transcript, fake_transcript)
        if len(mismatched) > 10:
            logger.error("  ... and %d more.", len(mismatched) - 10)
    else:
        logger.info("No mismatches found among pairs with a recorded transcript.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-tag", default="real_sea", help="Source tag for the real/bonafide side.")
    parser.add_argument("--fake-tag", default="fake_sea", help="Source tag for the fake/spoof side.")
    args = parser.parse_args()
    verify_pairing(args.real_tag, args.fake_tag)


if __name__ == "__main__":
    main()
