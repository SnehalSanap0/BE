"""Audit paired experiment metadata and emit a machine-readable report.

The command exits non-zero when a publication-critical invariant fails:
missing transcripts, incomplete real/fake transcript groups, or a transcript
appearing in more than one split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from configs.config import CONFIG
from utils.metadata_manager import MetadataManager


def build_audit(metadata_path: Path) -> dict[str, object]:
    """Return counts and leakage checks for one metadata manifest."""
    df = MetadataManager(metadata_path).dataframe
    featured = df[df["feature_extracted"].astype(bool)].copy()
    featured["transcript"] = featured["transcript"].fillna("").astype(str).str.strip()
    paired = featured[featured["transcript"] != ""]

    split_counts = featured.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict("index")
    transcript_split_counts = paired.groupby("transcript")["split"].nunique()
    label_sets = paired.groupby("transcript")["label"].agg(lambda values: sorted(set(values)))
    incomplete = label_sets[label_sets.map(set) != {"real", "fake"}]
    duplicate_paths = int(featured["filepath"].duplicated().sum())
    missing_feature_files = int((~featured["feature_path"].map(lambda value: Path(str(value)).is_file())).sum())

    failures: list[str] = []
    if featured.empty:
        failures.append("No feature-extracted rows exist.")
    if int((featured["transcript"] == "").sum()):
        failures.append("Feature-extracted rows with missing transcripts exist.")
    if int((transcript_split_counts > 1).sum()):
        failures.append("At least one transcript appears in multiple splits.")
    if len(incomplete):
        failures.append("At least one transcript group does not contain both labels.")
    if duplicate_paths:
        failures.append("Duplicate file paths exist in metadata.")
    if missing_feature_files:
        failures.append("Metadata points to missing feature files.")

    return {
        "metadata_path": str(metadata_path),
        "rows": int(len(df)),
        "feature_extracted_rows": int(len(featured)),
        "unique_transcripts": int(paired["transcript"].nunique()),
        "missing_transcript_rows": int((featured["transcript"] == "").sum()),
        "cross_split_transcripts": int((transcript_split_counts > 1).sum()),
        "incomplete_transcript_groups": int(len(incomplete)),
        "duplicate_filepaths": duplicate_paths,
        "missing_feature_files": missing_feature_files,
        "split_label_counts": split_counts,
        "config": {
            "random_seed": CONFIG.dataset.random_seed,
            "sample_rate": CONFIG.audio.sample_rate,
            "remove_silence": CONFIG.audio.remove_silence,
            "normalize": CONFIG.audio.normalize,
            "feature_type": CONFIG.features.type,
            "image_size": CONFIG.features.image_size,
        },
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=CONFIG.paths.metadata_file)
    parser.add_argument("--output", type=Path, default=CONFIG.paths.output_dir / "protocol_audit.json")
    args = parser.parse_args()

    report = build_audit(args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
