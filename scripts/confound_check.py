"""Per-source confound audit: quantify what distinguishes the four ingestion
sources besides their real/fake label, to explain the cross-source holdout
factorial's open question (docs/research_log.md, 2026-07-24 entries) --
specifically, why (Kathbath, IndicSynth) generalizes far better than
(Kathbath, SEA-Spoof) as a training pairing.

Design rationale:
    An earlier one-off version of this script (not committed) found real
    clips carry ~3x more trailing zero-padding than fake clips on average,
    a plausible partial shortcut. This version extends that check to compare
    sources *within* each label -- specifically IndicSynth vs SEA-Spoof
    (the fake side) and Kathbath vs Common Voice (the real side) -- since
    the factorial showed the fake-source choice interacts with the
    real-source choice rather than acting as a fixed effect. Candidate
    explanations from the research log this script is built to test:

      - SEA-Spoof's much larger raw size / multi-system composition vs.
        IndicSynth's single TTS system -> higher *within-source variance* on
        these features for SEA-Spoof.
      - Residual language leakage in SEA-Spoof despite the ``language=hi``
        ingestion filter -> unusually high speaker/duration diversity.
      - Kathbath's broadcast/noisier channel vs Common Voice's crowd-sourced
        channel -> a real-side confound layered on top of the fake-side one.

    Reuses :func:`scripts.cross_source_holdout_check.infer_source_tag` so
    "source" means exactly the same groups analyzed there, and reads
    ``metadata.csv``'s ``duration``/``sample_rate``/``speaker`` columns
    directly (already computed during preprocessing) rather than
    re-deriving them, only touching audio on disk for the two per-clip
    signal statistics (trailing-silence ratio, frame-level silence ratio)
    that metadata.csv doesn't already carry.

Usage:
    python -m scripts.confound_check
    python -m scripts.confound_check --sample-size 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from configs.config import CONFIG
from scripts.cross_source_holdout_check import infer_source_tag
from utils.logger import get_logger
from utils.metadata_manager import MetadataManager

logger = get_logger(__name__)

DEFAULT_SAMPLE_SIZE = 300
FRAME_LENGTH = 2048
HOP_LENGTH = 512
SILENCE_TOP_DB = 30

FAKE_SOURCES = {"fake", "fake_sea"}
REAL_SOURCES = {"real", "real_cv_m", "real_cv_f"}


def trailing_zero_ratio(audio: np.ndarray) -> float:
    """Fraction of an audio array's trailing samples that are exactly zero (right-padding).

    Args:
        audio: 1-D processed audio array (already peak-normalized and length-padded).

    Returns:
        Trailing zero-sample count divided by total length, in ``[0, 1]``.
    """
    nonzero_indices = np.flatnonzero(audio)
    if nonzero_indices.size == 0:
        return 1.0
    last_nonzero = nonzero_indices[-1]
    trailing_zeros = audio.shape[0] - 1 - last_nonzero
    return float(trailing_zeros / audio.shape[0])


def frame_silence_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Fraction of fixed-length frames whose RMS energy falls below a silence threshold.

    Args:
        audio: 1-D processed audio array.
        sample_rate: Sample rate of ``audio``, in Hz.

    Returns:
        Fraction of frames classified as silent, in ``[0, 1]``.
    """
    intervals = librosa.effects.split(audio, top_db=SILENCE_TOP_DB, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    if intervals.size == 0:
        return 1.0
    voiced_samples = int(np.sum(intervals[:, 1] - intervals[:, 0]))
    return float(1.0 - voiced_samples / audio.shape[0])


def rms_loudness(audio: np.ndarray) -> float:
    """Root-mean-square amplitude of an audio array.

    Args:
        audio: 1-D processed audio array.

    Returns:
        RMS value.
    """
    return float(np.sqrt(np.mean(np.square(audio))))


def sample_signal_stats(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Compute per-clip signal statistics for a stratified sample of each source.

    Args:
        df: Metadata rows with ``source_tag`` and ``processed_path`` columns,
            restricted to rows where preprocessing has completed.
        sample_size: Max clips to sample per ``source_tag``.
        seed: RNG seed for reproducible sampling.

    Returns:
        DataFrame with one row per sampled clip: ``source_tag``,
        ``trailing_zero_ratio``, ``silence_ratio``, ``rms``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for source_tag, group in df.groupby("source_tag"):
        sampled = group.sample(n=min(sample_size, len(group)), random_state=rng.integers(0, 2**31 - 1))
        for _, record in sampled.iterrows():
            audio, sample_rate = sf.read(record["processed_path"], dtype="float32", always_2d=False)
            rows.append(
                {
                    "source_tag": source_tag,
                    "trailing_zero_ratio": trailing_zero_ratio(audio),
                    "silence_ratio": frame_silence_ratio(audio, sample_rate),
                    "rms": rms_loudness(audio),
                }
            )
        logger.info("Sampled signal stats for %d/%d clips from source '%s'.", len(sampled), len(group), source_tag)

    return pd.DataFrame(rows)


def summarize(signal_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-source mean/std for signal stats plus metadata-derived diversity stats.

    Args:
        signal_df: Output of :func:`sample_signal_stats`.
        full_df: Full metadata rows (all feature-extracted clips), used for
            duration/sample_rate/speaker-count stats that don't require
            touching audio on disk.

    Returns:
        One row per ``source_tag`` with mean/std of each statistic plus
        clip count and distinct speaker count.
    """
    signal_summary = signal_df.groupby("source_tag").agg(
        trailing_zero_mean=("trailing_zero_ratio", "mean"),
        trailing_zero_std=("trailing_zero_ratio", "std"),
        silence_ratio_mean=("silence_ratio", "mean"),
        silence_ratio_std=("silence_ratio", "std"),
        rms_mean=("rms", "mean"),
        rms_std=("rms", "std"),
    )

    metadata_summary = full_df.groupby("source_tag").agg(
        clip_count=("audio_id", "count"),
        distinct_speakers=("speaker", "nunique"),
        duration_mean=("duration", "mean"),
        duration_std=("duration", "std"),
    )

    return metadata_summary.join(signal_summary).round(4)


def main() -> None:
    """CLI: sample and summarize per-source confound statistics, saving results to outputs/.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Max clips per source to compute signal stats for. Default {DEFAULT_SAMPLE_SIZE}.",
    )
    args = parser.parse_args()

    manager = MetadataManager(CONFIG.paths.metadata_file)
    df = manager.dataframe
    df = df[df["processed"].astype(bool)].copy()
    df["source_tag"] = df["filepath"].apply(infer_source_tag)

    signal_df = sample_signal_stats(df, args.sample_size, seed=CONFIG.dataset.random_seed)
    summary = summarize(signal_df, df)

    output_dir = CONFIG.paths.output_dir / "confound_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "source_confound_summary.csv")
    with (output_dir / "source_confound_summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(json.loads(summary.to_json(orient="index")), file_handle, indent=2)

    print("\n=== Per-source confound summary ===")
    print(summary.to_string())

    fake_rows = summary.loc[summary.index.isin(FAKE_SOURCES)]
    real_rows = summary.loc[summary.index.isin(REAL_SOURCES)]
    print("\n=== Fake-side comparison (IndicSynth 'fake' vs SEA-Spoof 'fake_sea') ===")
    print(fake_rows.to_string())
    print("\n=== Real-side comparison (Kathbath 'real' vs Common Voice 'real_cv_m'/'real_cv_f') ===")
    print(real_rows.to_string())

    logger.info("Confound summary written to %s", output_dir)


if __name__ == "__main__":
    main()
