"""Save side-by-side real/spoof spectrogram PNGs for matched SEA-Spoof pairs.

The smoke ingestion writes paired clips with matching numeric suffixes:
``real_sea_000123.wav`` and ``fake_sea_000123.wav``. This script finds those
pairs in metadata, loads their extracted ``.npy`` feature arrays, and saves
comparison images so you can inspect what the CNN sees.

Usage:
    python -m scripts.plot_paired_spectrograms --limit 10
    python -m scripts.plot_paired_spectrograms --pair-id 000042
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = ROOT_DIR / "configs" / "config.smoke_seaspoof.yaml"
os.environ.setdefault("DEEPFAKE_CONFIG_PATH", str(SMOKE_CONFIG_PATH))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import CONFIG  # noqa: E402


PAIR_PATTERN = re.compile(r"^(?:real|fake)_sea_(\d{6})")


def _pair_id(filepath: str) -> str | None:
    """Extract the paired SEA-Spoof numeric id from a raw filepath."""
    match = PAIR_PATTERN.match(Path(filepath).name)
    return match.group(1) if match else None


def _load_pairs(metadata_path: Path) -> pd.DataFrame:
    """Load feature-extracted real/fake rows that have a matching pair id."""
    df = pd.read_csv(metadata_path, keep_default_na=False)
    df = df[df["feature_extracted"].astype(bool)].copy()
    df["pair_id"] = df["filepath"].map(_pair_id)
    df = df[df["pair_id"].notna()]
    return df


def _save_pair_plot(pair_id: str, real_path: Path, fake_path: Path, output_path: Path) -> None:
    """Render one real/fake feature pair to a PNG."""
    real_feature = np.load(real_path)
    fake_feature = np.load(fake_path)
    difference = fake_feature - real_feature

    vmax = max(float(np.abs(real_feature).max()), float(np.abs(fake_feature).max()), 1e-6)
    diff_vmax = max(float(np.abs(difference).max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].imshow(real_feature, origin="lower", aspect="auto", cmap="magma", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"real {pair_id}")

    axes[1].imshow(fake_feature, origin="lower", aspect="auto", cmap="magma", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"spoof {pair_id}")

    axes[2].imshow(difference, origin="lower", aspect="auto", cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax)
    axes[2].set_title("spoof - real")

    for axis in axes:
        axis.set_xlabel("time")
        axis.set_ylabel("mel bin")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=CONFIG.paths.metadata_file)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.paths.output_dir / "paired_spectrograms")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--pair-id", help="Specific numeric pair id, e.g. 000042.")
    args = parser.parse_args()

    df = _load_pairs(args.metadata)
    if args.pair_id:
        df = df[df["pair_id"] == args.pair_id]

    written = 0
    for pair_id, pair_df in df.groupby("pair_id", sort=True):
        by_label = {str(row["label"]): row for _, row in pair_df.iterrows()}
        if "real" not in by_label or "fake" not in by_label:
            continue

        real_feature_path = Path(str(by_label["real"]["feature_path"]))
        fake_feature_path = Path(str(by_label["fake"]["feature_path"]))
        if not real_feature_path.exists() or not fake_feature_path.exists():
            continue

        output_path = args.output_dir / f"pair_{pair_id}.png"
        _save_pair_plot(pair_id, real_feature_path, fake_feature_path, output_path)
        print(output_path)

        written += 1
        if written >= args.limit:
            break

    if written == 0:
        raise SystemExit("No complete paired feature rows found. Run the smoke pipeline first.")


if __name__ == "__main__":
    main()
