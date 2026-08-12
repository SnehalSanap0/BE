"""Generate publication figures for the ICMLDE manuscript from saved results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_icmlde" / "PROCS_ICMLDE2024_LATEX_Templates" / "figures"
RESULTS = ROOT / "outputs" / "paper_seaspoof_paired"
METADATA = ROOT / "data" / "paper_seaspoof_paired" / "metadata" / "metadata.csv"


def save_pipeline() -> None:
    fig, axis = plt.subplots(figsize=(12, 5.0))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")

    def add_box(x: float, y: float, label: str, color: str, width: float = 1.75) -> None:
        box = FancyBboxPatch(
            (x, y), width, 0.72,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2, edgecolor="#333333", facecolor=color,
        )
        axis.add_patch(box)
        axis.text(x + width / 2, y + 0.36, label, ha="center", va="center", fontsize=8.5)

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.1, color="#333333",
        ))

    # Training path.
    train_y = 3.55
    train_x = [0.15, 2.15, 4.15, 6.15, 8.15, 10.15]
    train_labels = [
        "SEA-Spoof train\nHindi filter",
        "Match real/spoof\nby transcript",
        "Transcript-group split\n80/10/10",
        "16 kHz, trim,\nnormalize, 4 s",
        "Log-mel (128 bands)\nresize to 224 x 224",
        "EfficientNet-B0\nbest validation F1",
    ]
    train_colors = ["#DCEAF7", "#DCEAF7", "#DCEAF7", "#E8F3E8", "#FFF0CC", "#E9E0F5"]
    for x, label, color in zip(train_x, train_labels, train_colors):
        add_box(x, train_y, label, color, width=1.7)
    for left, right in zip(train_x[:-1], train_x[1:]):
        arrow((left + 1.7, train_y + 0.36), (right - 0.08, train_y + 0.36))
    axis.text(0.15, 4.55, "Training and checkpoint selection", fontsize=10, fontweight="bold")

    # External evaluation path. The same transform is applied, then the fixed
    # checkpoint is scored without retraining.
    eval_y = 1.15
    add_box(0.15, eval_y, "External sources\nKathbath, CV, IndicSynth", "#F8DDDD", width=2.2)
    add_box(3.0, eval_y, "Same preprocessing\nand feature extraction", "#E8F3E8", width=2.2)
    add_box(6.0, eval_y, "Frozen best checkpoint\n(no retraining)", "#E9E0F5", width=2.2)
    add_box(9.0, eval_y, "Per-source metrics\nAUC, EER, bal. accuracy", "#F8DDDD", width=2.4)
    arrow((2.35, eval_y + 0.36), (2.92, eval_y + 0.36))
    arrow((5.2, eval_y + 0.36), (5.92, eval_y + 0.36))
    arrow((8.2, eval_y + 0.36), (8.92, eval_y + 0.36))
    arrow((11.0, train_y), (7.1, eval_y + 0.72))
    axis.text(0.15, 2.15, "Cross-source evaluation", fontsize=10, fontweight="bold")
    fig.savefig(OUTPUT / "pipeline.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_source_accuracy() -> None:
    tags = ["real", "real_cv_m", "real_cv_f", "fake"]
    display = ["Kathbath", "Common Voice\nmale", "Common Voice\nfemale", "IndicSynth"]
    model_values: dict[str, list[float]] = {tag: [] for tag in tags}
    for path in sorted(RESULTS.glob("cross_source_seed_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for tag in tags:
            values = [float(run["per_source_accuracy"][tag]) for run in report["runs"]]
            model_values[tag].append(float(np.mean(values)))
    means = np.array([np.mean(model_values[tag]) for tag in tags]) * 100
    standard_deviations = np.array([np.std(model_values[tag], ddof=1) for tag in tags]) * 100

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(
        display, means, yerr=standard_deviations, capsize=4,
        color=["#D95F5F", "#4C78A8", "#72A65A", "#E3A33D"],
        edgecolor="#333333", linewidth=0.8,
    )
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 108)
    axis.set_title("Cross-source accuracy by dataset (three trained models)")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f"{value:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "source_accuracy.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "source_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_paired_spectrogram(pair_id: str = "000002") -> None:
    metadata = pd.read_csv(METADATA, keep_default_na=False)
    subset = metadata[metadata["filepath"].map(lambda value: f"_sea_{pair_id}." in Path(value).name)]
    by_label = {str(row["label"]): row for _, row in subset.iterrows()}
    if set(by_label) != {"real", "fake"}:
        raise RuntimeError(f"Pair {pair_id} is incomplete in {METADATA}.")
    real = np.load(Path(by_label["real"]["feature_path"]))
    fake = np.load(Path(by_label["fake"]["feature_path"]))
    difference = fake - real
    value_min = min(float(real.min()), float(fake.min()))
    value_max = max(float(real.max()), float(fake.max()))
    diff_max = max(float(np.abs(difference).max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    axes[0].imshow(real, origin="lower", aspect="auto", cmap="magma", vmin=value_min, vmax=value_max)
    axes[1].imshow(fake, origin="lower", aspect="auto", cmap="magma", vmin=value_min, vmax=value_max)
    axes[2].imshow(difference, origin="lower", aspect="auto", cmap="coolwarm", vmin=-diff_max, vmax=diff_max)
    for axis, title in zip(axes, ["(a) Bona fide", "(b) Spoof", "(c) Spoof - bona fide"]):
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Resized frequency coordinate")
        axis.set_xlabel("Resized time coordinate")
    fig.savefig(OUTPUT / "paired_spectrogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_pipeline()
    save_source_accuracy()
    save_paired_spectrogram()
    print(f"Saved ICMLDE figures to {OUTPUT}")


if __name__ == "__main__":
    main()
