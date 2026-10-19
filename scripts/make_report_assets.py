"""Generate every figure and LaTeX table of the final-year project report from the saved result files.

Nothing is typed in by hand: each number is read from ``outputs/`` (experiment JSON/CSV files), so the
report cannot drift from the results. Re-run after any experiment changes:

    python -m scripts.make_report_assets

Writes PNG figures to ``Final_Year_Project_Report/figures/`` and ``\\input``-able table fragments to
``Final_Year_Project_Report/tables/``. Experiments that have not been run (missing files) are skipped with
a warning instead of failing.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from configs.config import CONFIG, ROOT_DIR  # noqa: E402
from forensics.verdict import load_validation_error_rates  # noqa: E402

OUTPUTS = ROOT_DIR / "outputs"
REPORT_DIR = ROOT_DIR / "Final_Year_Project_Report"
FIGURES = REPORT_DIR / "figures"
TABLES = REPORT_DIR / "tables"

PAIRED_COLOR = "#1f6fb2"
NONPAIRED_COLOR = "#e07b24"
GOOD, BAD = "#2e8b57", "#c0392b"
SEEDS = (42, 1337, 2026)
SOURCES = ("real", "real_cv_m", "real_cv_f", "fake")
SOURCE_NAMES = {
    "real": "Kathbath\n(genuine)",
    "real_cv_m": "Common Voice\nmale (genuine)",
    "real_cv_f": "Common Voice\nfemale (genuine)",
    "fake": "IndicSynth\n(synthetic)",
}
#: rotation -> (training genuine source, training synthetic source)
ROTATIONS = {
    "original": ("Kathbath", "IndicSynth"),
    "real_swap_only": ("Common Voice", "IndicSynth"),
    "fake_swap_only": ("Kathbath", "SEA-Spoof"),
    "swapped": ("Common Voice", "SEA-Spoof"),
}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [skip] missing {path.relative_to(ROOT_DIR)}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"  figure  {name}")


def _tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def _write_table(name: str, header: list[str], rows: list[list[str]], column_spec: str) -> None:
    """Write a bare ``tabular`` (no float): the report wraps it in a ``table`` with caption and label."""
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\hline"]
    lines.append(" & ".join(rf"\textbf{{{_tex_escape(cell)}}}" for cell in header) + r" \\ \hline")
    for row in rows:
        cells = [cell.replace(r"\%", "%").replace("%", r"\%") for cell in row]
        lines.append(" & ".join(cells) + r" \\ \hline")
    lines.append(r"\end{tabular}")
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  table   {name}.tex")


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


# --------------------------------------------------------------------------------------------------
# Hindi: 2x2 cross-source factorial and its diagnostics
# --------------------------------------------------------------------------------------------------
def factorial_assets() -> None:
    results = {name: _load(OUTPUTS / f"cross_source_holdout_{name}_metrics.json") for name in ROTATIONS}
    if not all(results.values()):
        return

    _write_table(
        "factorial",
        ["Rotation", "Trained on (genuine + synthetic)", "Tested on", "Accuracy", "ROC-AUC", "EER"],
        [
            [
                _tex_escape(name.replace("_", " ")),
                f"{real} + {fake}",
                "the other two sources",
                _fmt(m["accuracy"]),
                _fmt(m["roc_auc"]),
                _fmt(m["eer"]),
            ]
            for name, ((real, fake), m) in zip(ROTATIONS, ((ROTATIONS[n], results[n]) for n in ROTATIONS))
        ],
        "|l|l|l|r|r|r|",
    )

    # 2x2 heat map: rows = training genuine source, columns = training synthetic source.
    grid = {("Kathbath", "IndicSynth"): "original", ("Kathbath", "SEA-Spoof"): "fake_swap_only",
            ("Common Voice", "IndicSynth"): "real_swap_only", ("Common Voice", "SEA-Spoof"): "swapped"}
    rows, columns = ["Kathbath", "Common Voice"], ["IndicSynth", "SEA-Spoof"]
    eer = np.array([[results[grid[(r, c)]]["eer"] for c in columns] for r in rows])
    auc = np.array([[results[grid[(r, c)]]["roc_auc"] for c in columns] for r in rows])
    figure, axes = plt.subplots(figsize=(5.4, 4.2))
    image = axes.imshow(eer, cmap="RdYlGn_r", vmin=0, vmax=0.45)
    for i in range(2):
        for j in range(2):
            axes.text(j, i, f"EER {eer[i, j]:.1%}\nAUC {auc[i, j]:.3f}", ha="center", va="center", fontsize=11,
                      color="black", weight="bold")
    axes.set_xticks([0, 1], [f"synthetic: {c}" for c in columns])
    axes.set_yticks([0, 1], [f"genuine: {r}" for r in rows])
    axes.set_title("Cross-source EER by training pairing\n(tested on the two unseen sources)", fontsize=11)
    figure.colorbar(image, ax=axes, label="Equal error rate", shrink=0.8)
    for spine in axes.spines.values():
        spine.set_visible(False)
    _save_figure(figure, "fig_factorial.png")

    # Shortcut ratio (embedding space) vs. cross-source EER.
    ratios = {name: _load(OUTPUTS / "embedding_analysis" / f"{name}_silhouette.json") for name in ROTATIONS}
    if all(ratios.values()):
        order = sorted(ROTATIONS, key=lambda n: results[n]["eer"])
        _write_table(
            "shortcut",
            ["Rotation", "Silhouette (source)", "Silhouette (label)", "Shortcut ratio", "Cross-source EER"],
            [
                [
                    _tex_escape(n.replace("_", " ")),
                    _fmt(ratios[n]["silhouette_by_source_tag"]),
                    _fmt(ratios[n]["silhouette_by_label"]),
                    _fmt(ratios[n]["shortcut_ratio"], 3),
                    _fmt(results[n]["eer"]),
                ]
                for n in order
            ],
            "|l|r|r|r|r|",
        )
        figure, axes = plt.subplots(figsize=(5.6, 4.0))
        for name in ROTATIONS:
            axes.scatter(ratios[name]["shortcut_ratio"], results[name]["eer"], s=90, color=PAIRED_COLOR, zorder=3)
            axes.annotate(name.replace("_", " "), (ratios[name]["shortcut_ratio"], results[name]["eer"]),
                          textcoords="offset points", xytext=(7, 5), fontsize=9)
        axes.set_xlabel("Shortcut ratio  (silhouette by source / silhouette by label)")
        axes.set_ylabel("Cross-source EER")
        axes.set_title("Worse generalisation tracks a more source-driven\nembedding space", fontsize=11)
        axes.set_xlim(0.45, 1.1)
        axes.grid(alpha=0.3)
        _save_figure(figure, "fig_shortcut_vs_eer.png")

    for name in ("original", "fake_swap_only"):
        source = OUTPUTS / "embedding_analysis" / f"{name}_tsne.png"
        if source.exists():
            FIGURES.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, FIGURES / f"fig_tsne_{name}.png")
            print(f"  figure  fig_tsne_{name}.png (copied)")


def confound_assets() -> None:
    path = OUTPUTS / "confound_analysis" / "source_confound_summary.csv"
    if not path.exists():
        print(f"  [skip] missing {path.relative_to(ROOT_DIR)}")
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    names = {"real": "Kathbath (genuine)", "real_cv_m": "Common Voice male (genuine)",
             "real_cv_f": "Common Voice female (genuine)", "fake": "IndicSynth (synthetic)",
             "fake_sea": "SEA-Spoof (synthetic)"}
    order = ["real", "real_cv_m", "real_cv_f", "fake", "fake_sea"]
    rows = sorted(rows, key=lambda r: order.index(r["source_tag"]))
    _write_table(
        "confound",
        ["Source", "Clips", "Trailing padding", "Silence ratio", "RMS loudness"],
        [
            [
                names[r["source_tag"]],
                r["clip_count"],
                f"{float(r['trailing_zero_mean']):.3f} $\\pm$ {float(r['trailing_zero_std']):.3f}",
                f"{float(r['silence_ratio_mean']):.3f} $\\pm$ {float(r['silence_ratio_std']):.3f}",
                f"{float(r['rms_mean']):.3f}",
            ]
            for r in rows
        ],
        "|l|r|r|r|r|",
    )
    figure, axes = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(rows))
    colors = [BAD if r["source_tag"].startswith("fake") else GOOD for r in rows]
    means = [float(r["trailing_zero_mean"]) for r in rows]
    spreads = [float(r["trailing_zero_std"]) for r in rows]
    axes.bar(positions, means, yerr=spreads, color=colors, capsize=3, alpha=0.85)
    axes.set_xticks(positions, [names[r["source_tag"]].replace(" (", "\n(") for r in rows], fontsize=8)
    axes.set_ylabel("Share of clip that is trailing padding")
    axes.set_title("Recording-channel confound: padding differs by source\n(green = genuine, red = synthetic)",
                   fontsize=11)
    _save_figure(figure, "fig_confound.png")


# --------------------------------------------------------------------------------------------------
# Hindi: paired (E2) vs non-paired (E1) SEA-Spoof experiments
# --------------------------------------------------------------------------------------------------
def _cross_source(folder: str) -> dict[int, list[dict]] | None:
    data = {seed: _load(OUTPUTS / folder / f"cross_source_seed_{seed}.json") for seed in SEEDS}
    return None if not all(data.values()) else {seed: d["runs"] for seed, d in data.items()}


def pairing_assets() -> None:
    paired, nonpaired = _cross_source("paper_seaspoof_paired"), _cross_source("paper_seaspoof_nonpaired")
    if not paired or not nonpaired:
        return

    def per_model(runs_by_seed: dict, metric: str) -> list[float]:
        return [float(np.mean([run["overall"][metric] for run in runs])) for runs in runs_by_seed.values()]

    metrics = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("f1", "F1"),
               ("roc_auc", "ROC-AUC"), ("eer", "EER"), ("mcc", "MCC")]
    rows = []
    for key, label in metrics:
        e1, e2 = _mean_sd(per_model(nonpaired, key)), _mean_sd(per_model(paired, key))
        rows.append([label, f"{e1[0]:.4f} $\\pm$ {e1[1]:.4f}", f"{e2[0]:.4f} $\\pm$ {e2[1]:.4f}"])
    _write_table("pairing_cross_source", ["Metric", "E1 non-paired", "E2 paired"], rows, "|l|r|r|")

    def per_source(runs_by_seed: dict) -> dict[str, tuple[float, float]]:
        out = {}
        for source in SOURCES:
            model_means = [float(np.mean([run["per_source_accuracy"][source] for run in runs]))
                           for runs in runs_by_seed.values()]
            out[source] = _mean_sd(model_means)
        return out

    e1_source, e2_source = per_source(nonpaired), per_source(paired)
    _write_table(
        "pairing_per_source",
        ["External source", "True label", "E1 non-paired", "E2 paired"],
        [
            [
                SOURCE_NAMES[s].split("\n")[0] + {"real_cv_m": " male", "real_cv_f": " female"}.get(s, ""),
                "synthetic" if s == "fake" else "genuine",
                f"{e1_source[s][0]:.4f} $\\pm$ {e1_source[s][1]:.4f}",
                f"{e2_source[s][0]:.4f} $\\pm$ {e2_source[s][1]:.4f}",
            ]
            for s in SOURCES
        ],
        "|l|l|r|r|",
    )

    figure, axes = plt.subplots(figsize=(7.4, 4.0))
    width, positions = 0.38, np.arange(len(SOURCES))
    for offset, data, label, color in ((-width / 2, e1_source, "E1 non-paired", NONPAIRED_COLOR),
                                       (width / 2, e2_source, "E2 transcript-paired", PAIRED_COLOR)):
        axes.bar(positions + offset, [data[s][0] for s in SOURCES], width, yerr=[data[s][1] for s in SOURCES],
                 label=label, color=color, capsize=3)
    axes.set_xticks(positions, [SOURCE_NAMES[s] for s in SOURCES], fontsize=8.5)
    axes.set_ylabel("Share classified correctly")
    axes.set_ylim(0, 1.3)
    axes.axhline(1.0, color="grey", linewidth=0.6)
    axes.set_title("Unseen-source accuracy: non-paired vs transcript-paired training", fontsize=11)
    axes.legend(loc="upper center", ncol=2, frameon=False)
    _save_figure(figure, "fig_pairing_per_source.png")

    # Aggregate external confusion matrices over every model and sampling manifest.
    figure, grid = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for axes, runs_by_seed, title in ((grid[0], nonpaired, "E1 non-paired"), (grid[1], paired, "E2 paired")):
        matrix = sum(np.array(run["overall"]["confusion_matrix"]) for runs in runs_by_seed.values() for run in runs)
        axes.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                axes.text(j, i, f"{matrix[i, j]}\n({matrix[i, j] / matrix[i].sum():.0%})", ha="center",
                          va="center", color="white" if matrix[i, j] / matrix[i].sum() > 0.5 else "black")
        axes.set_xticks([0, 1], ["pred. genuine", "pred. synthetic"], fontsize=8.5)
        axes.set_yticks([0, 1], ["true genuine", "true synthetic"], fontsize=8.5)
        axes.set_title(title, fontsize=11)
        for spine in axes.spines.values():
            spine.set_visible(False)
    figure.suptitle("External test (all 3 models x 3 sampling seeds)", fontsize=10, y=1.02)
    _save_figure(figure, "fig_confusion_pairing.png")

    # In-domain results (same-corpus test split) for both conditions.
    rows = []
    for seed in SEEDS:
        row = [str(seed)]
        for folder in ("paper_seaspoof_nonpaired", "paper_seaspoof_paired"):
            data = _load(OUTPUTS / folder / f"in_domain_seed_{seed}.json")
            metrics_ = data["test_metrics"] if data else None
            row += [_fmt(metrics_["accuracy"]), _fmt(metrics_["eer"])] if metrics_ else ["--", "--"]
        rows.append(row)
    _write_table("pairing_in_domain", ["Seed", "E1 accuracy", "E1 EER", "E2 accuracy", "E2 EER"], rows,
                 "|c|r|r|r|r|")


# --------------------------------------------------------------------------------------------------
# English WaveFake smoke run
# --------------------------------------------------------------------------------------------------
def english_assets() -> None:
    metrics = _load(OUTPUTS / "english_smoke" / "smoke_test_metrics.json")
    if not metrics:
        return
    _write_table(
        "english",
        ["Metric", "Value"],
        [[label, _fmt(metrics[key])] for key, label in
         (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"),
          ("recall", "Recall"), ("f1", "F1"), ("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("eer", "EER"))],
        "|l|r|",
    )
    matrix = np.array(metrics["confusion_matrix"])
    figure, axes = plt.subplots(figsize=(3.6, 3.2))
    axes.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axes.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.sum() / 4 else "black", fontsize=13)
    axes.set_xticks([0, 1], ["pred. genuine", "pred. synthetic"], fontsize=8.5)
    axes.set_yticks([0, 1], ["true genuine", "true synthetic"], fontsize=8.5)
    axes.set_title("English (WaveFake) test split", fontsize=10)
    for spine in axes.spines.values():
        spine.set_visible(False)
    _save_figure(figure, "fig_confusion_english.png")


# --------------------------------------------------------------------------------------------------
# Forensic workflow: score distribution, thresholds, codec robustness
# --------------------------------------------------------------------------------------------------
def forensic_assets() -> None:
    scores_path = CONFIG.forensics.validation_scores
    data = _load(scores_path)
    if data:
        unique = {}
        for run in data["runs"]:
            for prediction in run["predictions"]:
                unique.setdefault(prediction["filepath"], prediction)
        by_source = {s: np.array([p["fake_score"] for p in unique.values() if p["source_tag"] == s]) for s in SOURCES}
        figure, grid = plt.subplots(1, 4, figsize=(10.5, 2.9), sharey=True)
        for axes, source in zip(grid, SOURCES):
            axes.hist(by_source[source], bins=np.linspace(0, 1, 21), color=BAD if source == "fake" else GOOD)
            axes.axvline(CONFIG.forensics.genuine_below, color="black", linestyle=":", linewidth=1)
            axes.axvline(CONFIG.forensics.synthetic_above, color="black", linestyle="--", linewidth=1)
            axes.set_title(SOURCE_NAMES[source].replace("\n", " "), fontsize=9)
            axes.set_xlabel("probability synthetic")
        grid[0].set_ylabel("clips")
        figure.suptitle("Model scores on the four unseen external sources "
                        "(dotted/dashed = verdict thresholds)", fontsize=10, y=1.04)
        _save_figure(figure, "fig_score_distribution.png")

        rates = load_validation_error_rates(
            scores_path, CONFIG.forensics.genuine_below, CONFIG.forensics.synthetic_above,
            expected_checkpoint=ROOT_DIR / data["checkpoint"].replace("\\", "/"), project_root=ROOT_DIR,
        )
        if rates:
            rows = [
                ["Genuine speech reported LIKELY SYNTHETIC (false alarm)", f"{rates.genuine_called_synthetic:.1%}"],
                ["Synthetic speech reported LIKELY GENUINE (miss)", f"{rates.synthetic_called_genuine:.1%}"],
                ["Genuine speech reported INCONCLUSIVE", f"{rates.genuine_inconclusive:.1%}"],
                ["Synthetic speech reported INCONCLUSIVE", f"{rates.synthetic_inconclusive:.1%}"],
            ] + [[f"Wrong decisive verdict: {_tex_escape(src)}", f"{rate:.1%}"]
                 for src, rate in sorted(rates.per_source_wrong.items())]
            _write_table("verdict_error_rates", ["Measure (unseen sources)", "Rate"], rows, "|l|r|")

    codec = _load(OUTPUTS / "codec_robustness" / "codec_robustness.json")
    if codec:
        results = codec["results"]
        _write_table(
            "codec",
            ["Encoding of the same clips", "Genuine flagged synthetic", "Synthetic caught"],
            [[_tex_escape(r["condition"]), f"{r['genuine_flagged']:.0%}", f"{r['fake_caught']:.0%}"]
             for r in results],
            "|l|r|r|",
        )
        figure, axes = plt.subplots(figsize=(7.6, 4.2))
        positions = np.arange(len(results))
        axes.barh(positions + 0.2, [r["fake_caught"] for r in results], 0.4, color=GOOD,
                  label="synthetic clips caught (higher is better)")
        axes.barh(positions - 0.2, [r["genuine_flagged"] for r in results], 0.4, color=BAD,
                  label="genuine clips wrongly flagged (lower is better)")
        axes.set_yticks(positions, [r["condition"] for r in results], fontsize=8.5)
        for position, r in zip(positions, results):
            axes.text(r["fake_caught"] + 0.01, position + 0.2, f"{r['fake_caught']:.0%}", va="center", fontsize=7.5)
            axes.text(r["genuine_flagged"] + 0.01, position - 0.2, f"{r['genuine_flagged']:.0%}", va="center",
                      fontsize=7.5, color=BAD)
        axes.invert_yaxis()
        axes.set_xlim(0, 1.1)
        axes.set_xlabel(f"Share of {codec['clips_per_class']} clips per class")
        axes.set_title("Effect of re-encoding on the same unseen clips", fontsize=11)
        axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False, fontsize=8.5)
        _save_figure(figure, "fig_codec_robustness.png")


def scenario_and_efficiency_assets() -> None:
    """Forensic scenario test results, inference benchmark, and a sample of the real report output."""
    scenarios = _load(OUTPUTS / "forensic_scenarios" / "scenario_results.json")
    if scenarios:
        rows = []
        for row in scenarios["rows"]:
            outcome = "met" if row["expectation_met"] else ("not met (known limitation)" if row["known_limitation"] else "NOT met")
            rows.append([_tex_escape(row["sample"]), _tex_escape(row["expected"].replace("NOT_SYNTHETIC", "not synthetic")),
                         _tex_escape(row["verdict"]), f"{row['fake_score']:.3f}",
                         f"{row['windows_flagged_synthetic']}/{row['windows_scored']}", outcome])
        _write_table("scenarios", ["Sample", "Expected", "Verdict", "Score", "Windows flagged", "Expectation"], rows,
                     "|l|l|l|r|r|l|")
        for source, target in (("spliced_call_timeline.png", "fig_spliced_timeline.png"),
                               ("spliced_call_report.pdf", "forensic_sample_report.pdf")):
            path = OUTPUTS / "forensic_scenarios" / source
            if path.exists():
                FIGURES.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, FIGURES / target)
                print(f"  figure  {target} (copied)")

    benchmark = _load(OUTPUTS / "efficiency" / "inference_benchmark.json")
    if benchmark:
        cpu = next(r for r in benchmark["results"] if r["device"] == "cpu")
        rows = [
            ["Parameters", f"{cpu['parameters'] / 1e6:.2f} million"],
            ["Weights only (float32)", f"{cpu['parameters'] * 4 / 1e6:.1f} MB"],
            ["Checkpoint file (weights + optimiser state)", f"{cpu['checkpoint_mb']:.1f} MB"],
        ]
        for result in benchmark["results"]:
            name = "GPU" if result["device"].startswith("cuda") else "CPU"
            rows += [
                [f"{name}: feature extraction per 4 s window", f"{result['feature_extraction_ms_per_window']:.1f} ms"],
                [f"{name}: forward pass, one window", f"{result['forward_ms_per_window']:.1f} ms"],
                [f"{name}: forward pass, per window in a batch of 32", f"{result['forward_ms_per_window_batched']:.2f} ms"],
                [f"{name}: Grad-CAM explanation of one window", f"{result['gradcam_ms_per_window']:.0f} ms"],
            ]
            if "peak_gpu_memory_mb" in result:
                rows.append([f"{name}: peak memory during the benchmark", f"{result['peak_gpu_memory_mb']:.0f} MB"])
        _write_table("efficiency", ["Measure", "Value"], rows, "|l|r|")


def gradcam_assets() -> None:
    """Side-by-side Grad-CAM overlays for a matched genuine/synthetic pair, from files the CLI already wrote."""
    pairs = [("Genuine", OUTPUTS / "explanations" / "real_sea_000000_gradcam.png"),
             ("Synthetic", OUTPUTS / "explanations" / "fake_sea_000000_gradcam.png")]
    if not all(path.exists() for _, path in pairs):
        print("  [skip] Grad-CAM example images missing")
        return
    figure, grid = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for axes, (title, path) in zip(grid, pairs):
        axes.imshow(plt.imread(path))
        axes.set_title(f"{title} clip (same transcript)", fontsize=10)
        axes.axis("off")
    _save_figure(figure, "fig_gradcam_pair.png")


def main() -> None:
    steps = [("Hindi cross-source factorial", factorial_assets), ("Recording-channel confound", confound_assets),
             ("Paired vs non-paired", pairing_assets), ("English WaveFake", english_assets),
             ("Forensic workflow and codecs", forensic_assets), ("Scenarios and efficiency", scenario_and_efficiency_assets),
             ("Grad-CAM example", gradcam_assets)]
    for title, step in steps:
        print(title)
        step()
    print(f"Done. Figures: {FIGURES}\nTables:  {TABLES}")


if __name__ == "__main__":
    main()
