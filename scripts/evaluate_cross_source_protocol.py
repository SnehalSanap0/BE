"""Repeatable cross-source evaluation for a trained paired model.

Sampling manifests and clip-level scores are saved for auditability. Seeds
here control evaluation sampling only; publication tables must additionally
aggregate independently trained model seeds.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import compute_metrics
from feature_extraction.extractor import FeatureExtractor
from models.model_factory import build_model
from scripts.eval_smoke_model_on_legacy_sources import LEGACY_SOURCES, collect_legacy_files, predict
from training.checkpoint import BEST_CHECKPOINT_FILENAME, CheckpointManager
from configs.config import CONFIG


def _evaluate_seed(
    seed: int,
    samples_per_source: int,
    grouped_files: dict[str, list[Path]],
    model: torch.nn.Module,
    device: torch.device,
    extractor: FeatureExtractor,
) -> dict[str, object]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for source_tag, true_label in LEGACY_SOURCES.items():
        candidates = sorted(grouped_files[source_tag])
        selected = rng.sample(candidates, min(samples_per_source, len(candidates)))
        for path in selected:
            predicted, confidence = predict(path, model, device, extractor)
            fake_score = confidence if predicted == "fake" else 1.0 - confidence
            rows.append({
                "source_tag": source_tag,
                "filepath": str(path),
                "true_label": true_label,
                "predicted_label": predicted,
                "fake_score": fake_score,
            })

    y_true = np.array([int(row["true_label"] == "fake") for row in rows])
    y_pred = np.array([int(row["predicted_label"] == "fake") for row in rows])
    scores = np.array([row["fake_score"] for row in rows])
    overall = compute_metrics(y_true, y_pred, scores)
    per_source_accuracy = {
        tag: float(np.mean([row["predicted_label"] == row["true_label"] for row in rows if row["source_tag"] == tag]))
        for tag in LEGACY_SOURCES
    }

    # Each real corpus is paired with the same unseen IndicSynth subset to
    # produce a two-class cross-domain EER/AUC, rather than an undefined
    # single-class per-source value.
    pairwise: dict[str, object] = {}
    fake_rows = [row for row in rows if row["source_tag"] == "fake"]
    for real_tag in ("real", "real_cv_m", "real_cv_f"):
        domain_rows = [row for row in rows if row["source_tag"] == real_tag] + fake_rows
        domain_true = np.array([int(row["true_label"] == "fake") for row in domain_rows])
        domain_pred = np.array([int(row["predicted_label"] == "fake") for row in domain_rows])
        domain_scores = np.array([row["fake_score"] for row in domain_rows])
        pairwise[f"{real_tag}_vs_indicsynth"] = compute_metrics(domain_true, domain_pred, domain_scores)

    return {
        "sampling_seed": seed,
        "samples_per_source": samples_per_source,
        "overall": overall,
        "per_source_accuracy": per_source_accuracy,
        "pairwise_domains": pairwise,
        "predictions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-source", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=CONFIG.paths.output_dir / "cross_source_protocol.json")
    args = parser.parse_args()

    grouped = collect_legacy_files()
    missing = set(LEGACY_SOURCES) - set(grouped)
    if missing:
        raise FileNotFoundError(f"Missing legacy source(s): {sorted(missing)}")
    checkpoint = args.checkpoint or CONFIG.paths.checkpoint_dir / BEST_CHECKPOINT_FILENAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = CheckpointManager(checkpoint.parent).load(checkpoint, map_location=device)
    model = build_model(use_bilstm=state.get("extra", {}).get("use_bilstm", CONFIG.model.use_bilstm))
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    extractor = FeatureExtractor()

    runs = [
        _evaluate_seed(seed, args.samples_per_source, grouped, model, device, extractor)
        for seed in args.seeds
    ]
    metric_names = ("accuracy", "balanced_accuracy", "f1", "roc_auc", "eer", "mcc")
    aggregate = {
        name: {
            "mean": float(np.mean([run["overall"][name] for run in runs])),
            "std": float(np.std([run["overall"][name] for run in runs], ddof=1)) if len(runs) > 1 else 0.0,
        }
        for name in metric_names
    }
    report = {
        "checkpoint": str(checkpoint),
        "note": "Seeds vary evaluation sampling, not model training.",
        "aggregate": aggregate,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
