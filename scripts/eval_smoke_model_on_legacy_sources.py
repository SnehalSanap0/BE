"""Quick sanity check: does the SEA-Spoof-paired smoke model generalize at all
to the project's four earlier, never-trained-on sources (Kathbath, Common
Voice Hindi male/female, IndicSynth)?

This is deliberately lightweight compared to
:mod:`scripts.cross_source_holdout_check` -- it does not retrain anything. It
loads the existing ``checkpoints/smoke_seaspoof_paired/best_model.pt``
checkpoint, draws a random sample of raw clips per source straight from
``data/raw/hindi/{real,fake}``, and reports plain accuracy. It's a smoke test
of generalization, not a paper-grade result (no fixed protocol, no repeated
seeds) -- if this looks promising, follow up with a proper cross-source
holdout run.

Must be run with the smoke config active, so feature extraction (image_size,
n_mels, etc.) matches what the checkpoint was trained on:

    $env:DEEPFAKE_CONFIG_PATH="configs/config.smoke_seaspoof.yaml"
    python -m scripts.eval_smoke_model_on_legacy_sources --samples-per-source 15
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from configs.config import CONFIG
from dataset.deepfake_dataset import INDEX_TO_LABEL
from feature_extraction.extractor import FeatureExtractor
from models.model_factory import build_model
from preprocessing.transforms import preprocess_audio
from preprocessing.validator import AudioValidator
from training.checkpoint import BEST_CHECKPOINT_FILENAME, CheckpointManager
from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hindi"

SOURCE_TAG_PATTERN = re.compile(r"([a-z_]+?)_\d{6}")

#: The four sources the smoke model has never seen (excludes real_sea/fake_sea,
#: which are SEA-Spoof -- the dataset the smoke model was trained on).
LEGACY_SOURCES: dict[str, str] = {
    "real": "real",  # Kathbath
    "real_cv_m": "real",  # Common Voice Hindi male
    "real_cv_f": "real",  # Common Voice Hindi female
    "fake": "fake",  # IndicSynth
}


def infer_source_tag(filepath: Path) -> str:
    """Recover a file's ingestion source_tag from its filename prefix."""
    match = SOURCE_TAG_PATTERN.match(filepath.stem)
    return match.group(1) if match else filepath.stem


def collect_legacy_files() -> dict[str, list[Path]]:
    """Group raw legacy clips by source_tag, restricted to :data:`LEGACY_SOURCES`."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for label_dir in ("real", "fake"):
        for filepath in (LEGACY_RAW_DIR / label_dir).iterdir():
            if not filepath.is_file():
                continue
            source_tag = infer_source_tag(filepath)
            if source_tag in LEGACY_SOURCES:
                grouped[source_tag].append(filepath)
    return grouped


def predict(
    audio_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    feature_extractor: FeatureExtractor,
) -> tuple[str, float]:
    """Predict a single raw audio file's label, without Grad-CAM overhead."""
    validation = AudioValidator().validate(audio_path)
    if not validation.is_valid:
        raise ValueError(f"Invalid audio file '{audio_path}': {validation.reason}")

    raw_audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    processed_audio = preprocess_audio(raw_audio, sample_rate)
    feature = feature_extractor.extract_array(processed_audio, CONFIG.audio.sample_rate)

    input_tensor = torch.from_numpy(feature).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(torch.argmax(logits, dim=1).item())
        confidence = float(probabilities[0, predicted_index].item())

    return INDEX_TO_LABEL[predicted_index], confidence


def main() -> None:
    """Sample legacy-source clips, run them through the smoke checkpoint, and report accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-per-source", type=int, default=15,
        help="Random clips to draw from each of the four legacy sources. Default 15.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed. Default 42.")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Override checkpoint path. Defaults to CONFIG.paths.checkpoint_dir/best_model.pt "
             "(checkpoints/smoke_seaspoof_paired/best_model.pt when the smoke config is active).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_manager = CheckpointManager(CONFIG.paths.checkpoint_dir)
    checkpoint_path = args.checkpoint or (CONFIG.paths.checkpoint_dir / BEST_CHECKPOINT_FILENAME)
    state = checkpoint_manager.load(checkpoint_path, map_location=device)

    use_bilstm = state.get("extra", {}).get("use_bilstm", CONFIG.model.use_bilstm)
    model = build_model(use_bilstm=use_bilstm)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    feature_extractor = FeatureExtractor()

    grouped_files = collect_legacy_files()
    missing = set(LEGACY_SOURCES) - set(grouped_files)
    if missing:
        raise FileNotFoundError(
            f"No files found on disk for legacy source_tag(s) {missing} under {LEGACY_RAW_DIR}."
        )

    rng = random.Random(args.seed)
    results: list[dict] = []
    for source_tag, true_label in LEGACY_SOURCES.items():
        candidates = grouped_files[source_tag]
        sample = rng.sample(candidates, k=min(args.samples_per_source, len(candidates)))
        for filepath in sample:
            predicted_label, confidence = predict(filepath, model, device, feature_extractor)
            results.append(
                {
                    "source_tag": source_tag,
                    "filepath": str(filepath.relative_to(PROJECT_ROOT)),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "correct": predicted_label == true_label,
                }
            )
            logger.info(
                "[%s] %s: true=%s predicted=%s (confidence=%.4f) %s",
                source_tag, filepath.name, true_label, predicted_label, confidence,
                "OK" if predicted_label == true_label else "WRONG",
            )

    per_source_accuracy = {}
    for source_tag in LEGACY_SOURCES:
        source_results = [r for r in results if r["source_tag"] == source_tag]
        correct = sum(r["correct"] for r in source_results)
        per_source_accuracy[source_tag] = correct / len(source_results) if source_results else float("nan")

    overall_accuracy = sum(r["correct"] for r in results) / len(results)
    confusion = defaultdict(int)
    for r in results:
        confusion[f"true_{r['true_label']}_pred_{r['predicted_label']}"] += 1

    summary = {
        "checkpoint": str(checkpoint_path),
        "samples_per_source": args.samples_per_source,
        "seed": args.seed,
        "overall_accuracy": overall_accuracy,
        "per_source_accuracy": per_source_accuracy,
        "confusion_counts": dict(confusion),
        "results": results,
    }

    output_path = CONFIG.paths.output_dir / "legacy_source_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)

    logger.info("Overall accuracy on unseen legacy sources: %.4f", overall_accuracy)
    logger.info("Per-source accuracy: %s", per_source_accuracy)
    logger.info("Saved full results to %s", output_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
