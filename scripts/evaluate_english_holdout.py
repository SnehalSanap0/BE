"""Evaluate the English model checkpoint against held-out ASVspoof 2019 LA clips streamed from Hugging Face."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from datasets import Audio, load_dataset
from deployment.inference_service import InferenceService

NUM_CLIPS_PER_CLASS = 50
ENGLISH_CHECKPOINT = Path("checkpoints/english_smoke/best_model.pt")
IMAGE_SIZE = 128


def main() -> None:
    print(f"Loading English model from {ENGLISH_CHECKPOINT} (image_size={IMAGE_SIZE})...")
    service = InferenceService(checkpoint_path=ENGLISH_CHECKPOINT, feature_image_size=IMAGE_SIZE)

    print("Connecting to Hugging Face dataset 'Bisher/ASVspoof_2019_LA' (streaming=True)...")
    base_ds = load_dataset(
        "Bisher/ASVspoof_2019_LA",
        split="validation",
        streaming=True,
    ).cast_column("audio", Audio(decode=False))

    reals_stream = base_ds.take(NUM_CLIPS_PER_CLASS)
    fakes_stream = base_ds.skip(2548).take(NUM_CLIPS_PER_CLASS)

    predictions = []
    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "sample.flac"

        print(f"\nEvaluating {NUM_CLIPS_PER_CLASS} REAL (bonafide) clips...")
        for index, row in enumerate(reals_stream, start=1):
            audio_bytes = row["audio"]["bytes"]
            temp_path.write_bytes(audio_bytes)
            try:
                res = service.predict(temp_path)
                is_correct = (res.predicted_label == "real")
                predictions.append({
                    "true_label": "real",
                    "predicted_label": res.predicted_label,
                    "confidence": res.confidence,
                    "system_id": row.get("system_id", "-"),
                    "file": row.get("audio_file_name", f"real_{index}"),
                    "is_correct": is_correct,
                })
                print(f"  [{index}/{NUM_CLIPS_PER_CLASS}] REAL -> predicted '{res.predicted_label}' (conf: {res.confidence:.1%}) {'✓' if is_correct else '✗'}")
            except Exception as exc:
                print(f"  [{index}/{NUM_CLIPS_PER_CLASS}] Error processing real: {exc}")

        print(f"\nEvaluating {NUM_CLIPS_PER_CLASS} FAKE (spoof) clips across diverse TTS/VC systems...")
        for index, row in enumerate(fakes_stream, start=1):
            audio_bytes = row["audio"]["bytes"]
            temp_path.write_bytes(audio_bytes)
            try:
                res = service.predict(temp_path)
                is_correct = (res.predicted_label == "fake")
                predictions.append({
                    "true_label": "fake",
                    "predicted_label": res.predicted_label,
                    "confidence": res.confidence,
                    "system_id": row.get("system_id", "-"),
                    "file": row.get("audio_file_name", f"fake_{index}"),
                    "is_correct": is_correct,
                })
                print(f"  [{index}/{NUM_CLIPS_PER_CLASS}] FAKE ({row.get('system_id', '-')}) -> predicted '{res.predicted_label}' (conf: {res.confidence:.1%}) {'✓' if is_correct else '✗'}")
            except Exception as exc:
                print(f"  [{index}/{NUM_CLIPS_PER_CLASS}] Error processing fake: {exc}")

    elapsed = time.time() - start_time

    # Calculate summary metrics
    total = len(predictions)
    reals = [p for p in predictions if p["true_label"] == "real"]
    fakes = [p for p in predictions if p["true_label"] == "fake"]

    tp = sum(p["predicted_label"] == "fake" and p["true_label"] == "fake" for p in predictions)
    fp = sum(p["predicted_label"] == "fake" and p["true_label"] == "real" for p in predictions)
    tn = sum(p["predicted_label"] == "real" and p["true_label"] == "real" for p in predictions)
    fn = sum(p["predicted_label"] == "real" and p["true_label"] == "fake" for p in predictions)

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    real_accuracy = tn / len(reals) if reals else 0
    fake_recall = tp / len(fakes) if fakes else 0

    print("\n" + "=" * 60)
    print("           ENGLISH MODEL HOLDOUT EVALUATION REPORT           ")
    print("=" * 60)
    print(f"Benchmark Dataset: ASVspoof 2019 LA (validation split)")
    print(f"Total Evaluated:   {total} clips ({len(reals)} real, {len(fakes)} fake)")
    print(f"Elapsed Time:      {elapsed:.1f}s ({elapsed / total:.2f}s per clip)\n")

    print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
    print(f"Real Specificity:  {real_accuracy * 100:.2f}% (genuine called real: {tn}/{len(reals)})")
    print(f"Fake Recall:       {fake_recall * 100:.2f}% (spoofs caught: {tp}/{len(fakes)})")
    print(f"False Alarms:      {fp}/{len(reals)} ({fp / len(reals) * 100:.1f}% genuine misclassified as fake)")
    print(f"Missed Spoofs:     {fn}/{len(fakes)} ({fn / len(fakes) * 100:.1f}% fake misclassified as real)")
    print(f"Precision:         {precision:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
