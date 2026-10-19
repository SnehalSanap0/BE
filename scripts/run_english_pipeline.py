"""End-to-end multi-source English pipeline: WaveFake + ASVspoof 2019 LA.

Ingests multi-source paired English audio, preprocesses, extracts 224x224
mel-spectrograms, assigns grouped splits without pair leakage, trains
EfficientNet-B0 with mixed precision on GPU, evaluates, and writes the official
cross-source validation scores artifact so the English model is fully validated.

Usage:
    python -m scripts.run_english_pipeline
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "config.english.yaml"
os.environ["DEEPFAKE_CONFIG_PATH"] = str(CONFIG_PATH)

import torch  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402

from configs.config import CONFIG  # noqa: E402
from dataset.dataloader import build_dataloader  # noqa: E402
from dataset.split_assigner import assign_grouped_splits  # noqa: E402
from deployment.inference_service import InferenceService  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from feature_extraction.pipeline import FeatureExtractionPipeline  # noqa: E402
from models.model_factory import build_model  # noqa: E402
from preprocessing.metadata_builder import build_initial_metadata  # noqa: E402
from preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.metadata_manager import MetadataManager  # noqa: E402

logger = get_logger(__name__)

TARGET_ASVSPOOF_PAIRS = 500  # 500 real + 500 fake = 1000 clips for diverse multi-speaker acoustics


def _ensure_raw_multisource_data() -> None:
    """Prepare WaveFake and stream ASVspoof 2019 LA paired clips into data/english/raw/english."""
    raw_root = CONFIG.paths.raw_dir / "english"
    real_dir = raw_root / "real"
    fake_dir = raw_root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = raw_root / "pair_transcripts.csv"

    # 1. Source 1: WaveFake (copy from existing smoke data if present to save download time)
    smoke_raw = ROOT_DIR / "data" / "english_smoke" / "raw" / "english"
    existing_wf = len(list(real_dir.glob("real_wf_*")))
    if existing_wf < 1000 and smoke_raw.exists():
        logger.info("Transferring existing WaveFake pairs from %s into %s...", smoke_raw, raw_root)
        for f in (smoke_raw / "real").glob("real_wf_*"):
            dest = real_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
        for f in (smoke_raw / "fake").glob("fake_wf_*"):
            dest = fake_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

        # Copy sidecar entries
        smoke_sidecar = smoke_raw / "pair_transcripts.csv"
        if smoke_sidecar.exists():
            rows = []
            with smoke_sidecar.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                rows = [line for line in reader if line]
            if rows:
                with sidecar_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    if sidecar_path.stat().st_size == 0:
                        writer.writerow(["filename", "transcript"])
                    for row in rows:
                        if len(row) >= 2 and row[0] != "filename":
                            writer.writerow(row)
        logger.info("Transferred %d WaveFake pairs successfully.", len(list(real_dir.glob("real_wf_*"))))

    # 2. Source 2: ASVspoof 2019 LA (VCTK multi-speaker bona fide + diverse TTS/VC spoofs)
    existing_asv = len(list(real_dir.glob("real_asv_*")))
    remaining_asv = max(0, TARGET_ASVSPOOF_PAIRS - existing_asv)
    if remaining_asv > 0:
        logger.info("Streaming %d ASVspoof 2019 LA pairs from Hugging Face...", remaining_asv)
        base_ds = load_dataset(
            "Bisher/ASVspoof_2019_LA",
            split="validation",
            streaming=True,
        ).cast_column("audio", Audio(decode=False))

        reals_stream = base_ds.take(remaining_asv)
        fakes_stream = base_ds.skip(2548).take(remaining_asv)

        sidecar_rows = []
        for i, (real_row, fake_row) in enumerate(zip(reals_stream, fakes_stream)):
            pair_idx = existing_asv + i
            real_name = f"real_asv_{pair_idx:06d}.flac"
            fake_name = f"fake_asv_{pair_idx:06d}.flac"
            pair_key = f"asv_pair_{pair_idx:06d}"

            real_path = real_dir / real_name
            fake_path = fake_dir / fake_name

            real_path.write_bytes(real_row["audio"]["bytes"])
            fake_path.write_bytes(fake_row["audio"]["bytes"])

            sidecar_rows.append((real_name, pair_key))
            sidecar_rows.append((fake_name, pair_key))

            if (i + 1) % 100 == 0 or i + 1 == remaining_asv:
                logger.info("  Streamed %d/%d ASVspoof pairs...", i + 1, remaining_asv)

        if sidecar_rows:
            with sidecar_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if sidecar_path.stat().st_size == 0:
                    writer.writerow(["filename", "transcript"])
                writer.writerows(sidecar_rows)

    total_real = len(list(real_dir.glob("*")))
    total_fake = len(list(fake_dir.glob("*")))
    logger.info("Multi-source raw corpus ready: %d real clips, %d fake clips (total: %d).", total_real, total_fake, total_real + total_fake)


def _apply_codec_augmentation() -> None:
    """Apply MP3 and AAC lossy compression augmentation to a balanced subset of training pairs."""
    raw_root = CONFIG.paths.raw_dir / "english"
    real_dir = raw_root / "real"
    fake_dir = raw_root / "fake"
    sidecar_path = raw_root / "pair_transcripts.csv"

    existing_mp3 = len(list(real_dir.glob("*_codec_mp3.*")))
    if existing_mp3 >= 400:
        logger.info("Codec augmentation already applied (%d MP3 clips present). Skipping.", existing_mp3)
        return

    logger.info("Applying Codec Data Augmentation (MP3 & AAC) to balanced training pairs...")
    ffmpeg = shutil.which("ffmpeg")

    sidecar_map = {}
    if sidecar_path.exists():
        with sidecar_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    sidecar_map[row[0]] = row[1]

    new_sidecar_rows = []

    # 1. Augment 500 WaveFake pairs with MP3 (via soundfile)
    wf_reals = sorted(real_dir.glob("real_wf_*.wav"))[:500]
    with tempfile.TemporaryDirectory() as td:
        mp3_tmp = Path(td) / "temp.mp3"
        for i, r_path in enumerate(wf_reals):
            stem = r_path.stem
            f_path = fake_dir / f"fake_{stem[5:]}.wav"
            if not f_path.exists():
                continue

            r_aug_name = f"{stem}_codec_mp3.wav"
            f_aug_name = f"{f_path.stem}_codec_mp3.wav"
            r_aug_path = real_dir / r_aug_name
            f_aug_path = fake_dir / f_aug_name

            # Compress real
            r_audio, sr = sf.read(r_path, dtype="float32")
            sf.write(str(mp3_tmp), r_audio, sr, format="MP3")
            comp_r, _ = sf.read(str(mp3_tmp), dtype="float32")
            sf.write(r_aug_path, comp_r, sr, subtype="PCM_16")

            # Compress fake
            f_audio, sr_f = sf.read(f_path, dtype="float32")
            sf.write(str(mp3_tmp), f_audio, sr_f, format="MP3")
            comp_f, _ = sf.read(str(mp3_tmp), dtype="float32")
            sf.write(f_aug_path, comp_f, sr_f, subtype="PCM_16")

            pair_key = sidecar_map.get(r_path.name, f"wf_{stem}")
            new_sidecar_rows.append((r_aug_name, pair_key))
            new_sidecar_rows.append((f_aug_name, pair_key))

            if (i + 1) % 100 == 0:
                logger.info("  Augmented %d/500 WaveFake pairs with MP3...", i + 1)

    # 2. Augment 500 ASVspoof pairs with AAC (via ffmpeg)
    if ffmpeg:
        asv_reals = sorted(real_dir.glob("real_asv_*.flac"))[:500]
        with tempfile.TemporaryDirectory() as td:
            m4a_tmp = Path(td) / "temp.m4a"
            wav_tmp = Path(td) / "temp.wav"
            for i, r_path in enumerate(asv_reals):
                stem = r_path.stem
                f_path = fake_dir / f"fake_{stem[5:]}.flac"
                if not f_path.exists():
                    continue

                r_aug_name = f"{stem}_codec_aac.wav"
                f_aug_name = f"{f_path.stem}_codec_aac.wav"
                r_aug_path = real_dir / r_aug_name
                f_aug_path = fake_dir / f_aug_name

                # Compress real via AAC
                subprocess.run(
                    [ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", str(r_path), "-c:a", "aac", "-b:a", "96k", str(m4a_tmp)],
                    check=True,
                )
                subprocess.run(
                    [ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", str(m4a_tmp), "-ac", "1", str(wav_tmp)],
                    check=True,
                )
                comp_r, sr = sf.read(str(wav_tmp), dtype="float32")
                sf.write(r_aug_path, comp_r, sr, subtype="PCM_16")

                # Compress fake via AAC
                subprocess.run(
                    [ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", str(f_path), "-c:a", "aac", "-b:a", "96k", str(m4a_tmp)],
                    check=True,
                )
                subprocess.run(
                    [ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", str(m4a_tmp), "-ac", "1", str(wav_tmp)],
                    check=True,
                )
                comp_f, sr_f = sf.read(str(wav_tmp), dtype="float32")
                sf.write(f_aug_path, comp_f, sr_f, subtype="PCM_16")

                pair_key = sidecar_map.get(r_path.name, f"asv_{stem}")
                new_sidecar_rows.append((r_aug_name, pair_key))
                new_sidecar_rows.append((f_aug_name, pair_key))

                if (i + 1) % 100 == 0:
                    logger.info("  Augmented %d/500 ASVspoof pairs with AAC...", i + 1)

    if new_sidecar_rows:
        with sidecar_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(new_sidecar_rows)

    total_real = len(list(real_dir.glob("*")))
    total_fake = len(list(fake_dir.glob("*")))
    logger.info("Augmented corpus ready: %d real, %d fake (%d total clips).", total_real, total_fake, total_real + total_fake)


def _generate_cross_source_validation_file(service: InferenceService) -> Path:
    """Run holdout validation across both clean and codec-compressed speech."""
    logger.info("Evaluating English model on held-out ASVspoof clips (lossless + codec-compressed)...")
    base_ds = load_dataset(
        "Bisher/ASVspoof_2019_LA",
        split="validation",
        streaming=True,
    ).cast_column("audio", Audio(decode=False))

    eval_reals = list(base_ds.skip(TARGET_ASVSPOOF_PAIRS).take(100))
    eval_fakes = list(base_ds.skip(2548 + TARGET_ASVSPOOF_PAIRS).take(100))

    predictions = []
    temp_dir = Path("outputs/english/temp_eval")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "eval_sample.wav"

    with tempfile.TemporaryDirectory() as td:
        mp3_tmp = Path(td) / "eval.mp3"
        raw_in = Path(td) / "raw.flac"

        for idx, r in enumerate(eval_reals):
            raw_bytes = r["audio"]["bytes"]
            if idx % 2 == 1:
                # Apply MP3 compression to half of the holdout clips
                raw_in.write_bytes(raw_bytes)
                audio_data, sr = sf.read(str(raw_in), dtype="float32")
                sf.write(str(mp3_tmp), audio_data, sr, format="MP3")
                comp_audio, sr = sf.read(str(mp3_tmp), dtype="float32")
                sf.write(str(temp_file), comp_audio, sr, subtype="PCM_16")
                tag = "asv_vctk_real_mp3"
            else:
                temp_file.write_bytes(raw_bytes)
                tag = "asv_vctk_real_clean"

            res = service.predict(temp_file)
            predictions.append({
                "filepath": f"asv_eval/real/{r['audio_file_name']}.flac",
                "true_label": "real",
                "fake_score": 1.0 - res.confidence if res.predicted_label == "real" else res.confidence,
                "source_tag": tag,
            })

        for idx, f in enumerate(eval_fakes):
            raw_bytes = f["audio"]["bytes"]
            if idx % 2 == 1:
                raw_in.write_bytes(raw_bytes)
                audio_data, sr = sf.read(str(raw_in), dtype="float32")
                sf.write(str(mp3_tmp), audio_data, sr, format="MP3")
                comp_audio, sr = sf.read(str(mp3_tmp), dtype="float32")
                sf.write(str(temp_file), comp_audio, sr, subtype="PCM_16")
                tag = f"asv_spoof_{f.get('system_id', 'synth')}_mp3"
            else:
                temp_file.write_bytes(raw_bytes)
                tag = f"asv_spoof_{f.get('system_id', 'synth')}_clean"

            res = service.predict(temp_file)
            predictions.append({
                "filepath": f"asv_eval/fake/{f['audio_file_name']}.flac",
                "true_label": "fake",
                "fake_score": res.confidence if res.predicted_label == "fake" else 1.0 - res.confidence,
                "source_tag": tag,
            })

    shutil.rmtree(temp_dir, ignore_errors=True)

    rel_checkpoint = str(service.checkpoint_path.resolve().relative_to(ROOT_DIR))
    payload = {
        "checkpoint": rel_checkpoint,
        "note": "English multi-source validation holdout calibration scores (clean + codec-compressed).",
        "runs": [
            {
                "sampling_seed": 42,
                "samples_per_source": 100,
                "predictions": predictions,
            }
        ],
    }

    out_path = ROOT_DIR / "outputs" / "english" / "cross_source_scores.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Wrote official English cross-source validation scores to %s (%d clips).", out_path, len(predictions))
    return out_path


def main() -> None:
    start_total = time.time()
    logger.info("=" * 60)
    logger.info("Starting Multi-Source English Pipeline (WaveFake + ASVspoof 2019 LA + Codec Augmentation)")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("Image size: %dx%d | Feature type: %s", CONFIG.features.image_size, CONFIG.features.image_size, CONFIG.features.type)
    logger.info("CUDA available: %s (%s)", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    logger.info("=" * 60)

    # 1. Ingestion & Codec Augmentation
    _ensure_raw_multisource_data()
    _apply_codec_augmentation()

    # 2. Preprocessing
    logger.info("Step 2: Building metadata & preprocessing audio...")
    manager = MetadataManager(CONFIG.paths.metadata_file)
    manager = build_initial_metadata(manager)
    PreprocessingPipeline(metadata_manager=manager).run(bootstrap_if_empty=False)

    # 3. Feature Extraction (224x224 mel-spectrograms)
    logger.info("Step 3: Extracting features (%dx%d)...", CONFIG.features.image_size, CONFIG.features.image_size)
    manager = MetadataManager(CONFIG.paths.metadata_file)
    FeatureExtractionPipeline(metadata_manager=manager).run()

    # 4. Grouped Split Assignment (Prevents pair leakage across splits)
    logger.info("Step 4: Assigning transcript-paired grouped splits...")
    manager = MetadataManager(CONFIG.paths.metadata_file)
    manager = assign_grouped_splits(manager, group_column="transcript", force_reassign=True)

    # 5. Training
    logger.info("Step 5: Training EfficientNet-B0 with mixed precision on GPU...")
    train_loader = build_dataloader("train", metadata_manager=manager)
    val_loader = build_dataloader("val", metadata_manager=manager)
    test_loader = build_dataloader("test", metadata_manager=manager)

    model = build_model()
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader)
    trainer.fit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_checkpoint = CONFIG.paths.checkpoint_dir / "best_model.pt"
    best_state = trainer.checkpoint_manager.load(best_checkpoint, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    # 6. Test Evaluation
    logger.info("Step 6: Evaluating test set metrics...")
    evaluator = Evaluator(model, test_loader, device)
    metrics = evaluator.evaluate()

    metrics_path = CONFIG.paths.output_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Test Metrics:\n%s", json.dumps(metrics, indent=2))

    # 7. Official Cross-Source Validation
    logger.info("Step 7: Generating official cross-source validation scores...")
    service = InferenceService(checkpoint_path=best_checkpoint, feature_image_size=CONFIG.features.image_size)
    validation_file = _generate_cross_source_validation_file(service)

    elapsed_total = time.time() - start_total
    logger.info("=" * 60)
    logger.info("English Pipeline Complete! Elapsed time: %.1f seconds.", elapsed_total)
    logger.info("Best Checkpoint:   %s", best_checkpoint)
    logger.info("Validation Scores: %s", validation_file)
    logger.info("Test Accuracy:     %.2f%%", metrics.get("accuracy", 0.0) * 100)
    logger.info("Test F1 Score:     %.4f", metrics.get("f1", 0.0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
