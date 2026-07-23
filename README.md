# Deepfake Audio Detector

Binary classification of audio clips as **real human speech** or **AI-generated / synthetic speech**, built on EfficientNet-B0 (+ optional BiLSTM temporal head) over mel-spectrogram / MFCC / LFCC features. Initial focus: Hindi, drawing real speech from [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) and Common Voice Hindi, and synthetic speech from [IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth) and [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) — multiple sources per class deliberately, so "real vs fake" isn't confounded with "which dataset."

## Design principles

- **`metadata.csv` is the single source of truth.** Every stage (preprocessing, feature extraction, dataset loading) reads and writes exactly this file. No stage scans a data folder to decide what to do, except the one-time bootstrap in `preprocessing/metadata_builder.py`.
- **Every batch stage is resumable.** Kill any pipeline mid-run; rerunning the same command picks up exactly where it left off.
- **Bounded streaming ingestion.** The upstream datasets are ~1TB combined; `preprocessing/hf_ingestion.py` streams a configurable, capped sample per class directly from HuggingFace instead of downloading everything.
- **No hardcoded paths or magic numbers.** Everything tunable lives in `configs/config.yaml`, loaded once into a typed `CONFIG` singleton (`configs/config.py`).

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file at the project root with:

```
HF_TOKEN=hf_...
```

`ai4bharat/Kathbath` and `Jack-ppkdczgx/SEA-Spoof` are gated datasets — accept/request access at [huggingface.co/datasets/ai4bharat/Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) and [huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) while logged in, then generate a read-scoped token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Running the pipeline

Every command below is safe to interrupt (Ctrl+C) and rerun — each stage reads `data/metadata/metadata.csv` to figure out what's already done and only processes what's left.

```powershell
# 1. Stream each configured source (multiple per label - see
#    ingestion.real / ingestion.fake in configs/config.yaml) into
#    data/raw/hindi/{real,fake}/, up to each source's own target_count.
python -m preprocessing.hf_ingestion

# 2. Validate, transform (mono / 16kHz / trim silence / peak normalize /
#    pad-or-truncate to 4s), save to data/processed/. Updates metadata.csv.
python -m preprocessing.pipeline

# 3. Extract Mel-spectrogram / MFCC / LFCC features (feature type set by
#    features.type in config.yaml), resize to 224x224, save as .npy to
#    data/features/. Updates metadata.csv.
python -m feature_extraction.pipeline

# 4. Assign stratified train/val/test splits (ratios set by dataset.* in
#    config.yaml), written into metadata.csv's "split" column.
python -m dataset.split_assigner

# 5. Train the configured model (baseline EfficientNet-B0, or
#    EfficientNet-B0 + BiLSTM if model.use_bilstm is true). Resumable —
#    reruns pick up from the last checkpoint. Writes
#    checkpoints/last_checkpoint.pt and checkpoints/best_model.pt, plus
#    TensorBoard logs under logs/tensorboard/.
python -m training.train

# 6. Evaluate the best checkpoint on the test split: accuracy, precision,
#    recall, F1, ROC-AUC, EER, confusion matrix. Writes
#    outputs/test_metrics.json and outputs/confusion_matrix.png.
python -m evaluation.evaluate

# 7. Explain a single clip's prediction with a Grad-CAM heatmap overlaid
#    on its spectrogram. Writes outputs/explanations/<clip>_gradcam.png.
python -m explainability.explain --audio path\to\clip.wav

# 8. Serve predictions over HTTP. GET /health; POST /predict with a
#    multipart "audio" field returns {predicted_label, confidence,
#    gradcam_image_base64}.
python -m deployment.app
```

To scale the dataset up, bump any source's `target_count` (or add a new source) under `ingestion.real` / `ingestion.fake` in `configs/config.yaml` and rerun from step 1 — only the newly-needed clips are fetched and processed, nothing already done is repeated.

Watch TensorBoard during/after training with:

```powershell
tensorboard --logdir logs/tensorboard
```

## Cross-source generalization check

`dataset.split_assigner` splits train/val/test randomly across the *pooled* clips, so every source dataset ends up in all three splits. That lets a model hit misleadingly perfect metrics by learning "which recording chain is this" instead of genuine spoofing artifacts. `scripts/cross_source_holdout_check.py` re-splits by *source* instead, so the model trains on some sources and is evaluated only on sources it has never seen:

The project has exactly two source datasets per class (real: Kathbath vs Common Voice; fake: IndicSynth vs SEA-Spoof), so four rotations form a complete 2x2 factorial over which real-source and which fake-source the model trains on — always tested on the complementary pair. This isolates whether a generalization gap traces back to the real side, the fake side, or both:

```powershell
# Train real=Kathbath, fake=IndicSynth; test real=Common Voice, fake=SEA-Spoof — sources unseen in training.
python -m scripts.cross_source_holdout_check --rotation original --epochs 15

# The full reverse pairing: train real=Common Voice, fake=SEA-Spoof; test real=Kathbath, fake=IndicSynth.
python -m scripts.cross_source_holdout_check --rotation swapped --epochs 15

# Swap only the real source: train real=Common Voice, fake=IndicSynth; test real=Kathbath, fake=SEA-Spoof.
python -m scripts.cross_source_holdout_check --rotation real_swap_only --epochs 15

# Swap only the fake source: train real=Kathbath, fake=SEA-Spoof; test real=Common Voice, fake=IndicSynth.
python -m scripts.cross_source_holdout_check --rotation fake_swap_only --epochs 15
```

Each rotation reuses the already-processed/-featured clips (no preprocessing or feature extraction is repeated) and writes to its own files, so no rotation and the main pipeline ever overwrite each other:
- `data/metadata/metadata_cross_source_holdout_<rotation>.csv` — a re-split copy of `metadata.csv`
- `checkpoints/cross_source_holdout_<rotation>/` — that rotation's model checkpoints
- `outputs/cross_source_holdout_<rotation>_metrics.json` — that rotation's held-out test metrics

Pass `--rebuild-split` to recompute a rotation's split assignment (e.g. after ingesting more clips) instead of reusing the existing metadata copy. If test EER stays close to the pooled-split result, the model is learning genuine spoof-vs-real cues; if it jumps substantially, the pooled near-zero EER was a channel/source shortcut. See `docs/research_log.md` for results and interpretation as they come in.

## Configuration

All tunables (audio params, feature extraction, model architecture, training hyperparameters, evaluation metrics, explainability, deployment) live in `configs/config.yaml`, loaded once into the typed `CONFIG` singleton in `configs/config.py`. See that file for the full set of options.
