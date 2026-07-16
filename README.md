# Deepfake Audio Detector

Binary classification of audio clips as **real human speech** or **AI-generated / synthetic speech**, built on EfficientNet-B0 (+ optional BiLSTM temporal head) over mel-spectrogram / MFCC / LFCC features. Initial focus: Hindi, using [IndicSUPERB/Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) (real) and [IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth) (synthetic).

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

`ai4bharat/Kathbath` is a gated dataset — accept its terms at [huggingface.co/datasets/ai4bharat/Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) while logged in, then generate a read-scoped token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Running the pipeline

Every command below is safe to interrupt (Ctrl+C) and rerun — each stage reads `data/metadata/metadata.csv` to figure out what's already done and only processes what's left.

```powershell
# 1. Stream a bounded sample of real (Kathbath) + fake (IndicSynth) Hindi
#    clips into data/raw/hindi/{real,fake}/. Size is set by
#    ingestion.samples_per_class in configs/config.yaml (default: 3000/class).
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

To scale the dataset up, bump `ingestion.samples_per_class` in `configs/config.yaml` and rerun from step 1 — only the newly-needed clips are fetched and processed, nothing already done is repeated.

Watch TensorBoard during/after training with:

```powershell
tensorboard --logdir logs/tensorboard
```

## Configuration

All tunables (audio params, feature extraction, model architecture, training hyperparameters, evaluation metrics, explainability, deployment) live in `configs/config.yaml`, loaded once into the typed `CONFIG` singleton in `configs/config.py`. See that file for the full set of options.
