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

```powershell
# 1. Stream a bounded sample of real + fake Hindi clips into data/raw/
python -m preprocessing.hf_ingestion

# 2. Validate, transform (mono/16kHz/trim/normalize/pad-to-4s), save to data/processed/
python -m preprocessing.pipeline

# 3. Extract Mel/MFCC/LFCC features, resize, save to data/features/
python -m feature_extraction.pipeline

# 4. Assign stratified train/val/test splits
python -m dataset.split_assigner
```

Every command above is safe to interrupt and rerun. Adjust `ingestion.samples_per_class` in `configs/config.yaml` to scale the dataset up, then rerun step 1 onward — only new clips are fetched/processed.

## Configuration

All tunables (audio params, feature extraction, model architecture, training hyperparameters, evaluation metrics, deployment) live in `configs/config.yaml`. See that file for the full set of options.
