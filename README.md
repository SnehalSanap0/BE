# Deepfake Audio Detection

Binary classification of audio clips as **real human speech** or **AI-generated / synthetic speech**, built on EfficientNet-B0 (+ optional BiLSTM temporal head) over mel-spectrogram / MFCC / LFCC features. Initial focus: Hindi, drawing real speech from [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) and Common Voice Hindi, and synthetic speech from [IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth) and [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) — multiple sources per class deliberately, so "real vs fake" isn't confounded with "which dataset."

New to a term used in this README or in the code (EER, mel-spectrogram, shortcut learning, Grad-CAM, ...)? See **[docs/CONCEPTS_INDEX.md](docs/CONCEPTS_INDEX.md)** — written for understanding, not dictionary definitions.

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

# 8b. Alternative: interactive Gradio demo (upload/record audio, or pick a
#     bundled real/fake example clip). Shows the prediction, the Grad-CAM
#     overlay with a colorbar legend, and a caption naming the specific
#     time/frequency region that most influenced the prediction. Wraps the
#     same InferenceService as step 8, so predictions are identical.
python -m deployment.gradio_app
```

To scale the dataset up, bump any source's `target_count` (or add a new source) under `ingestion.real` / `ingestion.fake` in `configs/config.yaml` and rerun from step 1 — only the newly-needed clips are fetched and processed, nothing already done is repeated.

Watch TensorBoard during/after training with:

```powershell
tensorboard --logdir logs/tensorboard
```

## Fast paired SEA-Spoof smoke run

Paired runs now assign splits by transcript group, so all real/spoof clips
sharing a transcript remain in one partition. Independently audit a manifest
with `python -m scripts.audit_experiment_protocol` while its config is active.

For the publication-scale 3,000-pair run from SEA-Spoof `train`:

```powershell
$env:DEEPFAKE_CONFIG_PATH="configs/config.paper_seaspoof.yaml"
python -m scripts.run_paired_seaspoof_experiment --seed 42
```

Repeat with independently trained seeds.

For the E1 control condition — ordinary, non-paired, count-matched SEA-Spoof
training (same sources/counts/model/budget as the E2 run above, but real/fake
clips are sampled independently instead of matched by transcript):

```powershell
$env:DEEPFAKE_CONFIG_PATH="configs/config.paper_seaspoof_e1.yaml"
python -m scripts.run_nonpaired_seaspoof_experiment --seed 42
```

For reproducible cross-source
sampling with saved clip-level scores and manifests:

```powershell
$env:DEEPFAKE_CONFIG_PATH="configs/config.smoke_seaspoof.yaml"
python -m scripts.evaluate_cross_source_protocol --samples-per-source 100 --seeds 42 1337 2026
```

For a quick end-to-end GPU/pipeline check on transcript-matched Hindi real/spoof pairs, use the isolated smoke config. This writes only to `data/smoke_seaspoof_paired/`, `checkpoints/smoke_seaspoof_paired/`, `logs/smoke_seaspoof_paired/`, and `outputs/smoke_seaspoof_paired/`.

```powershell
# Use the environment that has torch installed.
.\venv312\Scripts\Activate.ps1

# Optional sanity check: should print True and your NVIDIA GPU name.
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Run paired SEA-Spoof ingestion, preprocessing, feature extraction, split,
# training, and evaluation using configs/config.smoke_seaspoof.yaml.
$env:DEEPFAKE_CONFIG_PATH="configs/config.smoke_seaspoof.yaml"
python -m scripts.run_paired_seaspoof_smoke
```

`configs/config.smoke_seaspoof.yaml` currently uses SEA-Spoof `validation`, paired by transcript, with `real_sea`/`fake_sea` targets. If the dataset shard order causes slow scans, the paired ingester supports two optional controls:

```powershell
# Scan validation shards from the end first.
$env:SEA_SPOOF_SHARD_REVERSE="1"
python -m scripts.run_paired_seaspoof_smoke

# Or skip the first N shards.
Remove-Item Env:\SEA_SPOOF_SHARD_REVERSE -ErrorAction SilentlyContinue
$env:SEA_SPOOF_SHARD_START_INDEX="3"
python -m scripts.run_paired_seaspoof_smoke
```

**Do not set `DEEPFAKE_CONFIG_PATH` in `.env`** — `.env` is loaded on every run (not just smoke runs), so a value left there silently overrides `config.yaml` for everything, including the main pipeline. Set it as a session environment variable (`$env:DEEPFAKE_CONFIG_PATH=...`, as above) each time you want the smoke config, not as a persisted `.env` entry.

To inspect what EfficientNet sees for matching real/spoof transcripts, render side-by-side `.npy` feature comparisons:

```powershell
# Saves real | spoof | spoof-real PNGs under outputs/smoke_seaspoof_paired/paired_spectrograms/.
python -m scripts.plot_paired_spectrograms --limit 10

# Render one specific pair by filename suffix, e.g. real_sea_000042.wav / fake_sea_000042.wav.
python -m scripts.plot_paired_spectrograms --pair-id 000042
```

### Verifying real/fake pairing

Paired ingestion (`ingest_paired_sources`) matches real/fake clips by transcript, and the filename index (`real_sea_000042` / `fake_sea_000042`) reflects that pairing — but the filename convention doesn't survive preprocessing or feature extraction (both stages rename files to a UUID `audio_id`). To make pairing checkable at every stage, not just in `data/raw/`, ingestion also records each pair's transcript into `metadata.csv`'s `transcript` column (via a `pair_transcripts.csv` sidecar the metadata bootstrap reads once). This only happens for clips ingested after this feature was added — older rows show an empty transcript rather than a false mismatch.

```powershell
# Rebuild metadata.csv (backfills the transcript column from pair_transcripts.csv for any newly-discovered files).
python -m preprocessing.metadata_builder

# Check that every real_sea/fake_sea pair actually shares the same transcript.
python -m scripts.verify_seaspoof_pairing
```

### Sanity-checking the smoke model against the four earlier sources

Quick, no-retraining check of whether the SEA-Spoof smoke checkpoint generalizes at all to the project's four earlier, never-trained-on sources (Kathbath, Common Voice Hindi male, Common Voice Hindi female, IndicSynth). It loads `checkpoints/smoke_seaspoof_paired/best_model.pt`, draws a random sample of raw clips per source straight from `data/raw/hindi/{real,fake}`, and reports plain per-source and overall accuracy. This is a smoke test, not a paper-grade result (no fixed protocol, single run) — treat a promising number as a reason to run the proper `cross_source_holdout_check.py` protocol below, not as a result to cite.

```powershell
# Must use the smoke config so feature extraction (image_size, n_mels, etc.)
# matches what the checkpoint was trained on.
$env:DEEPFAKE_CONFIG_PATH="configs/config.smoke_seaspoof.yaml"
python -m scripts.eval_smoke_model_on_legacy_sources --samples-per-source 15
```

Writes `outputs/smoke_seaspoof_paired/legacy_source_eval.json` (per-clip predictions, per-source accuracy, confusion counts). Pass `--seed` to change which random clips are sampled, or `--checkpoint` to point at a different checkpoint file.

## English WaveFake smoke pipeline

A first extension of the pipeline beyond Hindi, using the [WaveFake](https://arxiv.org/abs/2111.02813) benchmark (Frank & Schonherr, 2021) — real English speech from LJSpeech plus fakes from 7 distinct neural vocoders (WF1-WF7) — via the community HF parquet mirror [`ajaykarthick/wavefake-audio`](https://huggingface.co/datasets/ajaykarthick/wavefake-audio) (not the official WaveFake distribution, which is a Zenodo archive; this mirror was used because it drops into the existing streaming-ingestion pipeline with no new code).

Ingestion is transcript-paired, like SEA-Spoof: `preprocessing.hf_ingestion.ingest_wavefake_paired` matches each real clip with exactly one fake vocoder variant sharing its underlying LJSpeech `audio_id` (deterministically rotated across WF1-WF7 so every vocoder is represented), writes both into the same `pair_transcripts.csv` sidecar SEA-Spoof pairing uses, and `scripts/run_english_smoke.py` assigns splits via `assign_grouped_splits(group_column="transcript")` — so a pair's real+fake clips can never land in different train/val/test partitions. Still a same-corpus smoke run, not a cross-source result (no second, unseen English corpus yet) — see `docs/research_log.md` for the full design rationale and results.

```powershell
$env:DEEPFAKE_CONFIG_PATH="configs/config.english_smoke.yaml"
python -m scripts.run_english_smoke
```

Writes to dedicated `data/english_smoke/`, `checkpoints/english_smoke/`, `logs/english_smoke/`, and `outputs/english_smoke/` roots, same isolation pattern as the SEA-Spoof smoke config. Scale up by raising `target_count` per source in `configs/config.english_smoke.yaml`.

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

## Diagnosing *why* cross-source generalization varies

Two scripts investigate the mechanism behind the factorial's source-pairing interaction (see `docs/research_log.md`), rather than just its existence:

```powershell
# Per-source signal-statistic audit: trailing-silence ratio, frame-level
# silence ratio, RMS loudness, sampled per source_tag from processed audio,
# plus clip-count/duration/speaker stats pulled directly from metadata.csv.
# Writes outputs/confound_analysis/source_confound_summary.{csv,json}.
python -m scripts.confound_check --sample-size 300

# Embedding-space diagnostic: extracts each rotation checkpoint's 1280-d
# pooled backbone embedding for a balanced cross-source sample, t-SNE-plots
# it colored by source and by label, and reports silhouette(source) vs
# silhouette(label) as a "shortcut_ratio" -- how much the model's learned
# representation encodes recording source relative to the intended
# real/fake task. Writes outputs/embedding_analysis/<rotation>_tsne.png
# and <rotation>_silhouette.json.
python -m scripts.embedding_analysis --rotation fake_swap_only
python -m scripts.embedding_analysis --all   # all 4 rotations + the pooled-split model
```

## Configuration

All tunables (audio params, feature extraction, model architecture, training hyperparameters, evaluation metrics, explainability, deployment) live in `configs/config.yaml`, loaded once into the typed `CONFIG` singleton in `configs/config.py`. See that file for the full set of options.
