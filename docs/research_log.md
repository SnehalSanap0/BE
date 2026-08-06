# Research Log

Running, crisp record of design decisions, bugs found, and experiment results — kept for writing up the methodology/results sections later. Newest entries at the bottom. Each entry: what we did, why, what we found.

---

### 2026-07-20 — Multi-source ingestion design
Switched ingestion from one dataset per class to **multiple independent sources per class**:
- `real`: Kathbath (3000) + Common Voice Hindi male (750) + Common Voice Hindi female (750)
- `fake`: IndicSynth (3000) + SEA-Spoof, filtered to `language=hi, label=spoof` (1500)

**Why:** a single dataset per label confounds "real vs fake" with "which dataset," letting a detector learn dataset/recording fingerprints instead of genuine spoofing artifacts, rather than the intended task.

---

### 2026-07-23 — Data audit: real-class shortfall
Raw file counts on disk vs configured targets:

| source_tag | target | actual |
|---|---|---|
| real (Kathbath) | 3000 | 3000 |
| real_cv_m (CV male) | 750 | 750 |
| real_cv_f (CV female) | 750 | **560** |
| fake (IndicSynth) | 3000 | 3000 |
| fake_sea (SEA-Spoof) | 1500 | 1500 |

**Finding:** `real_cv_f` under-filled by 190 clips — the streamed HF split for that source simply doesn't contain 750 Hindi female clips; `ingest_source` has no warning for under-fill, it just stops when the stream is exhausted. Net effect: real=4310, fake=4500 (not perfectly balanced).

---

### 2026-07-23 — Pipeline staleness bug: metadata not rebuilt after adding sources
`metadata.csv` still held only 6000 records (the old single-source-per-class ingestion) days after the new multi-source ingestion (8810 raw clips) had already been streamed to disk — `preprocessing.pipeline` / `feature_extraction.pipeline` / `dataset.split_assigner` had not been rerun. **The first two training runs on 2026-07-23 (BiLSTM and baseline) were both trained on the stale 6000-record dataset, not the new one.** Fixed by rerunning steps 2–4 of the pipeline (8810 records after rebuild, 2810 newly processed/featured).

**Side note:** the two runs also overwrote each other's checkpoint files (`last_checkpoint.pt`/`best_model.pt` are architecture-agnostic filenames) since `training.resume=true` — a BiLSTM checkpoint and a baseline checkpoint can silently clobber one another if `model.use_bilstm` changes between runs without clearing `checkpoints/`.

---

### 2026-07-23 — First training run (post-rebuild), pooled random split
Config: EfficientNet-B0 baseline (no BiLSTM), full 8810-clip multi-source dataset, standard `dataset.split_assigner` (stratified by label only, split randomly across pooled sources).

**Result:** `accuracy=1.0000, f1=1.0000, roc_auc=1.0000, eer=0.0000` — by epoch 1, early-stopped at epoch 8/50.

**Read:** suspiciously perfect — real anti-spoofing benchmarks don't hit exact 0 EER. Hypothesis: multiple sources per class stops the model keying on *which dataset*, but not on *recording-condition/channel artifacts shared within a class* (e.g. natural mic recordings vs. vocoder output). Random pooled splitting also puts every source in train AND test, so even a channel-fingerprint shortcut looks perfect.

---

### 2026-07-23 — Confound smoke test: trailing-silence / RMS / silence-ratio by label
Script: `scripts/confound_check.py` (scratch, one-off). Computed per-clip trailing zero-padding ratio, RMS loudness, and frame-level silence ratio from the already-processed audio, grouped by label and source_tag.

**Finding:** no single feature is *perfectly* separable (ranges overlap), so it's not a trivial one-feature leak. But there's a strong systematic shift: real clips average ~3x more trailing silence/padding than fake (real 0.116 vs fake 0.040 trailing-zero ratio; 0.26 vs 0.14 silence ratio), consistent across sub-sources (Common Voice sources pad 26–34%, TTS sources pad 2–8%). Natural speech has more pauses/breath than TTS output — plausible partial shortcut on top of other channel-level cues a CNN can see that this script can't measure directly.

---

### 2026-07-23 — Cross-source holdout generalization test
Script: `scripts/cross_source_holdout_check.py`. Re-split by **source dataset**, not randomly:
- train/val: only `real` (Kathbath) + `fake` (IndicSynth) — model never sees the other two sources during training or validation
- test: only `real_cv_m`/`real_cv_f` (Common Voice) + `fake_sea` (SEA-Spoof) — recording chains and synthesis systems never seen in training

Reuses existing `MetadataManager`/`build_model`/`build_dataloader`/`Trainer`/`Evaluator`; writes to a separate metadata file (`metadata_cross_source_holdout.csv`) and checkpoint dir (`checkpoints/cross_source_holdout/`) so the main pipeline/checkpoints are untouched.

**Result:** `accuracy=0.9612, precision=0.9741, recall=0.9527, f1=0.9633, roc_auc=0.9939, eer=0.0374`. Confusion matrix: 71/1500 actual spoof clips misclassified as real (4.73% miss rate), 38/1310 actual real clips misclassified as fake (2.90% false-alarm rate).

**Interpretation:**
- EER 0.0000 → 0.0374 on completely unseen sources confirms the pooled-split "perfect" score was partly inflated by a channel/recording-condition shortcut, not solely genuine spoof-detection.
- But 96%+ accuracy / ~3.7% EER on entirely unseen recording chains is a legitimately solid cross-corpus result (in line with published cross-dataset anti-spoofing numbers) — so the model did learn some real, transferable signal on top of the shortcut, not nothing.
- Operationally, the miss rate (spoof passed as real) exceeds the false-alarm rate — the asymmetry to consider when picking a decision threshold for a security-relevant deployment, rather than using the EER-equalizing threshold (0.443) by default.
- **Methodological takeaway for the paper:** report cross-source (leave-one-source-out) generalization as the primary metric, not the pooled random-split metric — the latter is an optimistic upper bound, not a realistic estimate.

**Open question:** only one train/test source pairing has been tested so far. Rotating which sources are held out (e.g. train on Common Voice+SEA-Spoof, test on Kathbath+IndicSynth) would show whether ~3-4% EER is a stable estimate or specific to this particular source pairing.

---

### 2026-07-24 — Cross-source holdout, reversed pairing ("swapped" rotation)
Script generalized (`scripts/cross_source_holdout_check.py --rotation {original,swapped}`) to test the reverse direction of the prior experiment: train/val on Common Voice (`real_cv_m`/`real_cv_f`) + SEA-Spoof (`fake_sea`), test on Kathbath (`real`) + IndicSynth (`fake`) — the mirror image of the "original" rotation.

**Result:** `accuracy=0.5027, precision=0.9444, recall=0.0057, f1=0.0113, roc_auc=0.8761, eer=0.2160, eer_threshold=0.00227`. Confusion matrix: 2983/3000 actual IndicSynth spoof clips misclassified as real (99.4% miss rate) at the standard 0.5 threshold; only 1/3000 Kathbath real clips misclassified as fake.

**Interpretation:**
- At the default decision threshold, this direction collapses to near-chance (50.3% accuracy) — far worse than the "original" rotation's 96.1%. The model predicts almost everything as "real," missing nearly all spoofed clips.
- Not pure failure to learn, though: `roc_auc=0.876` shows real ranking signal survives; the near-zero `eer_threshold` reveals a **calibration collapse under domain shift** — the model's fake-probability output is compressed toward zero for this entire unseen domain (both classes), not just for the hard cases. Recalibrating to the domain-specific EER threshold recovers to 21.6% EER — still much worse than the 3.74% seen in the other direction, but a different (and less alarming) failure mode than "zero signal."
- **Generalization here is asymmetric, not a fixed property of the model.** Training on {Kathbath, IndicSynth} → testing on {Common Voice, SEA-Spoof} generalized well (3.74% EER). Training on {Common Voice, SEA-Spoof} (the larger, more synthesis-diverse fake source) → testing on {Kathbath, IndicSynth} generalized far worse. More training-source diversity did not translate to better transfer in this direction — source *composition*, not just diversity/size, drives the result.
- **Methodological takeaway for the paper:** (1) the earlier 3.74% EER must be reported as direction-specific, not "the model generalizes cross-corpus" in general — this result disproves that broader claim; (2) a fixed 0.5 decision threshold is unsafe across domains — a system tuned this way could silently pass ~99% of spoofed audio if deployed against a domain resembling Kathbath/IndicSynth without per-domain threshold recalibration; (3) worth reporting AUC alongside EER/accuracy specifically because AUC survived here while threshold-dependent metrics collapsed — a case where discriminative power and calibration diverge sharply.

**Open question:** whether this asymmetry is driven by IndicSynth being a single-TTS-system fake source (narrow spoof-artifact diversity) versus SEA-Spoof spanning many systems, by Kathbath vs. Common Voice's differing channel/noise characteristics, or by both — worth an ablation (e.g. swap only the fake source or only the real source between rotations) if pursued further.

---

### 2026-07-24 — Full 2x2 factorial: isolating which source drives the asymmetry
Added `real_swap_only` and `fake_swap_only` rotations to `scripts/cross_source_holdout_check.py`, completing a full 2x2 factorial over (real-source, fake-source) used for training — the project has exactly two of each (real: Kathbath (A) vs Common Voice (B); fake: IndicSynth (A) vs SEA-Spoof (B)), each rotation trains on one (real, fake) combination and tests on the complementary pair.

**Full result table** (EER / ROC-AUC on the held-out complementary pair):

| train real ↓ / fake → | IndicSynth (A) | SEA-Spoof (B) |
|---|---|---|
| **Kathbath (A)** | `original`: EER=0.0374, AUC=0.9939 | `fake_swap_only`: EER=0.4046, AUC=0.6273 |
| **Common Voice (B)** | `real_swap_only`: EER=0.2325, AUC=0.8165 | `swapped`: EER=0.2160, AUC=0.8761 |

`real_swap_only` (train real=Common Voice + fake=IndicSynth, test real=Kathbath + fake=SEA-Spoof): `accuracy=0.6711, precision=0.5388, recall=0.0927, f1=0.1581, roc_auc=0.8165, eer=0.2325`. `fake_swap_only` (train real=Kathbath + fake=SEA-Spoof, test real=Common Voice + fake=IndicSynth): `accuracy=0.4211, precision=0.7695, recall=0.2403, f1=0.3663, roc_auc=0.6273, eer=0.4046`.

**Interpretation — this is an interaction effect, not a main effect of either source alone:**
- Neither "the fake-source choice dominates" nor "the real-source choice dominates" holds in isolation. If it did, one row or one column of the table would be consistently better; instead the effect of swapping the fake source **depends on which real-source it's paired with**.
- Along the Kathbath row: swapping only the fake source (IndicSynth→SEA-Spoof) is catastrophic — EER 0.0374→0.4046, and AUC collapses from near-perfect (0.9939) to barely-above-chance (0.6273). This is not a threshold/calibration issue like the other three cells — the model's actual score ranking breaks down.
- Along the Common Voice row, the identical fake-source swap barely moves the result (EER 0.2325 vs 0.2160, AUC 0.8165 vs 0.8761) — essentially flat.
- **(Kathbath, IndicSynth) is a uniquely well-generalizing training pairing; (Kathbath, SEA-Spoof) is uniquely the worst** — same real source, opposite outcomes, purely a function of which fake corpus it's trained alongside.
- The other three cells (`swapped`, `real_swap_only`, and to a lesser extent `fake_swap_only`) share the same near-zero `eer_threshold` signature seen in the earlier "swapped" entry — a calibration collapse under domain shift, where AUC partly survives (0.82–0.88) but the fixed decision threshold does not. `fake_swap_only` is qualitatively different: its AUC (0.6273) shows the *ranking itself*, not just the threshold, has broken down — the one training combination that fails to learn a transferable representation at all.

**Methodological takeaway for the paper:** don't report cross-source generalization as a single number or even as "asymmetric between two directions" — it is a genuine interaction between which real corpus and which fake corpus the model is trained on, and the single best-generalizing setup in this whole study (`original`, EER 3.74%) is adjacent in the table to the single worst (`fake_swap_only`, EER 40.46%), differing by only one factor. This argues for reporting the full factorial (or at least flagging that source-pairing interactions exist) rather than a single train/test split, cross-source or otherwise, as evidence of generalization.

**Open question, still unresolved:** *why* Kathbath+IndicSynth specifically produces a well-calibrated, transferable decision boundary while Kathbath+SEA-Spoof produces the least transferable one is not yet explained — candidates worth checking: SEA-Spoof's much larger size/class-imbalance effects during training, its multilingual composition (it's filtered to `hi` for ingestion, but the model may still be seeing residual variation the filter doesn't fully control for), or some interaction between Kathbath's noisier/broadcast channel and SEA-Spoof's spoof-artifact distribution specifically. Not yet tested.

---

### 2026-07-24 — Per-source confound audit (`scripts/confound_check.py`, rewritten)

The original one-off confound script (never committed) found real clips carry ~3x more trailing silence/padding than fake clips on average. Rewrote it as `scripts/confound_check.py` to directly compare sources *within* each label — specifically IndicSynth (`fake`) vs SEA-Spoof (`fake_sea`) — since the factorial above showed the fake-source choice interacts with the real-source choice rather than acting as a fixed effect. Sampled 300 clips/source (all 5 sources), computing trailing-zero-padding ratio, frame-level silence ratio (`librosa.effects.split`, top_db=30), and RMS loudness from the already-processed audio; duration/sample_rate/clip-count pulled directly from `metadata.csv`.

**Result** (mean ± std; full table in `outputs/confound_analysis/source_confound_summary.csv`):

| source_tag | trailing_zero_mean | trailing_zero_std | silence_ratio_mean | silence_ratio_std | rms_mean |
|---|---|---|---|---|---|
| fake (IndicSynth) | 0.0144 | 0.0403 | 0.0689 | 0.0636 | 0.1518 |
| fake_sea (SEA-Spoof) | 0.0779 | 0.1556 | 0.1120 | 0.1495 | 0.1622 |
| real (Kathbath) | 0.0365 | 0.0832 | 0.1270 | 0.1003 | 0.1115 |
| real_cv_f (Common Voice) | 0.2810 | 0.2582 | 0.3310 | 0.2452 | 0.1308 |
| real_cv_m (Common Voice) | 0.3591 | 0.2525 | 0.3965 | 0.2252 | 0.1175 |

**Interpretation:**
- Confirms and sharpens the earlier scratch-script finding: Common Voice sources pad 28–36% on average vs. Kathbath's 3.65% and IndicSynth's 1.44% — the real-side channel confound is real and large.
- **New finding, directly relevant to the open question:** SEA-Spoof carries **~5.4x more trailing silence** than IndicSynth (0.0779 vs 0.0144) and **~1.6x more frame-level silence** (0.1120 vs 0.0689) on average — a distinguishing channel-level signature on the fake side specifically, not just the previously-known real-side one.
- SEA-Spoof's std on both silence metrics is **3–4x higher than IndicSynth's** (trailing-zero std 0.1556 vs 0.0403; silence-ratio std 0.1495 vs 0.0636) — i.e., SEA-Spoof clips are far more heterogeneous among themselves than IndicSynth clips are. This is consistent with the research log's standing hypothesis that SEA-Spoof spans many synthesis systems while IndicSynth is a single TTS system: a narrow, consistent fake-source (IndicSynth) pairs well with Kathbath (`original`, EER 3.74%), while a wide, heterogeneous one (SEA-Spoof) paired with the same real source (`fake_swap_only`) is the worst-generalizing cell in the factorial (EER 40.46%).
- **Limitation found, not yet fixed:** `metadata.csv`'s `speaker` column reads back as exactly 1 distinct value for every source — it was never populated per-clip during ingestion (defaults to `"unknown"`), so it can't currently be used as an independent diversity proxy. If speaker-level diversity turns out to matter, `preprocessing/metadata_builder.py`'s ingestion path needs to actually record per-clip speaker IDs where the upstream dataset provides them (Kathbath and Common Voice both do).
- **Methodological note for the paper:** report this table alongside the factorial as evidence for the *mechanism* behind the interaction effect, not just its existence.

---

### 2026-07-24 — Embedding-space diagnostic (`scripts/embedding_analysis.py`): source vs. label separability

To test the shortcut-learning hypothesis directly rather than only via proxy signal statistics, extracted each rotation's trained backbone's pooled 1280-d embedding for a balanced sample (400 clips/source, all 5 sources, 2000 total) drawn from the *full* dataset — not just that rotation's own held-out split, so all five checkpoints are compared on the identical clip sample. Computed silhouette score of the embedding space against `source_tag` labels and against `real`/`fake` labels separately, plus a `shortcut_ratio = silhouette(source) / silhouette(label)` (higher = embedding space structured more by recording source than by the intended task). t-SNE plots colored both ways saved to `outputs/embedding_analysis/<rotation>_tsne.png`.

**Result:**

| rotation | silhouette(source) | silhouette(label) | shortcut_ratio | cross-source EER (from factorial) |
|---|---|---|---|---|
| main (pooled random split) | 0.1469 | 0.4608 | 0.319 | n/a (EER=0.0000, inflated — see 2026-07-23 entry) |
| original | 0.1436 | 0.2810 | 0.511 | 0.0374 |
| real_swap_only | 0.1492 | 0.2746 | 0.543 | 0.2325 |
| swapped | 0.1417 | 0.2224 | 0.637 | 0.2160 |
| fake_swap_only | 0.1691 | 0.1769 | **0.956** | **0.4046** |

**Interpretation:**
- The `shortcut_ratio` ordering tracks the cross-source EER ordering closely (`original` best on both; `fake_swap_only` worst on both by a wide margin), giving direct embedding-space evidence for what the confound audit above only suggested indirectly: the worse a rotation generalizes to unseen sources, the more its embedding space encodes *which source a clip came from* relative to *whether it's real or fake*.
- `fake_swap_only`'s embeddings barely separate label better than source at all (0.1769 vs 0.1691, ratio 0.956) — i.e. for this training pairing (Kathbath + SEA-Spoof), the model's internal representation has almost collapsed the real/fake distinction into the source-identity distinction. This is a direct mechanistic explanation for that cell's near-chance-ranking AUC (0.6273, see 2026-07-24 factorial entry above), not just a correlation.
- `main` (pooled random split, the "suspiciously perfect" EER=0.0000 model from 2026-07-23) has the *lowest* shortcut_ratio of all five and by far the highest label-silhouette (0.4608) — consistent with that model having no incentive to be source-invariant (it saw every source in training) and instead learning a representation that separates label very cleanly *for sources it has seen*, which is exactly why its pooled metrics were misleadingly perfect but say nothing about unseen-source generalization.
- **Open question from 2026-07-24 (factorial entry) substantially answered:** the mechanism is a genuine interaction between recording-channel silence/padding statistics (confound audit above) and the resulting embedding-space geometry (this entry) — SEA-Spoof's higher internal heterogeneity in silence statistics appears to prevent the model from cleanly separating it from Kathbath along the real/fake axis, collapsing much of that axis into a source-identity axis instead. Still not fully isolated to a single root cause (would need a controlled intervention, e.g. loudness/silence-normalizing all sources before training and re-running the factorial, to move from correlational to causal evidence) — noted as future work.
- **For the paper:** this embedding diagnostic plus the confound audit together upgrade the factorial's finding from "generalization varies by source pairing" (observed) to "generalization varies with how much the embedding space conflates source identity with label, which itself tracks measurable channel-level heterogeneity in the training sources" (partially explained) — a stronger claim, with the causal intervention flagged as the natural next experiment rather than a gap.

---

### 2026-08-06 - Paired Hindi SEA-Spoof smoke pipeline

Added a separate smoke-test path for paired Hindi SEA-Spoof bonafide/spoof examples:
- Config: `configs/config.smoke_seaspoof.yaml`
- Runner: `scripts/run_paired_seaspoof_smoke.py`
- Artifact roots: `data/smoke_seaspoof_paired/`, `checkpoints/smoke_seaspoof_paired/`, `logs/smoke_seaspoof_paired/`, `outputs/smoke_seaspoof_paired/`

**Why:** the immediate test was whether the model can train on transcript-matched Hindi real/spoof pairs so content/transcript is less likely to define the label. SEA-Spoof is the better immediate dataset than MHDDV for this repo because `preprocessing.hf_ingestion.ingest_paired_sources` already pairs SEA-Spoof bonafide/spoof rows by transcript (`text`) and writes matched `real_sea_000000.*` / `fake_sea_000000.*` files.

**Implementation notes:**
- `configs.config` now supports `DEEPFAKE_CONFIG_PATH`, so the smoke run can use an isolated config without editing the main `configs/config.yaml`.
- SEA-Spoof `train` shards were too large and ordered poorly for a quick test; smoke config uses `split: validation`, which contained enough Hindi bonafide/spoof transcript matches in `validation-00000.parquet`.
- Added optional shard scan controls for future quick tests: `SEA_SPOOF_SHARD_REVERSE=1` and `SEA_SPOOF_SHARD_START_INDEX=N`.
- Fixed a PyArrow crash in the second audio-fetch pass (`ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs`) by reading the nested `audio` struct row-group-wise instead of via `iter_batches`.

**Smoke progression:**
- 12 pairs, 2 epochs: pipeline completed, but the model collapsed to predicting one class; test set was only 4 clips, so metrics were not meaningful.
- 100 pairs, pretrained EfficientNet, more epochs: model stopped collapsing and reached `accuracy=0.8333`, `f1=0.8387`, `roc_auc=0.9156`, `eer=0.2333` on a 30-clip test set.
- 500 pairs, pretrained EfficientNet-B0, batch size 32, 15 epochs: latest saved smoke metrics in `outputs/smoke_seaspoof_paired/smoke_test_metrics.json`.
- Final 1000-pair checkpoint model: `accuracy=0.9866`, `precision=1.0000`, `recall=0.9732`, `f1=0.9864`, `roc_auc=1.0000`, `eer=0.0000`, `eer_threshold=0.0735`. Confusion matrix: 149/149 real correctly classified, 145/149 fake correctly classified, 4 fake clips missed as real.
- Attempting to scale validation smoke ingestion to 1500 pairs was inefficient: after 1000 existing pairs, `validation-00000.parquet` found only 261 additional matched keys and `validation-00001.parquet` found 0, while each shard download is multi-GB. Conclusion: SEA-Spoof validation is enough for smoke/Gradio testing, but chasing larger paired counts through validation shards is not a good training-data strategy.

| metric | value |
|---|---:|
| accuracy | 0.9533 |
| precision | 0.9250 |
| recall | 0.9867 |
| f1 | 0.9548 |
| roc_auc | 0.9964 |
| eer | 0.0400 |
| eer_threshold | 0.5403 |

Confusion matrix:

| actual \ predicted | real | fake |
|---|---:|---:|
| real | 69 | 6 |
| fake | 1 | 74 |

**Interpretation:** this is a successful paired same-source pipeline check, not cross-source evidence. The model can learn a strong decision boundary on transcript-paired SEA-Spoof validation examples, and the end-to-end GPU path is working. Because train/val/test all come from SEA-Spoof validation-derived paired examples, these metrics may still include SEA-Spoof-specific artifacts and should not replace the cross-source factorial as the generalization result.

**Code note:** the `preprocessing/hf_ingestion.py` row-group audio read is required for SEA-Spoof paired ingestion because PyArrow failed on nested `audio` batch conversion. The shard-order environment controls are not required for correctness; they are a convenience for avoiding badly ordered shards during quick smoke runs.

---

### 2026-08-06 - Paired spectrogram viewer for real/spoof inspection

Added `scripts/plot_paired_spectrograms.py` to render extracted `.npy` feature maps for matched SEA-Spoof pairs.

**Why:** if `real_sea_000042.wav` and `fake_sea_000042.wav` share the same transcript, comparing their extracted mel-spectrogram arrays helps inspect what EfficientNet is likely seeing. The third panel (`spoof - real`) highlights visible differences such as high-frequency texture, silence/padding, energy distribution, or preprocessing artifacts.

**Usage:**

```powershell
python -m scripts.plot_paired_spectrograms --limit 10
python -m scripts.plot_paired_spectrograms --pair-id 000042
```

Outputs are written to `outputs/smoke_seaspoof_paired/paired_spectrograms/`.

**Methodological use:** these plots are qualitative diagnostics. If visible differences are dominated by padding/silence or other channel-level cues rather than speech-region spoof artifacts, that suggests the next controlled experiment should normalize or crop speech-active regions before retraining.
