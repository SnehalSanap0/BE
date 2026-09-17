# Research Log

Running, crisp record of design decisions, bugs found, and experiment results — kept for writing up the methodology/results sections later. Newest entries at the bottom. Each entry: what we did, why, what we found.

---

## Accuracy vs. configuration (paired SEA-Spoof smoke tracker)

> **Protocol correction (2026-08-09):** entries below that predate the
> leakage-safe rerun used label-stratified row splitting, which could place
> clips sharing a transcript in different partitions. They are development
> results, not citable estimates. The surviving 500-pair manifest contained
> 164 transcript groups spanning splits.

Quick-reference table for how accuracy/EER move as `target_count` (paired clips) and
other `configs/config.smoke_seaspoof.yaml` knobs change, pulled out of the narrative
entries below so trends are visible without re-reading prose each time. **Append new
smoke-run rows here** (bottom of table) instead of only writing a prose entry —
values not printed by a given run are left as "—", not guessed.

| Date | Pairs (`target_count`) | Split | Shard order | Epochs | Batch size | Test clips | Accuracy | Precision | Recall | F1 | ROC-AUC | EER | Notes |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-06 | 12 | validation | default | 2 | — | 4 | — | — | — | — | — | — | Model collapsed to predicting one class; 4-clip test set too small for metrics to mean anything. |
| 2026-08-06 | 100 | validation | default | — (more than smoke test above) | — | 30 | 0.8333 | — | — | 0.8387 | 0.9156 | 0.2333 | Stopped collapsing once epochs/pretraining increased. |
| 2026-08-06 | 500 | validation | default | 15 | 32 | — | — | — | — | — | — | — | Numbers only in `outputs/smoke_seaspoof_paired/smoke_test_metrics.json` at the time, not copied into the log. |
| 2026-08-06 | 1000 | validation | default | — | — | 149 real + 149 fake (approx.) | 0.9866 | 1.0000 | 0.9732 | 0.9864 | 1.0000 | 0.0000 | Final 1000-pair checkpoint; confusion matrix 149/149 real correct, 145/149 fake correct. `eer_threshold=0.0735`. |
| 2026-08-06 | ~1000 (same run as above, later eval) | validation | default | — | — | 150 (69+6+1+74) | 0.9533 | 0.9250 | 0.9867 | 0.9548 | 0.9964 | 0.0400 | `eer_threshold=0.5403`. Logged separately from the row above in the original entry; not clearly distinguished whether this is a rerun or a different eval pass — flagged for cleanup. |
| 2026-08-09 | 12 | validation | **reversed** (`SEA_SPOOF_SHARD_REVERSE=1`) | 10 (config default; early-stopping patience=2) | 16 | 4 (2 real + 2 fake) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.7500 | 0.5000 | `eer_threshold=0.5194`. Confusion matrix `[[1,1],[1,1]]` — chance-level at 4 test clips, consistent with the 2026-08-06/12-pair row above: 12 pairs is too small a smoke scale for meaningful metrics regardless of shard order. ROC-AUC (0.75) above chance suggests some ranking signal despite accuracy being exactly chance. |
| 2026-08-09 | 24 | validation | default | 10 (config default; early-stopping patience=2) | 16 | 8 (4 real + 4 fake) | 0.7500 | 1.0000 | 0.5000 | 0.6667 | 0.7500 | 0.5000 | `eer_threshold=0.4444`. Confusion matrix `[[4,0],[2,2]]` — all 4 real clips correctly classified, but only 2/4 fake clips caught (2 fake misclassified as real). Precision=1.0 but recall=0.5 for fake: at this tiny scale the model is defaulting toward predicting "real" rather than learning a balanced boundary. Still too few test clips (8) to trust as a real result — same caveat as the row above. |
| 2026-08-09 | 250 | validation | default | 10 (config default; early-stopping patience=2, triggered after epoch 7) | 32 | 30 (15 real + 15 fake) | 0.8333 | 0.8571 | 0.8000 | 0.8276 | 0.9467 | 0.2000 | `eer_threshold=0.4136`. Best checkpoint at epoch 4 (`val_f1=0.8276`), reloaded after early stopping. Confusion matrix `[[13,2],[3,12]]` — 13/15 real correct, 12/15 fake correct. First mid-scale (250-pair) run in this tracker; between the noisy <30-pair rows above and the 1000-pair rows below, roughly consistent with the trend of improving/stabilizing metrics as pair count grows. |
| 2026-08-09 | 250 (config unchanged from row above) | validation | default | 10 (config default; early-stopping patience=2, triggered after epoch 6) | 32 | 76 (38 real + 38 fake) | 0.9079 | 0.9189 | 0.8947 | 0.9067 | 0.9820 | 0.1053 | `eer_threshold=0.4509`. Best checkpoint at epoch 3 (`val_f1=0.9474`), reloaded after early stopping. Confusion matrix `[[35,3],[4,34]]` — 35/38 real correct, 34/38 fake correct. Same `target_count: 250` as the row above but a notably larger test split (76 vs 30 clips) and much better metrics — `target_count` is a per-source cap, not a guaranteed exact pair count, so shard-matching luck on this run apparently found more matched pairs; flagged since it makes the two "250" rows not directly comparable. |
| 2026-08-09 | 500 | validation | default | 10 (config default; early-stopping patience=2, triggered after epoch 6) | 32 | 150 (75 real + 75 fake) | 0.9867 | 1.0000 | 0.9733 | 0.9865 | 1.0000 | 0.0000 | `eer_threshold=0.0610`. Best checkpoint at epoch 3 (`val_f1=0.9737`), reloaded after early stopping. Confusion matrix `[[75,0],[2,73]]` — all 75 real clips correct, 73/75 fake correct. `target_count` raised to 500 for this run (up from 250 in the two rows above); best metrics in the table so far, consistent with the tracker's overall trend of improving/stabilizing metrics as pair count grows. |

**Reading this table:** accuracy is noisy and near-meaningless below roughly 100 paired
clips (4-12 test clips flips between "collapses to one class" and "exactly 50/50" runs
depending on which 2-4 clips land in the test split) — 1000 pairs is the first scale
in this tracker with a test set large enough (150 clips) for the metrics to be
stable/trustworthy. Treat any smoke-run number below that scale as a pipeline sanity
check, not an accuracy result.

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

---

### 2026-08-09 — Publication-scale paired run and external evaluation

Completed the leakage-safe SEA-Spoof `train` experiment with 3,000 matched
transcript pairs (6,000 clips). Audit: 3,000 unique transcripts, 0 missing
transcripts, 0 cross-split transcripts, 0 incomplete groups, 0 duplicate
paths, and 0 missing features. Each run used 4,800/600/600
train/validation/test clips. Three independent model seeds completed: 42,
1337, and 2026. Same-corpus test accuracy ranged from 0.9950 to 1.0000.

Each model was evaluated on three deterministic external manifests of 100
clips per source (Kathbath, Common Voice male, Common Voice female,
IndicSynth). Mean ± SD across the three independently trained models:

| Metric | Mean ± training-seed SD |
|---|---:|
| Accuracy | 0.8422 ± 0.0193 |
| Balanced accuracy | 0.8907 ± 0.0115 |
| F1 | 0.7586 ± 0.0217 |
| ROC-AUC | 0.9501 ± 0.0205 |
| EER | 0.1098 ± 0.0369 |
| MCC | 0.6914 ± 0.0264 |

Mean source accuracies were Kathbath 0.3944, Common Voice male 1.0000,
Common Voice female 0.9867, and IndicSynth 0.9878. The residual domain gap
is therefore concentrated on Kathbath genuine speech. This is a completed E2
result, not yet evidence that pairing caused an improvement: an identically
evaluated, count-matched non-paired E1 control remains required.

---

### 2026-08-09 — Transcript-leakage correction and clean 500-pair rerun

The new `scripts/audit_experiment_protocol.py` found 164 transcript groups
spanning train/validation/test in the prior 500-pair manifest. Class-wise row
stratification had split paired real and spoof rows independently. Earlier
paired-smoke scores are exploratory and must not be cited as unbiased
generalization estimates.

After transcript-grouped reassignment, the audit reports 250 transcript
groups, 0 cross-split transcripts, 0 incomplete groups, and 678/166/156
train/validation/test rows with balanced labels. The corrected single run
obtained accuracy 0.9295, precision 0.9136, recall 0.9487, F1 0.9308,
ROC-AUC 0.9911, and EER 0.0513; confusion matrix `[[71,7],[4,74]]`.
This supersedes the leaked 0.9867 accuracy / 0.0000 EER result, but remains a
same-source, single-run smoke result rather than the paper's main evidence.

The corrected checkpoint's 15-per-source diagnostic at seed 42 obtained
0.8167 overall: Kathbath 0.6000, CV male 0.7333, CV female 0.9333, and
IndicSynth 1.0000. This remains exploratory sampling.

---

### 2026-08-09 — Smoke checkpoint spot-checked against the four earlier (unseen) sources

Added `scripts/eval_smoke_model_on_legacy_sources.py`: no retraining, just loads the `target_count: 500` SEA-Spoof-paired smoke checkpoint (`checkpoints/smoke_seaspoof_paired/best_model.pt`, the 0.9867-accuracy run logged above) and runs it on random raw clips from the four sources it has never seen — Kathbath (`real`), Common Voice Hindi male/female (`real_cv_m`/`real_cv_f`), IndicSynth (`fake`) — sampled directly from `data/raw/hindi/`.

**Run:** `--samples-per-source 15 --seed 42`.

| source | true label | accuracy |
|---|---|---:|
| real (Kathbath) | real | 0.6000 |
| real_cv_m (CV male) | real | 0.8667 |
| real_cv_f (CV female) | real | 1.0000 |
| fake (IndicSynth) | fake | 0.9333 |

Overall accuracy: 0.8500 (51/60). Confusion: 37 real→real, 8 real→fake, 14 fake→fake, 1 fake→real.

**Interpretation:** this is a single-seed, 60-clip spot check, not a paper-grade result — but it's informative as a first signal. The model, trained only on transcript-paired SEA-Spoof, clears well above chance on all four completely unseen sources, including the unseen fake source (IndicSynth, a different TTS system than SEA-Spoof's), which is a better sign than the earlier pooled-split/factorial work found for cross-source fake generalization. The weak point is Kathbath (0.60, 8/20 misclassified as fake) — worth checking against the confound-audit findings (`scripts/confound_check.py`) for whether Kathbath's channel/silence statistics differ enough from SEA-Spoof's training distribution to explain the miss, before concluding the model has learned genuine source-invariant spoof cues. **Next step:** run this through the proper `scripts/cross_source_holdout_check.py` protocol (fixed splits, multiple seeds) before this becomes a citable generalization claim.

---

### 2026-08-12 — E1 non-paired control: completed, and the causal pairing comparison

Ran the E1 control the paper draft's §5.2 experiment matrix had flagged as the missing piece needed to claim transcript pairing itself (not just sample count or pipeline differences) drives the E2 cross-source result.

**Method.** `configs/config.paper_seaspoof_e1.yaml` — identical to E2's `configs/config.paper_seaspoof.yaml` (same SEA-Spoof sources, `target_count: 3000` for both `real_sea`/`fake_sea`, same EfficientNet-B0 architecture, same AdamW/cosine/50-epoch/patience-7 training budget — see `docs/paper_draft.md` §5.0 for the shared configuration table), differing only in a new `ingestion.allow_pairing: false` flag. When false, `preprocessing/hf_ingestion.ingest_all()` no longer routes matching real/fake sources through `ingest_paired_sources` (transcript matching); real and fake clips are sampled independently up to their own `target_count`, and `scripts/run_nonpaired_seaspoof_experiment.py` assigns splits with ordinary label-stratified `dataset.split_assigner.assign_splits` instead of `assign_grouped_splits` — there is no transcript column to group by. Three independent training seeds: 42, 1337, 2026, matching E2.

**Engineering note:** the first implementation routed the non-paired real/fake sources through the existing `ingest_source` (HF `streaming=True`) path, since it was the only path available for non-paired sources. Because `real_sea`/`fake_sea` are the same underlying dataset/split, this independently re-streamed the same shard bytes twice (once per label) via slow serial byte-range HTTP requests — observably far slower than E2's ingestion, which downloads each shard once via `_download_shard` (`hf_hub_download`, cache-aware) and reads it locally with pyarrow. Added `ingest_cosourced_independent()` to reuse the same single-download-per-shard path as `ingest_paired_sources`, just bucketing rows by label independently instead of matching transcripts. `ingest_all()` now dispatches: `allow_pairing=True` → `ingest_paired_sources`; same-dataset sources with `allow_pairing=False` → `ingest_cosourced_independent`; genuinely separate-dataset sources (Kathbath, Common Voice, IndicSynth) → unchanged `ingest_source` per source. Verified by mocking all three ingestion functions and checking dispatch for the E1, E2, and main configs.

**Same-corpus (in-domain) results**, 600-clip held-out SEA-Spoof test split per seed (ordinary stratified, not transcript-grouped):

| Seed | Accuracy | F1 | ROC-AUC | EER |
|---:|---:|---:|---:|---:|
| 42 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 1337 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 2026 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

All three seeds hit exact 1.0 accuracy (confusion matrix `[[300,0],[0,300]]` every time) — even more perfectly separable in-domain than E2's paired runs (0.9950–1.0000). This confirms same-corpus SEA-Spoof real/fake separability does not depend on transcript pairing at all; channel/vocoder-level cues alone are sufficient in-domain. The pairing effect, if any, can only show up cross-source.

**Cross-source results**, evaluated with `scripts/evaluate_cross_source_protocol.py` against the identical external manifests E2 used (Kathbath, Common Voice male/female, IndicSynth; 100 clips/source, sampling seeds 42/1337/2026 per trained model):

| Training seed | Accuracy | Balanced acc. | F1 | ROC-AUC | EER | MCC |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7525 | 0.5050 | 0.0197 | 0.9655 | 0.0939 | 0.0698 |
| 1337 | 0.7542 | 0.5106 | 0.0449 | 0.8505 | 0.2344 | 0.1044 |
| 2026 | 0.7675 | 0.5394 | 0.1513 | 0.8980 | 0.1717 | 0.2216 |
| **Mean ± SD** | **0.7581 ± 0.0082** | **0.5183 ± 0.0185** | **0.0720 ± 0.0699** | **0.9047 ± 0.0578** | **0.1667 ± 0.0704** | **0.1319 ± 0.0795** |

Per-source accuracy (mean across the three trained models' three sampling manifests each):

| Source | Label | E1 (non-paired) mean acc. | E2 (paired) mean acc. |
|---|---|---:|---:|
| Kathbath | real | 0.9933 ± 0.0067 | 0.3944 |
| Common Voice male | real | 1.0000 ± 0.0000 | 1.0000 |
| Common Voice female | real | 1.0000 ± 0.0000 | 0.9867 |
| IndicSynth | **fake** | **0.0389 ± 0.0391** | 0.9878 |

**Interpretation — this resolves the paper's central deferred claim.** E1's balanced accuracy (0.5183 ± 0.0185) is chance level. Per-source numbers show why: without transcript pairing, the model achieves near-perfect accuracy on every unseen *real* source but catches only 3.89% of unseen spoof clips (IndicSynth) — it collapsed to predicting "real" almost universally on external data. F1 (0.072) and MCC (0.132) near zero confirm this isn't a calibration/threshold artifact like the earlier 2×2 factorial's "swapped"/"real_swap_only" rotations (where ROC-AUC partly survived a broken threshold) — ROC-AUC is comparatively higher here (0.9047) than the collapsed operating-point metrics, but the model never learned to recognize an unseen synthesis system as fake at any usable threshold in practice, and MCC/F1 make clear this is a near-total failure to transfer spoof-detection signal, not just a threshold problem.

E2 (transcript-paired) fails differently and much less severely: it correctly flags 98.78% of unseen IndicSynth spoof clips — a completely different, unseen TTS system than SEA-Spoof — at the cost of misclassifying 60.56% of Kathbath real clips as fake (the channel/silence confound already characterized in the 2026-07-24 entries). Its balanced accuracy (0.8907 ± 0.0115) is far above chance and its MCC (0.6914 ± 0.0264) shows real, substantial transferable signal, not a coin flip.

**Conclusion for the paper:** transcript pairing is not a minor content-control nicety — without it, the detector learns essentially nothing that transfers to an unseen synthesis system (E1: 3.89% fake-source recall externally). With it, the detector transfers to an unseen synthesis system convincingly (E2: 98.78%) while trading in a different, better-characterized weakness (Kathbath channel confound). This is the causal evidence `docs/paper_draft.md` §8 and §11's reproducibility checklist flagged as missing before any claim about pairing's benefit could be made. The E3/E4 normalization ablation (testing whether silence/loudness normalization fixes E2's Kathbath weakness specifically) remains the next open item.

---

### 2026-09-17 — Grad-CAM UI: colorbar legend and peak-region caption

For a live demo, a raw Grad-CAM heatmap asks the viewer to already know "warm color = high influence" and to eyeball time/frequency positions themselves. Added two pieces to close that gap:

- `explainability.visualizer.render_colormap_legend`: a static colorbar image (same colormap as the overlay) labeled "High"/"Low influence," rendered once and shown alongside every Grad-CAM output in the Gradio demo.
- `explainability.visualizer.describe_peak_region`: finds the highest-valued Grad-CAM pixels (top decile, floored to at least half the map's max) and maps their row/column position back to an approximate frequency band (mel scale, `CONFIG.features.fmin/fmax`) and time window (`CONFIG.audio.target_duration_seconds`), e.g. `"1.7-4.0 kHz, 0.41-3.34s"`. This is approximate — the spectrogram is resized to a square before the model sees it, so the mapping is proportional, not read off the original FFT bins — but sufficient to tell a non-technical viewer where to look without interpreting a heatmap by eye.

Wired both into `ExplanationResult` (new `peak_region_caption` field, populated in `explain_audio_file`), into `deployment/gradio_app.py`'s UI (heatmap + legend side-by-side, plus a caption sentence naming the label, confidence, and driving region), and into the Flask API (`deployment/app.py`'s `/predict` response now includes `gradcam_peak_region`). Also added a one-click real/fake example-clip picker to the Gradio demo (`gr.Examples`, pointing at existing Hindi raw clips and, once ingested, English WaveFake clips) so a reviewer doesn't need their own audio file to try it.

**Verification:** ran end-to-end against the existing `checkpoints/best_model.pt` (Hindi) on one real and one fake clip — both predicted correctly (confidence ~1.0 each) with visibly distinct peak regions (real: 0.0-0.9 kHz / 0.00-1.92s; fake: 1.7-4.0 kHz / 0.41-3.34s), consistent with the general expectation that synthetic-speech artifacts concentrate at different frequency bands than natural speech, though this is a two-clip spot check, not a systematic audit of what the model attends to across many clips.

---

### 2026-09-17 — English pipeline: WaveFake benchmark smoke run (in progress)

First extension of the pipeline beyond Hindi. Added `configs/config.english_smoke.yaml` and `scripts/run_english_smoke.py`, reusing the existing generic (language-driven-by-config) ingestion/preprocessing/feature/training/eval pipeline unchanged — no pipeline code required a Hindi-specific assumption, confirming the "no hardcoded paths/languages" design principle held up under a real second-language test.

**Dataset:** [WaveFake](https://arxiv.org/abs/2111.02813) (Frank & Schonherr, 2021) — real English speech from LJSpeech, fake speech from 7 distinct neural vocoders (WF1-WF7) — via the community HF parquet mirror `ajaykarthick/wavefake-audio` (not the official Zenodo-distributed release; chosen because it drops into the existing bounded-streaming ingestion path with zero new code, verified by test-streaming it directly before committing to it). Considered official ASVspoof2019 first, but its only HF mirror (`LanceaKing/asvspoof2019`) uses a legacy loading-script format the project's `datasets` library version can't load (the same class of problem that already forced the Common Voice Hindi source onto a Parquet mirror) and the official release would require a new download+protocol-parsing path — not attempted given the same-night deadline.

**Design choice:** one `ingestion.fake` source per vocoder (WF1..WF7, 200 clips each) rather than one pooled fake filter, mirroring this project's existing multi-source-per-class philosophy (see 2026-07-20 entry) so the fake class isn't defined by a single synthesis system's artifacts. Real: 1400 LJSpeech clips.

**Known limitation, not yet fixed:** WaveFake rows share an underlying LJSpeech utterance ID across the real clip and all 7 fake variants (confirmed directly: `real_or_fake="R"` and `"WF1".."WF7"` rows for the same `audio_id` carry the same transcript-adjacent content), but that ID is not threaded through the transcript-pairing sidecar this project built for SEA-Spoof (`pair_transcripts.csv` / `assign_grouped_splits`) — that mechanism was wired specifically for SEA-Spoof's `text` column. This run uses ordinary label-stratified splits (`dataset.split_assigner.assign_splits`), so same-utterance clips may land in different partitions. Given this project's own finding that pooled/ungrouped splits inflate same-corpus metrics (see 2026-07-23 entry), **any same-corpus accuracy from this run should be read the same way as that early pooled-split Hindi result: a pipeline/architecture sanity check, not a citable number**, until utterance-ID grouping is added the way transcript grouping was for SEA-Spoof.

**Status at time of writing:** ingestion running (real-clip streaming ~50% complete); training/evaluation not yet finished. This entry will be updated with final same-corpus metrics once the run completes — recorded now, before the result is known, specifically so the "smoke test, not a result" framing above isn't written after seeing the number.

---

### 2026-09-18 — English pipeline: switched to transcript-paired ingestion (correcting the above)

The unpaired approach above was stopped before preprocessing/training started, once it was clear it repeated exactly the same gap the SEA-Spoof pairing work already fixed for Hindi. Rather than accept that gap for a first English result, added `preprocessing.hf_ingestion.ingest_wavefake_paired`: pairs each real clip with exactly one fake vocoder variant sharing its `audio_id`, deterministically rotated across WF1-WF7 (`hash(audio_id) % 7`) so every vocoder is represented across the dataset while each individual pair stays 1:1 — the same shape as a SEA-Spoof `(real_sea_NNNNNN, fake_sea_NNNNNN)` pair, sharing the same `pair_transcripts.csv` sidecar mechanism and `assign_grouped_splits` protection, so a pair's real+fake clips can no longer land in different train/val/test partitions.

Engineering note: this single-pass approach (stream once, keep whichever row matches either "R" or that utterance's assigned vocoder, discard the rest) also turned out to be more network-efficient than the original independent-per-source `ingest_source` calls, which each restarted the HF stream from row 0 once per source (8 separate full restarts for real + 7 vocoders) — the paired approach scans the stream exactly once.

Verified against a 5-pair test run: `pair_transcripts.csv` showed exactly 5 matched `(real_wf_NNNNNN, fake_wf_NNNNNN)` pairs sharing identical `audio_id` transcript keys (e.g. `LJ001-0001`), 40 rows scanned total (5 pairs x 8 rows/utterance, as expected since the vocoder-matching filter still has to see every row for an utterance before it knows whether it matches).

Updated `configs/config.english_smoke.yaml` to `target_count: 1000` pairs (1000 real + 1000 fake, matching the Hindi tracker's own historical judgment that 1000 pairs was "the first scale ... with a test set large enough ... for the metrics to be stable/trustworthy," see the smoke-tracker table's note). `scripts/run_english_smoke.py` now calls `ingest_wavefake_paired` directly (bypassing `ingest_all()`, since the generic multi-source dispatch in `hf_ingestion.py` doesn't have a concept of "N fake variants per one real clip") and uses `assign_grouped_splits(group_column="transcript")` instead of the plain stratified `assign_splits`. Result pending -- see the next entry once the run completes.

**Bug found and fixed mid-run:** the first version of `ingest_wavefake_paired` only wrote the `pair_transcripts.csv` sidecar once, at the very end of ingestion (batched, not incremental). An interrupted run left 100 pairs' audio files on disk with no recorded transcript link -- an unrecoverable pairing loss on crash, not just a resume inefficiency. Fixed by flushing the sidecar after every single pair. The 100 orphaned pairs were removed and the remaining 900 good pairs renumbered contiguously (0..899) before resuming ingestion to the 1000-pair target, to avoid the renumbered/orphan-deleted state colliding with `existing_pairs`-based resume counting on the next run. `data/english_smoke/metadata`, `processed`, and `features` were cleared and rebuilt fresh afterward, since `preprocessing.metadata_builder` derives each row's `audio_id` (a UUID) from the raw filepath — renaming raw files invalidates all previously-derived IDs.

---

### 2026-09-18 — English pipeline: WaveFake same-corpus smoke result

Completed run: 1000 transcript-paired real/fake clips (LJSpeech + WaveFake vocoders, rotated WF1-WF7), grouped 70/15/15 split (700/150/150 pairs -> 149/149 real/fake in the realized test split after grouping), EfficientNet-B0, 25-epoch budget with early stopping (patience 5, triggered after epoch 12, best checkpoint from epoch 7 reloaded for final eval), GPU (RTX 4050).

**Test-split result:**

| Metric | Value |
|---|---:|
| Accuracy | 1.0000 |
| Balanced accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| MCC | 1.0000 |
| ROC-AUC | 1.0000 |
| EER | 0.0000 |

Confusion matrix: 149/149 real correct, 149/149 fake correct (`eer_threshold=0.7735`).

**Interpretation -- read this the same way the project's own E1 control and the very first pooled-split Hindi run (2026-07-23) were read, not as a generalization result:** a perfect same-corpus score here is expected, not surprising. WaveFake's 7 vocoders leave distinctive channel/synthesis artifacts that a CNN can separate cleanly when train and test are drawn from the same corpus, transcript-paired or not -- this project's own E1 non-paired control hit an identical 1.0/1.0/1.0/0.0 in-domain on SEA-Spoof (see 2026-08-12 entry) despite later failing almost completely cross-source (3.89% fake recall on unseen data). This English number demonstrates the pipeline, ingestion, pairing, and training loop all work correctly end-to-end on a second language and a second benchmark -- it is not evidence the model would detect an unseen English TTS/vocoder system, which is exactly the question the Hindi cross-source factorial exists to answer and which this run does not attempt.

**What would be needed before this number means anything about generalization:** a second, disjoint English source (e.g. an ASVspoof2019/2021 LA subset, or a different vocoder set entirely) held out the way `scripts/cross_source_holdout_check.py` holds out Common Voice/SEA-Spoof from Kathbath/IndicSynth. Not attempted tonight given the deadline -- flagged here as the immediate next step before citing this pipeline's English result as anything beyond a working smoke test.
