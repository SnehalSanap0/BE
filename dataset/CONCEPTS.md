# Understanding train/val/test splits

## Why data has to be split at all

If you train a model and then test it on the *same* examples it trained on, you learn nothing about whether it generalizes — it could simply be memorizing. Splitting data into separate groups lets you check performance on examples the model has genuinely never seen. This project uses three groups, each with a distinct job:

- **Train** — the examples the model actually learns from (adjusts its weights on).
- **Validation (val)** — used *during* training to make decisions like "which epoch's checkpoint is the best one so far" and "should training stop early" (`training.early_stopping_patience`), without ever letting the model directly learn from these examples.
- **Test** — touched only at the very end, to report final performance. If you use test-set performance to make *any* decision during development (like tuning a hyperparameter), it silently stops being a fair, unbiased estimate.

## Stratified splitting — keeping class balance consistent across splits

A plain random split could, by bad luck, put far more "fake" clips in train and far fewer in test (or vice versa), especially with a smaller dataset. **Stratified** splitting instead splits *within each class separately*, so train/val/test each end up with roughly the same real-to-fake ratio as the full dataset. `dataset/split_assigner.py` does this by label, using the ratios in `configs/config.yaml` (`train_ratio: 0.8`, `val_ratio: 0.1`, `test_ratio: 0.1`).

## Why the *default* pooled random split can be misleading here — the central lesson of this project

Stratifying by label alone controls for class balance, but says nothing about *which source dataset* each clip came from. Since this project pools multiple independent source datasets per label (Kathbath + Common Voice for "real"; IndicSynth + SEA-Spoof for "fake"), a standard stratified-by-label split still lets every source dataset land in train, val, *and* test simultaneously. That means any shortcut tied to "which source is this" (see `scripts/CONCEPTS.md`) is available during training *and* still present, working just as well, during testing — so the test-set numbers can look excellent even if the model never learned the intended task at all.

## Source-based (cross-source) holdout splitting — the alternative used to check for this

Instead of splitting individual clips randomly, `scripts/cross_source_holdout_check.py` splits by *entire source dataset*: some source datasets are reserved for training/validation only, and the *other, completely different* source datasets are reserved for testing only, with no overlap. This directly answers "if I'd never seen this recording setup before, would I still get it right?" — which the standard split structurally cannot answer, because it never denies the model access to any source.

Practically, this project doesn't need two separate copies of the raw audio for this — since preprocessing/feature extraction are already done and cached, `build_holdout_metadata` just re-labels which rows count as "train"/"val"/"test" in a separate copy of `metadata.csv`, keyed by which `source_tag` each row's filename indicates it came from (reusing every already-extracted feature file, no reprocessing needed).

## Rational summary

- **"Isn't a random split the 'correct'/standard way to do it?"** — it's the standard way when you only care about one dataset's internal generalization. It stops being sufficient the moment your dataset for one class is actually a *pool of several different sources* that differ in ways unrelated to the label — which is exactly this project's situation, and exactly why it uses cross-source holdout testing as well, not instead.
- **"Why not just always use source-based splitting, everywhere, and skip the random split entirely?"** — the random-split result isn't useless; it's an *upper bound* / sanity check that the model can fit the data at all, and comparing it against the source-based result is itself the evidence that a shortcut existed (a large gap between the two numbers proves it; a small gap would have suggested there wasn't much of a shortcut to find).
- **"What happens to validation in the source-based holdout — does it also risk leaking sources into test?"** — no: `build_holdout_metadata` draws validation clips only from the *training* sources (a small held-back fraction of them), while the holdout sources are reserved for test exclusively — validation is used only to pick the best checkpoint during training, never to inform anything about the never-seen test sources.
