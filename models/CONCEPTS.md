# Understanding the model architecture

## Why treat audio like an image at all

The model in this project (EfficientNet-B0) was originally designed to classify photographs. This project doesn't feed it audio waveforms — it first converts each clip into a mel-spectrogram, a 2-D grid of numbers (time along one axis, frequency along the other, loudness as the "pixel" value). Once audio is turned into a 2-D grid, an image classifier can be pointed at it directly. This is a common, well-established trick in audio ML — "borrow a proven image architecture instead of designing an audio one from scratch."

## Transfer learning / pretrained backbone — why not train from scratch

Training a large CNN entirely from random starting weights needs enormous amounts of data to learn basic visual concepts (edges, textures, shapes) before it can even start learning the task-specific ones. **Transfer learning** shortcuts this: start from a version of EfficientNet-B0 already trained on ImageNet (millions of ordinary photographs), which has already learned generally useful pattern-detection filters, and only fine-tune it on this project's much smaller dataset (~8,800 clips) for the real/fake task specifically. This is why `configs/config.yaml` has `pretrained: true` — the model starts smart instead of starting blank.

Interesting technical wrinkle: ImageNet photos have 3 color channels (RGB), but a spectrogram has just 1 channel (loudness). The code (`models/backbone.py`) adapts the pretrained first layer's weights from 3 channels down to 1 automatically — the pretrained knowledge is preserved, just reshaped to fit single-channel input.

## Backbone vs. classifier head — two different jobs in one model

- **Backbone** (EfficientNet-B0 itself): takes in the spectrogram image and boils it down into a compact 1280-number summary (the **embedding**, discussed in `scripts/CONCEPTS.md`) — this is the "understanding" part. It doesn't know what "real" or "fake" mean; it just extracts patterns.
- **Classifier head** (a small dropout + linear layer, in `models/baseline_classifier.py`): takes that 1280-number summary and draws a boundary through it to output "real" or "fake." This is the only part of the model that's actually task-specific.

Splitting the model this way is why the embedding-analysis diagnostic (`scripts/CONCEPTS.md`) can inspect the backbone's output *before* the classifier makes its final call — it lets you check whether the "understanding" part is any good, separately from whether the final decision boundary happens to work.

## Pooling — how a 2-D feature map becomes a single list of numbers

Partway through the network, the backbone still has a *spatial* feature map (roughly a smaller grid, still with a time axis and a frequency axis, but each grid cell now holding a richer summary instead of a raw loudness value). **Global average pooling** collapses this entire grid into a single number per feature channel, by averaging — turning a 7×7 grid of 1280 channels into just 1280 numbers total. This is what makes the embedding a fixed-size vector regardless of the exact spectrogram size, and what feeds the classifier head.

## Freezing a backbone — a knob this project has but doesn't currently use

`freeze_backbone` (currently `false` in config) controls whether the pretrained backbone's weights are allowed to keep changing during training on this project's data, or locked in place so only the classifier head learns. Freezing trains faster and needs less data but limits how much the model can adapt its "understanding" to spectrograms specifically (vs. natural photos); not freezing (this project's choice) lets the whole network adapt, at the cost of needing more data/epochs and some risk of overfitting to the smaller dataset.

## Dropout — a simple overfitting guard

Dropout randomly zeroes out a fraction of values during training (here, before the final classification layer) so the classifier can't become overly dependent on any single feature always being present. It's a widely-used, cheap regularization technique, not something specific to this project's design.

## The optional BiLSTM head — what it would add and why it's optional

The spectrogram's time axis has an order — sound at t=1s comes before sound at t=2s. The plain baseline model (via pooling) discards that order, treating the whole clip as "a bag of patterns" with no sense of sequence. `models/efficientnet_bilstm.py` is an alternative that adds a **BiLSTM** (bidirectional recurrent layer) after the backbone, explicitly modeling how the pattern evolves *forward and backward through time* before classifying — potentially catching temporal artifacts (like unnatural pacing/rhythm typical of some TTS systems) that pure spatial pooling would miss. It's config-gated (`use_bilstm`) precisely so it can be compared against the simpler baseline to check whether the added complexity earns its keep — the baseline exists explicitly as "the accuracy floor it must beat," per that module's own docstring.

## Rational summary

- **"Why EfficientNet and not a custom CNN?"** — it's a well-studied, efficient, pretrained architecture; using it means benefiting from ImageNet's learned visual patterns (transfer learning) rather than needing a huge audio-specific dataset to learn basic pattern detection from zero.
- **"What exactly is the 1280-number embedding? Is it interpretable?"** — it's the backbone's compressed summary of a clip, not directly human-readable number-by-number, but its *geometry* (which clips land near which other clips) is interpretable via visualization (t-SNE) and is exactly what the shortcut-learning diagnostics inspect.
- **"Why does freezing/not freezing the backbone matter for this project's shortcut-learning story?"** — an unfrozen backbone can adapt its features specifically to whatever pattern most easily separates the training labels, including a shortcut, if one is available and easier than the genuine task — this is *part of* why cross-source testing (not just accuracy) is necessary to check what it actually adapted to.
