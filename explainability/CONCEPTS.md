# Understanding Grad-CAM / explainability

## The problem this solves

A trained neural network is a black box by default — it gives you a prediction ("fake," 92% confidence) but not a reason. For a security-relevant tool, "trust me" isn't good enough; a reviewer wants to know *what in the audio* made the model say "fake." Grad-CAM is one way to partially open that box.

## The core idea, without the math

Think of the model's decision as being built up region-by-region as it scans the spectrogram image. Grad-CAM asks a very specific, answerable question: **"If I nudge the values in this particular region of the image slightly, how much does the model's 'fake' score change?"** Regions where a small nudge causes a big change in the score are regions the model is "paying attention to." Regions where nudging does almost nothing are regions the model is ignoring.

Doing this for every region of the spectrogram produces a heatmap — brighter/warmer color = more influence on the final decision. That heatmap is then overlaid on top of the original spectrogram image so a human can see both at once (this is the `outputs/explanations/<clip>_gradcam.png` file).

## Why a spectrogram, and why this looks like an "image explanation"

This project doesn't feed raw audio waveforms into the model — it converts each clip into a mel-spectrogram first (see `feature_extraction/CONCEPTS.md`), which is a 2-D image where one axis is time and the other is frequency. Because the model architecture is an image classifier (EfficientNet-B0), Grad-CAM — originally built for image classifiers — works out of the box. The heatmap tells you *which time window and which frequency band* of the clip drove the prediction, not which audio "sounds" (there's no direct sound-to-region translation without also looking at the spectrogram axes).

## What Grad-CAM tells you — and, just as important, what it doesn't

**It tells you:** which pixels of the spectrogram had the most influence on the score the model actually produced.

**It does NOT tell you:**
- **Whether that's the *correct* reason.** A model can attend to a real acoustic artifact of synthetic speech (good) or to a shortcut like recording-channel noise/silence padding (bad) — Grad-CAM lights up either way. You still need domain judgment (or a diagnostic like the confound/embedding analysis, see `scripts/CONCEPTS.md`) to tell which one you're looking at.
- **Causal proof.** It's a correlational signal ("this region mattered to this prediction"), not a guarantee that manipulating that region would flip the true label.
- **Global model behavior.** Each Grad-CAM image explains one single prediction on one single clip — it says nothing about how the model behaves in general until you look at many of them and look for a pattern.

## Why this matters for this specific project

Given the research log's finding that the pooled-split model likely partly relies on a channel/silence shortcut, Grad-CAM overlays are genuinely useful evidence here: if the heatmap on a "fake" prediction consistently lights up on trailing silence/padding regions rather than the speech itself, that's a red flag pointing at the same shortcut the confound analysis found statistically — two independent methods agreeing, which is much stronger evidence than either alone.

## Turning the heatmap into a sentence: the peak-region caption

A raw heatmap still asks the viewer to eyeball "which part is brightest" and translate that into time/frequency themselves. `explainability/visualizer.describe_peak_region` does that translation automatically: it finds the highest-valued pixels in the Grad-CAM map and converts their row/column position back into an approximate frequency band (via the mel scale, `CONFIG.features.fmin/fmax`) and time window (via `CONFIG.audio.target_duration_seconds`), producing a caption like `"1.7-4.0 kHz, 0.41-3.34s"`. This is approximate, not exact — the spectrogram is resized to a square before the model ever sees it, so the mapping back is proportional, not read off the original per-frame FFT bins — but it's precise enough to tell a viewer *where to look* without them having to interpret a colormap by eye. The Gradio demo (`deployment/gradio_app.py`) also renders a colorbar legend (`render_colormap_legend`) alongside the heatmap so "warm color = high influence" doesn't have to be assumed knowledge.

## Rational summary

- **"How does Grad-CAM know what mattered?"** — it uses the gradient of the predicted class score with respect to the model's internal feature maps, weighting each spatial location by how much it influenced the final score, then overlays that as a heatmap. (You don't need calculus to explain this — "how much does nudging this region change the answer" is the whole intuition.)
- **"Is a Grad-CAM heatmap proof the model is right?"** — no. It's proof of *what the model looked at*, not proof the model looked at the *correct* thing. Explaining a wrong answer still produces a heatmap.
- **"Why not just trust the confidence score?"** — confidence tells you *how sure* the model is, not *why*. A model can be very confident and very wrong for the wrong reasons (exactly the shortcut-learning risk this project investigates).
