# Understanding deployment / inference

## Training vs. inference — two different phases with different needs

**Training** is the slow, offline process of adjusting a model's weights against labeled data — it needs the full dataset, a GPU running for a long time, and no urgency about response speed. **Inference** is using an *already-trained, frozen* model to make a prediction on one new input, as fast and reliably as possible. Everything in `deployment/` is about inference only — no weights ever change here, the model is loaded once and reused for every request.

## Why the model is loaded exactly once, not per request

Loading a model checkpoint (reading weights from disk, constructing the network, moving it to GPU) takes real time — doing it fresh for every incoming request would make each prediction unnecessarily slow. `InferenceService.__init__` loads the checkpoint once when the service starts, and every subsequent `.predict()` call reuses that already-loaded model in memory. This is the whole reason `InferenceService` exists as its own class separate from the web-framework code — it owns the model's lifecycle independently of how many requests come and go.

## Why the exact same preprocessing/feature code is reused from training

If inference preprocessed audio even slightly differently than training did, the model would effectively be looking at different input than it was trained to interpret — this is a well-known, subtle failure mode in deployed ML systems ("training/serving skew"). This project avoids it by construction: `explainability/explain.py`'s `explain_audio_file` calls the identical `preprocessing.transforms.preprocess_audio` and `FeatureExtractor.extract_array` functions the offline training pipeline uses, so what the model sees at inference time is guaranteed to match what it saw during training/evaluation.

## Flask API vs. Gradio UI — same brain, two different "faces"

Both `deployment/app.py` (Flask) and `deployment/gradio_app.py` (Gradio) wrap the exact same `InferenceService` — neither reimplements prediction logic, they just present it differently:

- **Flask** exposes a raw HTTP JSON API (`POST /predict` with an audio file, returns `{predicted_label, confidence, gradcam_image_base64}`). This is meant to be called *by other software* — a frontend app, a script, another service — not used directly by a human.
- **Gradio** provides an interactive web page a human can use directly in a browser (upload or record audio, immediately see the label, confidence, and Grad-CAM heatmap rendered as an image) — no HTTP client or JSON parsing needed. It's the right choice for a demo, a poster-session link, or letting a non-technical reviewer try the model themselves.

Because both call the same `InferenceService`, a prediction made through the Flask API and the same clip run through Gradio will always agree — there's exactly one place the actual prediction logic lives.

## What "confidence" actually means here

The model's raw output (logits) is passed through softmax to become a probability distribution over the two classes; "confidence" is just the probability assigned to whichever class the model predicted (`torch.argmax`). A high confidence means the model's internal score was strongly skewed toward that class — it says nothing on its own about whether that class is *correct* (see the calibration discussion in `scripts/CONCEPTS.md`: a model can be very confident and very wrong under domain shift).

## Rational summary

- **"Why maintain two separate deployment interfaces instead of just one?"** — they serve different consumers: Flask for programmatic integration, Gradio for human interactive use — maintaining one shared `InferenceService` underneath means this doesn't duplicate any actual prediction logic, only the presentation layer.
- **"What would happen if preprocessing changed after training but before deployment?"** — the model would receive inputs shaped differently than what it learned to interpret, likely degrading predictions in an inconsistent way; this project prevents that by sharing one preprocessing code path between training and inference rather than reimplementing it twice.
- **"Is a high-confidence wrong prediction possible?"** — yes, and this project has direct evidence of it (see the ElevenLabs example discussed in this project's conversation history / research log) — confidence reflects the model's internal certainty given what it learned, not ground truth, especially for inputs unlike anything in its training distribution.
