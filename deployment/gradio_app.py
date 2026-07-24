"""Gradio demo: upload or record a clip, get back a prediction and a Grad-CAM overlay.

Sits alongside :mod:`deployment.app` (the Flask JSON API) rather than
replacing it -- both wrap the same :class:`deployment.inference_service.InferenceService`,
so predictions are guaranteed identical between the two. This module exists
for a interactive, linkable demo (e.g. hosted on Hugging Face Spaces) that
needs no HTTP client to try, which the Flask API alone doesn't provide.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from dataset.deepfake_dataset import LABEL_TO_INDEX
from deployment.inference_service import InferenceService
from utils.logger import get_logger

logger = get_logger(__name__)

_OTHER_LABEL = {"real": "fake", "fake": "real"}


def create_demo(inference_service: InferenceService | None = None) -> gr.Blocks:
    """Construct the Gradio demo, loading the model exactly once.

    Args:
        inference_service: Pre-constructed inference service to use instead
            of loading a fresh one. Primarily for tests, which can inject a
            stub. Defaults to a new :class:`InferenceService`.

    Returns:
        A configured, unlaunched :class:`gradio.Blocks` app.
    """
    service = inference_service or InferenceService()

    def predict_and_explain(audio_path: str | None) -> tuple[dict[str, float], np.ndarray]:
        if audio_path is None:
            raise gr.Error("Upload or record an audio clip first.")

        try:
            result = service.predict(Path(audio_path))
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        other_label = _OTHER_LABEL[result.predicted_label]
        label_probs = {
            result.predicted_label: result.confidence,
            other_label: 1.0 - result.confidence,
        }
        return label_probs, result.overlay_image

    demo = gr.Interface(
        fn=predict_and_explain,
        inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio clip (Hindi speech)"),
        outputs=[
            gr.Label(num_top_classes=len(LABEL_TO_INDEX), label="Prediction"),
            gr.Image(label="Grad-CAM explanation (mel-spectrogram)"),
        ],
        title="Hindi Deepfake Audio Detector",
        description=(
            "EfficientNet-B0 over mel-spectrograms, trained to tell real Hindi speech "
            "(Kathbath, Common Voice) apart from AI-generated speech (IndicSynth, SEA-Spoof). "
            "The Grad-CAM overlay highlights which time/frequency regions of the spectrogram "
            "drove the prediction. See the project README for cross-source generalization caveats."
        ),
        flagging_mode="never",
    )
    logger.info("Gradio demo created (device inference service ready).")
    return demo


if __name__ == "__main__":
    app = create_demo()
    app.launch()
