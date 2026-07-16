"""Flask inference service: upload a WAV, get back a prediction, confidence and Grad-CAM overlay.

Request flow:
    POST /predict (multipart form, field "audio")
        -> validate extension/size
        -> preprocessing.transforms.preprocess_audio (same code path as training)
        -> feature_extraction.extractor.FeatureExtractor.extract_array (same code path as training)
        -> model forward pass -> softmax probability
        -> explainability.gradcam.GradCAMExplainer -> heatmap overlay
        -> JSON response: {predicted_label, confidence, gradcam_image_base64}
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

from configs.config import CONFIG
from deployment.inference_service import InferenceService
from utils.logger import get_logger

logger = get_logger(__name__)


def _encode_overlay_as_base64_png(overlay_image) -> str | None:
    """Encode an RGB overlay image as a base64 PNG string for JSON transport.

    Args:
        overlay_image: ``uint8`` RGB array, shape ``(H, W, 3)``.

    Returns:
        Base64-encoded PNG string, or ``None`` if encoding failed.
    """
    success, encoded = cv2.imencode(".png", cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
    if not success:
        return None
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def create_app(inference_service: InferenceService | None = None) -> Flask:
    """Construct the Flask application, loading the model exactly once.

    Args:
        inference_service: Pre-constructed inference service to use instead
            of loading a fresh one. Primarily for tests, which can inject a
            stub. Defaults to a new :class:`InferenceService`.

    Returns:
        A configured :class:`flask.Flask` application instance.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = CONFIG.deployment.max_upload_size_mb * 1024 * 1024

    service = inference_service or InferenceService()

    @app.get("/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.post("/predict")
    def predict() -> tuple[Response, int] | Response:
        if "audio" not in request.files:
            return jsonify({"error": "Request must include a file under the 'audio' field."}), 400

        uploaded_file = request.files["audio"]
        if not uploaded_file.filename:
            return jsonify({"error": "No file selected."}), 400

        filename = secure_filename(uploaded_file.filename)
        extension = Path(filename).suffix.lower()
        if extension not in CONFIG.deployment.allowed_extensions:
            return (
                jsonify(
                    {
                        "error": f"Unsupported file extension '{extension}'. "
                        f"Allowed: {CONFIG.deployment.allowed_extensions}."
                    }
                ),
                400,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / filename
            uploaded_file.save(temp_path)

            try:
                result = service.predict(temp_path)
            except ValueError as exc:
                logger.warning("Rejected upload '%s': %s", filename, exc)
                return jsonify({"error": str(exc)}), 400
            except Exception:
                logger.exception("Prediction failed for uploaded file '%s'.", filename)
                return jsonify({"error": "Internal error during prediction."}), 500

            gradcam_base64 = _encode_overlay_as_base64_png(result.overlay_image)

        return jsonify(
            {
                "predicted_label": result.predicted_label,
                "confidence": result.confidence,
                "gradcam_image_base64": gradcam_base64,
            }
        )

    logger.info("Flask app created (device inference service ready).")
    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host=CONFIG.deployment.host, port=CONFIG.deployment.port, debug=CONFIG.deployment.debug)
