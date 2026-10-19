"""Flask inference service: upload audio, get back a prediction, per-window scores and Grad-CAM overlay.

Request flow:
    POST /predict (multipart form, field "audio"; wav/flac/mp3/ogg/opus, plus m4a/amr/... with ffmpeg)
        -> validate extension/size
        -> forensics.analyze.analyze_recording: score the whole recording in
           overlapping model-length windows (same preprocessing and features
           as training), then Grad-CAM the most suspicious window
        -> JSON response: {predicted_label, confidence (of the most suspicious
                           window), fake_score, verdict, most_suspicious_window_seconds,
                           segments, evidence_sha256, gradcam_image_base64, gradcam_peak_region}

    POST /forensic-report (multipart form: "audio", "case_id", "examiner")
        -> forensics.analyze.analyze_evidence: hashes the evidence, predicts,
           writes PDF + JSON reports, appends the chain-of-custody log
        -> the PDF report as a download (verdict/hash/report ID in X-* headers)
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_file
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from configs.config import CONFIG
from deployment.inference_service import InferenceService
from forensics.analyze import analyze_evidence, analyze_recording
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

    def _validated_upload() -> tuple[FileStorage, str] | tuple[Response, int]:
        """Return ``(file, safe_filename)`` for the "audio" field, or an error response."""
        if "audio" not in request.files:
            return jsonify({"error": "Request must include a file under the 'audio' field."}), 400

        uploaded_file = request.files["audio"]
        if not uploaded_file.filename:
            return jsonify({"error": "No file selected."}), 400

        filename = secure_filename(uploaded_file.filename) or "upload"
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
        return uploaded_file, filename

    @app.post("/predict")
    def predict() -> tuple[Response, int] | Response:
        upload = _validated_upload()
        if isinstance(upload[0], Response):
            return upload
        uploaded_file, filename = upload

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / filename
            uploaded_file.save(temp_path)

            try:
                segment_analysis, result = analyze_recording(temp_path, service, display_name=filename)
            except ValueError as exc:
                logger.warning("Rejected upload '%s': %s", filename, exc)
                return jsonify({"error": str(exc)}), 400
            except Exception:
                logger.exception("Prediction failed for uploaded file '%s'.", filename)
                return jsonify({"error": "Internal error during prediction."}), 500

            gradcam_base64 = _encode_overlay_as_base64_png(result.overlay_image)

        suspicious = segment_analysis.most_suspicious
        return jsonify(
            {
                "predicted_label": result.predicted_label,
                "confidence": result.confidence,
                "fake_score": suspicious.fake_score,
                "verdict": suspicious.verdict,
                "most_suspicious_window_seconds": [suspicious.start_seconds, suspicious.end_seconds],
                "segments": [segment.to_dict() for segment in segment_analysis.segments],
                "gradcam_image_base64": gradcam_base64,
                "gradcam_peak_region": result.peak_region_caption,
            }
        )

    @app.post("/forensic-report")
    def forensic_report() -> tuple[Response, int] | Response:
        upload = _validated_upload()
        if isinstance(upload[0], Response):
            return upload
        uploaded_file, filename = upload

        case_id = request.form.get("case_id", "")
        examiner = request.form.get("examiner", "")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / filename
            uploaded_file.save(temp_path)
            try:
                analysis = analyze_evidence(
                    temp_path,
                    case_id=case_id,
                    examiner=examiner,
                    service=service,
                    model_label=f"EfficientNet-B0 deepfake detector ({service.checkpoint_path.parent.name})",
                    original_filename=uploaded_file.filename,
                )
            except ValueError as exc:
                logger.warning("Rejected upload '%s': %s", filename, exc)
                return jsonify({"error": str(exc)}), 400
            except Exception:
                logger.exception("Analysis failed for uploaded file '%s'.", filename)
                return jsonify({"error": "Internal error during analysis."}), 500

        findings = analysis.findings
        response = send_file(analysis.pdf_path, mimetype="application/pdf", as_attachment=True)
        response.headers["X-Report-Id"] = findings.report_id
        response.headers["X-Verdict"] = findings.verdict
        return response

    logger.info("Flask app created (device inference service ready).")
    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host=CONFIG.deployment.host, port=CONFIG.deployment.port, debug=CONFIG.deployment.debug)
