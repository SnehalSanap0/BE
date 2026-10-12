# Audio Deepfake Detection – HTTP API

Flask service, source: `deployment/app.py`. Start it with `python -m deployment.app`.

- **Base URL (default):** `http://localhost:5000`
- **Auth:** none
- **Upload limit:** 25 MB (`deployment.max_upload_size_mb`). Larger uploads get HTTP 413.
- **Accepted formats:** set by `deployment.allowed_extensions`. The default `configs/config.yaml` allows
  `.wav .flac .mp3 .ogg .opus .m4a .aac .amr .3gp .webm .wma`. The paper/sea-spoof configs allow `.wav` only.
  Non-WAV formats other than flac/mp3/ogg/opus need `ffmpeg` on the server.
- **Errors:** every error is JSON `{"error": "<message>"}` with HTTP 400 (bad input) or 500 (server failure).

---

## GET /health

Liveness check.

**200**
```json
{ "status": "ok" }
```

---

## POST /predict

Scores the whole recording in overlapping windows and returns the verdict, per-window scores and a Grad-CAM image for the most suspicious window.

**Request:** `multipart/form-data`

| Field   | Type | Required | Notes       |
|---------|------|----------|-------------|
| `audio` | file | yes      | Audio file. |

**200 response**

| Field                            | Type                  | Description |
|----------------------------------|-----------------------|-------------|
| `predicted_label`                | string                | Model label for the most suspicious window (`"real"` / `"fake"`). |
| `confidence`                     | number 0–1            | Model confidence for that window. |
| `fake_score`                     | number 0–1 \| null    | Probability the audio is synthetic, from the most suspicious window. |
| `verdict`                        | string \| null        | `"LIKELY GENUINE"`, `"INCONCLUSIVE"` or `"LIKELY SYNTHETIC"`. |
| `most_suspicious_window_seconds` | `[start, end]`        | Time range of that window, in seconds. |
| `segments`                       | array                 | One entry per analysis window, in time order. See below. |
| `evidence_sha256`                | string                | SHA-256 of the uploaded file. |
| `gradcam_image_base64`           | string \| null        | Base64 PNG (no `data:` prefix). `null` if encoding failed. |
| `gradcam_peak_region`            | string                | Text caption of where the heatmap peaks. |

`segments[]` item:

| Field            | Type              | Description |
|------------------|-------------------|-------------|
| `start_seconds`  | number            | Window start. |
| `end_seconds`    | number            | Window end. |
| `speech_fraction`| number 0–1        | Share of the window that is non-silent. |
| `fake_score`     | number \| null    | `null` if the window was skipped as silence. |
| `verdict`        | string \| null    | `null` if skipped. Same three values as above. |

Example:
```json
{
  "predicted_label": "fake",
  "confidence": 0.93,
  "fake_score": 0.93,
  "verdict": "LIKELY SYNTHETIC",
  "most_suspicious_window_seconds": [2.0, 6.0],
  "segments": [
    { "start_seconds": 0.0, "end_seconds": 4.0, "speech_fraction": 0.8, "fake_score": 0.41, "verdict": "INCONCLUSIVE" },
    { "start_seconds": 2.0, "end_seconds": 6.0, "speech_fraction": 0.9, "fake_score": 0.93, "verdict": "LIKELY SYNTHETIC" },
    { "start_seconds": 4.0, "end_seconds": 8.0, "speech_fraction": 0.0, "fake_score": null, "verdict": null }
  ],
  "evidence_sha256": "9f86d0...",
  "gradcam_image_base64": "iVBORw0KGgo...",
  "gradcam_peak_region": "..."
}
```

Render the heatmap: `<img src={"data:image/png;base64," + gradcam_image_base64}>`.

**Errors:** 400 (missing `audio` field, empty filename, unsupported extension, unreadable or too-short audio), 413 (file too large), 500.

---

## POST /forensic-report

Runs the same analysis, writes a PDF + JSON report, logs chain of custody, and returns the PDF.

**Request:** `multipart/form-data`

| Field      | Type   | Required | Notes |
|------------|--------|----------|-------|
| `audio`    | file   | yes      | As above. |
| `case_id`  | string | yes      | Non-blank. |
| `examiner` | string | yes      | Non-blank. |

**200 response:** `application/pdf` attachment. Metadata is in headers:

| Header              | Value |
|---------------------|-------|
| `X-Report-Id`       | Report ID. |
| `X-Verdict`         | `LIKELY GENUINE` / `INCONCLUSIVE` / `LIKELY SYNTHETIC`. |
| `X-Evidence-SHA256` | SHA-256 of the evidence file. |

**Errors:** 400 (as above, or missing `case_id` / `examiner`), 413, 500 (JSON body, not PDF).

---

## Notes for the frontend

- **CORS is not enabled.** A browser app on another origin will be blocked. Either proxy `/predict` through the frontend dev server, or ask the backend to add `flask-cors`.
- Reading custom headers (`X-Verdict` etc.) cross-origin also needs `Access-Control-Expose-Headers` once CORS is added.
- Analysis can take several seconds on long files. Show a loading state and set a generous request timeout.
- `INCONCLUSIVE` is a legitimate outcome, not an error. If the model has no validated error rates, the server returns only `INCONCLUSIVE` and scores are for information only. Show the verdict, not just the score.
- Show a disclaimer on `LIKELY SYNTHETIC` for compressed audio. The report itself warns this must not be relied on without expert review.

## Quick test

```bash
curl http://localhost:5000/health
curl -F "audio=@sample.wav" http://localhost:5000/predict
curl -F "audio=@sample.wav" -F "case_id=C-001" -F "examiner=Jane" \
     -o report.pdf -D - http://localhost:5000/forensic-report
```
