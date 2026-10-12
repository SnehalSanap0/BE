"""Measure model size and inference speed (GPU and CPU) of the deployed detector.

Reports parameter count, checkpoint size, and per-4-second-window latency split into feature extraction and
the forward pass, plus the extra cost of a Grad-CAM explanation, so the "lightweight enough for a small
machine" claim of the project rests on measurements.

Usage:
    python -m scripts.benchmark_inference --repeats 30

Output: ``outputs/efficiency/inference_benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from configs.config import CONFIG, ROOT_DIR
from deployment.inference_service import InferenceService
from preprocessing.audio_loader import load_audio
from preprocessing.transforms import preprocess_audio
from forensics.segments import to_model_rate_mono
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = CONFIG.paths.output_dir / "efficiency" / "inference_benchmark.json"
SAMPLE_CLIP = ROOT_DIR / "data" / "smoke_seaspoof_paired" / "raw" / "hindi" / "fake" / "fake_sea_000000.flac"


def _time_ms(function, repeats: int, device: torch.device) -> float:
    """Median wall-clock milliseconds of ``function()`` over ``repeats`` calls (after one warm-up)."""
    function()
    durations = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    return float(np.median(durations))


def benchmark(device: torch.device, repeats: int, checkpoint: Path | None) -> dict:
    service = InferenceService(checkpoint_path=checkpoint, device=device)
    audio = preprocess_audio(to_model_rate_mono(*load_audio(SAMPLE_CLIP)), CONFIG.audio.sample_rate)
    feature = service.feature_extractor.extract_array(audio, CONFIG.audio.sample_rate)
    batch = torch.from_numpy(np.stack([feature] * 32)).float().unsqueeze(1).to(device)
    single = batch[:1]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        features_ms = _time_ms(lambda: service.feature_extractor.extract_array(audio, CONFIG.audio.sample_rate),
                               repeats, device)
        forward_ms = _time_ms(lambda: service.model(single), repeats, device)
        batch_ms = _time_ms(lambda: service.model(batch), max(5, repeats // 3), device)
    cam_ms = _time_ms(lambda: service.explainer.generate(single, target_class=1), max(5, repeats // 3), device)
    result = {
        "device": str(device),
        "feature_extraction_ms_per_window": features_ms,
        "forward_ms_per_window": forward_ms,
        "forward_ms_per_32_window_batch": batch_ms,
        "forward_ms_per_window_batched": batch_ms / 32,
        "gradcam_ms_per_window": cam_ms,
        "scoring_ms_per_window_end_to_end": features_ms + batch_ms / 32,
    }
    if device.type == "cuda":
        result["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
    parameters = sum(p.numel() for p in service.model.parameters())
    return result | {"parameters": parameters, "checkpoint_mb": service.checkpoint_path.stat().st_size / 1e6}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    results = [benchmark(device, args.repeats, args.checkpoint) for device in devices]
    payload = {
        "machine": {
            "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads(),
        },
        "input": f"one {CONFIG.audio.target_duration_seconds}-second window, {CONFIG.features.image_size}x"
                 f"{CONFIG.features.image_size} log-mel feature",
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for result in results:
        logger.info("%s: %.1f ms features + %.1f ms forward (single), %.2f ms/window batched, Grad-CAM %.1f ms, "
                    "%.2fM params, %.1f MB checkpoint", result["device"], result["feature_extraction_ms_per_window"],
                    result["forward_ms_per_window"], result["forward_ms_per_window_batched"],
                    result["gradcam_ms_per_window"], result["parameters"] / 1e6, result["checkpoint_mb"])


if __name__ == "__main__":
    main()
