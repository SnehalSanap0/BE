"""Measure how audio compression changes the detector's behaviour on genuine and synthetic speech.

Each held-out clip is re-encoded with ffmpeg (MP3, AAC, Opus at several bitrates) and scored again, so the
effect of the codec alone is isolated: the clips are the same, only the encoding changes. Two error types
are reported per condition:

* ``genuine_flagged`` -- share of genuine clips scored above the "likely synthetic" threshold (false alarm)
* ``fake_caught`` -- share of synthetic clips scored above that threshold (detection)

Usage:
    python -m scripts.codec_robustness_check --clips-per-class 40

Output: ``outputs/codec_robustness/codec_robustness.json``. The default clips are SEA-Spoof *validation*
clips from the paired smoke run: unseen by the paper model (trained on SEA-Spoof ``train``), but from the
same corpus, so results here are about codecs, not domain shift.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from configs.config import CONFIG, ROOT_DIR
from deployment.inference_service import InferenceService
from forensics.segments import to_model_rate_mono
from preprocessing.audio_loader import ffmpeg_available, load_audio
from preprocessing.transforms import preprocess_audio
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CLIP_ROOT = ROOT_DIR / "data" / "smoke_seaspoof_paired" / "raw" / "hindi"
OUTPUT_PATH = CONFIG.paths.output_dir / "codec_robustness" / "codec_robustness.json"

#: name -> (file extension, ffmpeg encoder arguments); ``None`` = the original, unencoded clip.
CONDITIONS: dict[str, tuple[str, list[str]] | None] = {
    "Original (FLAC)": None,
    "WAV 44.1 kHz": (".wav", ["-ar", "44100", "-c:a", "pcm_s16le"]),
    "MP3 192 kbps": (".mp3", ["-ar", "44100", "-c:a", "libmp3lame", "-b:a", "192k"]),
    "MP3 128 kbps": (".mp3", ["-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k"]),
    "MP3 64 kbps": (".mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    "MP3 32 kbps": (".mp3", ["-c:a", "libmp3lame", "-b:a", "32k"]),
    "AAC 64 kbps (M4A)": (".m4a", ["-c:a", "aac", "-b:a", "64k"]),
    "Opus 32 kbps": (".opus", ["-c:a", "libopus", "-b:a", "32k"]),
    "Opus 16 kbps (WhatsApp-like)": (".opus", ["-c:a", "libopus", "-b:a", "16k"]),
    "Phone band (8 kHz, 300-3400 Hz)": (
        ".wav", ["-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_s16le"],
    ),
}


def _encode(source: Path, extension: str, ffmpeg_args: list[str], work_dir: Path) -> Path:
    """Re-encode one clip with ffmpeg and return the new file's path."""
    target = work_dir / f"{source.stem}{extension}"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source), *ffmpeg_args, str(target)],
        check=True,
    )
    return target


def _scores(service: InferenceService, paths: list[Path]) -> np.ndarray:
    """Probability-synthetic of each file, scored as a single 4 s window."""
    windows = [preprocess_audio(to_model_rate_mono(*load_audio(path)), CONFIG.audio.sample_rate) for path in paths]
    return service.fake_scores(windows)


def main() -> None:
    """Run every codec condition over a random sample of genuine and synthetic clips."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT, help="Folder with real/ and fake/ clips.")
    parser.add_argument("--clips-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: deployment.model_checkpoint.")
    args = parser.parse_args()

    if not ffmpeg_available():
        raise SystemExit("ffmpeg is required on PATH for this experiment.")

    rng = random.Random(args.seed)
    genuine = rng.sample(sorted((args.clip_root / "real").glob("*.flac")), args.clips_per_class)
    synthetic = rng.sample(sorted((args.clip_root / "fake").glob("*.flac")), args.clips_per_class)
    service = InferenceService(checkpoint_path=args.checkpoint)
    threshold = CONFIG.forensics.synthetic_above

    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for name, condition in CONDITIONS.items():
            scores = {}
            for label, clips in (("genuine", genuine), ("synthetic", synthetic)):
                if condition is None:
                    paths = clips
                else:
                    extension, ffmpeg_args = condition
                    folder = Path(temp_dir) / f"{name.split()[0]}_{label}"
                    folder.mkdir(exist_ok=True)
                    paths = [_encode(clip, extension, ffmpeg_args, folder) for clip in clips]
                scores[label] = _scores(service, paths)
            result = {
                "condition": name,
                "genuine_flagged": float((scores["genuine"] > threshold).mean()),
                "fake_caught": float((scores["synthetic"] > threshold).mean()),
                "genuine_median_score": float(np.median(scores["genuine"])),
                "fake_median_score": float(np.median(scores["synthetic"])),
            }
            results.append(result)
            logger.info(
                "%-34s genuine flagged %4.0f%%   fake caught %4.0f%%",
                name, result["genuine_flagged"] * 100, result["fake_caught"] * 100,
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(service.checkpoint_path),
        "clip_root": str(args.clip_root),
        "clips_per_class": args.clips_per_class,
        "seed": args.seed,
        "synthetic_threshold": threshold,
        "note": "Same unseen SEA-Spoof validation clips in every condition; only the encoding changes.",
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
