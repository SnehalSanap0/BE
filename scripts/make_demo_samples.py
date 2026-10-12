"""Build a demo/test pack of forensic scenarios with known ground truth, and optionally run them.

Every sample is assembled from SEA-Spoof *validation* clips that the deployed (paper) model never saw in
training, so the expected answers are meaningful. The pack covers the cases a forensic examiner would meet:

    genuine_call              four genuine clips joined            -> must NOT be called synthetic
    synthetic_message         four synthetic clips joined          -> LIKELY SYNTHETIC
    spliced_call              genuine, genuine, SYNTHETIC, genuine -> LIKELY SYNTHETIC, splice located
    genuine_stereo            genuine call as stereo               -> same as genuine_call (downmix check)
    synthetic_noisy           synthetic message + noise (10 dB)    -> LIKELY SYNTHETIC (KNOWN LIMITATION)
    genuine_phone_band        genuine, 300-3400 Hz / 8 kHz         -> must NOT be called synthetic
    genuine_mp3_128           genuine, MP3 128 kbps                -> must NOT be called synthetic
    genuine_whatsapp          genuine, Opus 16 kbps    (KNOWN LIMITATION: false alarms)
    synthetic_mp3_32          synthetic, MP3 32 kbps   (KNOWN LIMITATION: fakes slip through)
    genuine_silence_padded    genuine clip with 20 s of silence each side -> silence is skipped

Usage:
    python -m scripts.make_demo_samples            # build samples/forensic_demo/ + manifest.json
    python -m scripts.make_demo_samples --run      # also analyse each and write outputs/forensic_scenarios/

``samples/`` is git-ignored: the audio derives from a gated dataset and must not be redistributed.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from configs.config import CONFIG, ROOT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

CLIP_ROOT = ROOT_DIR / "data" / "smoke_seaspoof_paired" / "raw" / "hindi"
SAMPLE_DIR = ROOT_DIR / "samples" / "forensic_demo"
RESULT_PATH = CONFIG.paths.output_dir / "forensic_scenarios" / "scenario_results.json"
SAMPLE_RATE = CONFIG.audio.sample_rate
GAP_SECONDS = 0.4


def _load_clip(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return audio / (np.abs(audio).max() + 1e-9) * 0.5


def _join(clips: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Concatenate clips with short gaps; return the audio and each clip's (start, end) seconds."""
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    parts, spans, cursor = [], [], 0.0
    for clip in clips:
        spans.append((cursor, cursor + len(clip) / SAMPLE_RATE))
        parts += [clip, gap]
        cursor += len(clip) / SAMPLE_RATE + GAP_SECONDS
    return np.concatenate(parts), spans


def _ffmpeg(source: Path, target: Path, args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source), *args, str(target)], check=True)


def build_samples(seed: int = 7) -> dict:
    rng = random.Random(seed)
    genuine_files = rng.sample(sorted((CLIP_ROOT / "real").glob("*.flac")), 12)
    synthetic_files = rng.sample(sorted((CLIP_ROOT / "fake").glob("*.flac")), 8)
    genuine = [_load_clip(path) for path in genuine_files]
    synthetic = [_load_clip(path) for path in synthetic_files]

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    def save(name: str, audio: np.ndarray, expected: str, description: str, **extra) -> Path:
        path = SAMPLE_DIR / name
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        manifest[name] = {"expected": expected, "description": description, **extra}
        return path

    genuine_call, _ = _join(genuine[0:4])
    synthetic_message, _ = _join(synthetic[0:4])
    spliced, spans = _join([genuine[4], genuine[5], synthetic[4], genuine[6]])
    save("genuine_call.wav", genuine_call, "NOT_SYNTHETIC", "Four genuine clips joined into one call.")
    save("synthetic_message.wav", synthetic_message, "LIKELY SYNTHETIC", "Four synthetic clips joined (fake voice message).")
    save("spliced_call.wav", spliced, "LIKELY SYNTHETIC",
         "Genuine call with one synthetic sentence spliced into the middle.", splice_seconds=list(spans[2]))
    stereo_path = SAMPLE_DIR / "genuine_stereo.wav"
    sf.write(stereo_path, np.stack([genuine_call, genuine_call * 0.8], axis=1), SAMPLE_RATE, subtype="PCM_16")
    manifest["genuine_stereo.wav"] = {"expected": "NOT_SYNTHETIC", "description": "The genuine call as stereo audio."}

    noise = np.random.default_rng(seed).standard_normal(len(synthetic_message)).astype(np.float32)
    signal_power, noise_power = float(np.mean(synthetic_message**2)), float(np.mean(noise**2))
    noise *= np.sqrt(signal_power / (noise_power * 10 ** (10 / 10)))
    save("synthetic_noisy.wav", np.clip(synthetic_message + noise, -1, 1), "LIKELY SYNTHETIC",
         "Synthetic message with white noise added at 10 dB signal-to-noise ratio. Limitation found while "
         "building this pack: heavy noise lowers the score so the verdict becomes INCONCLUSIVE.",
         known_limitation=True)

    padded = np.concatenate([np.zeros(20 * SAMPLE_RATE, np.float32), genuine[7], np.zeros(20 * SAMPLE_RATE, np.float32)])
    save("genuine_silence_padded.wav", padded, "NOT_SYNTHETIC",
         "One genuine clip surrounded by 20 s of silence on each side; silent windows must be skipped.")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_genuine = Path(temp_dir) / "genuine_base.wav"
        base_synthetic = Path(temp_dir) / "synthetic_base.wav"
        sf.write(base_genuine, genuine_call, SAMPLE_RATE)
        sf.write(base_synthetic, synthetic_message, SAMPLE_RATE)
        _ffmpeg(base_genuine, SAMPLE_DIR / "genuine_phone_band.wav",
                ["-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_s16le"])
        manifest["genuine_phone_band.wav"] = {"expected": "NOT_SYNTHETIC",
                                              "description": "Genuine call band-limited to 300-3400 Hz at 8 kHz (telephone)."}
        _ffmpeg(base_genuine, SAMPLE_DIR / "genuine_mp3_128.mp3", ["-c:a", "libmp3lame", "-b:a", "128k"])
        manifest["genuine_mp3_128.mp3"] = {"expected": "NOT_SYNTHETIC", "description": "Genuine call as MP3 128 kbps."}
        _ffmpeg(base_genuine, SAMPLE_DIR / "genuine_whatsapp.opus", ["-c:a", "libopus", "-b:a", "16k"])
        manifest["genuine_whatsapp.opus"] = {
            "expected": "NOT_SYNTHETIC", "known_limitation": True,
            "description": "Genuine call as Opus 16 kbps (WhatsApp-like). Known limitation: the model flags it as synthetic."}
        _ffmpeg(base_synthetic, SAMPLE_DIR / "synthetic_mp3_32.mp3", ["-c:a", "libmp3lame", "-b:a", "32k"])
        manifest["synthetic_mp3_32.mp3"] = {
            "expected": "LIKELY SYNTHETIC", "known_limitation": True,
            "description": "Synthetic message as MP3 32 kbps. Known limitation: heavy compression hides the fake."}

    (SAMPLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %d samples to %s", len(manifest), SAMPLE_DIR)
    return manifest


def run_scenarios(manifest: dict) -> None:
    from deployment.inference_service import InferenceService
    from forensics.analyze import analyze_evidence
    from forensics.custody_log import CustodyLog
    from forensics.verdict import LIKELY_SYNTHETIC

    service = InferenceService()
    rows = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for name, info in manifest.items():
            analysis = analyze_evidence(
                SAMPLE_DIR / name, case_id=f"SCENARIO-{Path(name).stem}", examiner="Automated test", service=service,
                model_label="EfficientNet-B0", report_dir=RESULT_PATH.parent / "reports",
                custody_log=CustodyLog(Path(temp_dir) / "custody.jsonl"),
            )
            findings = analysis.findings
            scored = findings.segment_analysis.scored
            suspicious = findings.segment_analysis.most_suspicious
            splice = info.get("splice_seconds")
            located = None
            if splice:
                located = suspicious.start_seconds < splice[1] and suspicious.end_seconds > splice[0]
            if name == "spliced_call.wav":
                import matplotlib.pyplot as plt
                from forensics.report import render_timeline_image
                plt.imsave(RESULT_PATH.parent / "spliced_call_timeline.png", render_timeline_image(findings))
                (RESULT_PATH.parent / "spliced_call_report.pdf").write_bytes(analysis.pdf_path.read_bytes())
            expected = info["expected"]
            met = (findings.verdict == LIKELY_SYNTHETIC) if expected == LIKELY_SYNTHETIC \
                else (findings.verdict != LIKELY_SYNTHETIC)
            if splice:
                met = met and bool(located)
            rows.append({
                "sample": name, "description": info["description"], "expected": expected,
                "verdict": findings.verdict, "fake_score": findings.fake_score,
                "windows_scored": len(scored),
                "windows_skipped": len(findings.segment_analysis.segments) - len(scored),
                "windows_flagged_synthetic": sum(s.verdict == LIKELY_SYNTHETIC for s in scored),
                "splice_seconds": splice, "splice_located": located,
                "expectation_met": met, "known_limitation": bool(info.get("known_limitation")),
            })
            logger.info("%-26s -> %-16s score %.3f  expectation %s%s", name, findings.verdict, findings.fake_score,
                        "met" if met else "NOT met", "  (known limitation)" if info.get("known_limitation") else "")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({"checkpoint": str(service.checkpoint_path), "rows": rows}, indent=2),
                           encoding="utf-8")
    logger.info("Wrote %s", RESULT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="store_true", help="Also analyse every sample and save the results.")
    args = parser.parse_args()
    manifest = build_samples()
    if args.run:
        run_scenarios(manifest)


if __name__ == "__main__":
    main()
