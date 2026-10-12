"""Decode audio files of any common format into a float32 waveform.

libsndfile (via soundfile) natively reads WAV, FLAC, MP3 and OGG Vorbis/Opus
-- which covers WhatsApp/Telegram voice notes. Formats it cannot read
(M4A/AAC, AMR call recordings, 3GP, WebM, WMA) are decoded with ffmpeg when
it is on PATH.

ffmpeg is run defensively because evidence files are untrusted input: the
demuxer is pinned from the file extension (so a playlist or concat script
renamed to ``.m4a`` is not interpreted as one), only the ``file`` protocol is
allowed (no network or other-file reads), stdin is closed, and decoding has
a timeout.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.logger import get_logger

logger = get_logger(__name__)

#: Extensions decoded via ffmpeg, mapped to the ffmpeg demuxer they must be parsed as.
FFMPEG_DEMUXERS: dict[str, str] = {
    ".m4a": "mov",
    ".mp4": "mov",
    ".3gp": "mov",
    ".aac": "aac",
    ".amr": "amr",
    ".webm": "matroska",
    ".mka": "matroska",
    ".wma": "asf",
}

_FFMPEG_TIMEOUT_SECONDS = 120

#: Encoding label recorded for files decoded via ffmpeg (all such formats here are lossy).
FFMPEG_ENCODING = "decoded via ffmpeg"


@dataclass(frozen=True)
class AudioInfo:
    """Technical description of an audio file.

    Attributes:
        duration_seconds: Audio duration.
        sample_rate: Native sample rate in Hz.
        channels: Number of channels.
        container_format: Container name, e.g. ``"WAV"`` or ``"M4A"``.
        encoding: Sample encoding, e.g. ``"PCM_16"``, or ``"decoded via ffmpeg"``.
    """

    duration_seconds: float
    sample_rate: int
    channels: int
    container_format: str
    encoding: str


def ffmpeg_available() -> bool:
    """Return ``True`` if an ``ffmpeg`` executable is on PATH."""
    return shutil.which("ffmpeg") is not None


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode a non-libsndfile format to float32 via a hardened ffmpeg call.

    Raises:
        ValueError: If the extension is not allowed, ffmpeg is missing, or decoding fails.
    """
    demuxer = FFMPEG_DEMUXERS.get(path.suffix.lower())
    if demuxer is None:
        raise ValueError(f"Unsupported audio format '{path.suffix}'.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(f"Decoding '{path.suffix}' files requires ffmpeg on PATH (https://ffmpeg.org).")

    with tempfile.TemporaryDirectory() as temp_dir:
        decoded_path = Path(temp_dir) / "decoded.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-protocol_whitelist", "file",
            "-f", demuxer,
            "-i", str(path.resolve()),
            "-vn",
            "-acodec", "pcm_f32le",
            str(decoded_path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, timeout=_FFMPEG_TIMEOUT_SECONDS, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Decoding '{path.name}' timed out.") from exc
        if completed.returncode != 0 or not decoded_path.exists():
            message = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
            raise ValueError(f"Could not decode '{path.name}': {message[-1] if message else 'ffmpeg failed'}")
        audio, sample_rate = sf.read(str(decoded_path), dtype="float32", always_2d=False)
    return audio, int(sample_rate)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode an audio file to float32 samples at its native sample rate.

    Args:
        path: Audio file on disk.

    Returns:
        ``(audio, sample_rate)``; ``audio`` is 1-D for mono, ``(frames, channels)`` otherwise.

    Raises:
        ValueError: If the file cannot be decoded.
    """
    path = Path(path)
    if path.suffix.lower() not in FFMPEG_DEMUXERS:
        try:
            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
            return audio, int(sample_rate)
        except Exception as exc:  # noqa: BLE001 - any libsndfile failure means "unreadable"
            raise ValueError(f"Unreadable or corrupted audio: {exc}") from exc
    return _decode_with_ffmpeg(path)


def read_audio_info(path: Path) -> AudioInfo:
    """Describe an audio file, from its header when libsndfile can read it, else by decoding.

    Args:
        path: Audio file on disk.

    Returns:
        An :class:`AudioInfo`.

    Raises:
        ValueError: If the file cannot be read.
    """
    path = Path(path)
    if path.suffix.lower() not in FFMPEG_DEMUXERS:
        try:
            info = sf.info(str(path))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Cannot read audio metadata from '{path.name}': {exc}") from exc
        return AudioInfo(
            duration_seconds=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            container_format=str(info.format),
            encoding=str(info.subtype),
        )

    audio, sample_rate = _decode_with_ffmpeg(path)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    return AudioInfo(
        duration_seconds=audio.shape[0] / sample_rate,
        sample_rate=sample_rate,
        channels=channels,
        container_format=path.suffix.lstrip(".").upper(),
        encoding=FFMPEG_ENCODING,
    )
