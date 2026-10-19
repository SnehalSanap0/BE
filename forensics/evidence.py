"""Evidence identification: cryptographic hash and technical metadata of an audio file.

The SHA-256 digest is the evidence's identity throughout a forensic analysis:
it goes into the report and the custody log, and is recomputed after analysis
to show the file was not modified while being examined.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from preprocessing.audio_loader import FFMPEG_ENCODING, read_audio_info

_HASH_CHUNK_BYTES = 1024 * 1024

#: libsndfile subtypes that are lossy codecs.
_LOSSY_ENCODINGS = frozenset({"MPEG_LAYER_I", "MPEG_LAYER_II", "MPEG_LAYER_III", "OPUS", "VORBIS"})


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, streamed in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidenceInfo:
    """Identity and technical description of one evidence audio file.

    Attributes:
        filename: Original file name (no directory).
        size_bytes: File size in bytes.
        duration_seconds: Audio duration.
        sample_rate: Native sample rate in Hz.
        channels: Number of audio channels.
        container_format: Container reported by libsndfile, e.g. ``"WAV"``.
        encoding: Sample encoding reported by libsndfile, e.g. ``"PCM_16"``.
        sha256: Optional SHA-256 hex digest.
    """

    filename: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    container_format: str
    encoding: str
    sha256: str = ""

    @property
    def is_lossy(self) -> bool:
        """``True`` for lossy codecs (MP3, Opus, Vorbis, and every ffmpeg-decoded format: AAC, AMR, ...)."""
        return self.encoding in _LOSSY_ENCODINGS or self.encoding == FFMPEG_ENCODING

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of every field."""
        data = asdict(self)
        data["is_lossy"] = self.is_lossy
        if not self.sha256:
            data.pop("sha256", None)
        return data


def describe_evidence(path: Path, display_name: str | None = None, compute_hash: bool = True) -> EvidenceInfo:
    """Read technical metadata of an audio file without decoding samples.

    Args:
        path: Audio file on disk.
        display_name: Name to record instead of ``path.name`` -- e.g. the
            original upload name when ``path`` is a temporary copy.
        compute_hash: Whether to compute SHA-256 digest (default: True).

    Returns:
        An :class:`EvidenceInfo` for the file.

    Raises:
        ValueError: If the file cannot be read as audio.
    """
    path = Path(path)
    info = read_audio_info(path)

    return EvidenceInfo(
        filename=display_name or path.name,
        size_bytes=path.stat().st_size,
        duration_seconds=info.duration_seconds,
        sample_rate=info.sample_rate,
        channels=info.channels,
        container_format=info.container_format,
        encoding=info.encoding,
        sha256=sha256_file(path) if compute_hash else "",
    )
