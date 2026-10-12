"""Pure, single-purpose audio transformation functions.

Each function performs exactly one transformation and has no side effects:
given the same input it always returns the same output, and it never reads
or writes files, metadata, or global state. This is deliberate — transforms
are the unit of reuse between the offline preprocessing pipeline
(:mod:`preprocessing.pipeline`) and the online inference path
(:mod:`deployment.app`), so they must behave identically in both contexts.

Composed together, in order, they implement the target pipeline:

    Raw Audio -> Mono -> 16 kHz -> Trim Silence -> Peak Normalize ->
    Pad/Truncate to fixed duration -> Processed Audio
"""

from __future__ import annotations

import librosa
import numpy as np

from configs.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix multi-channel audio to a single channel by averaging.

    Args:
        audio: Audio samples. Either 1-D ``(num_samples,)`` for mono, or
            2-D multi-channel in either layout: ``(num_samples, num_channels)``
            as returned by ``soundfile.read``, or ``(num_channels, num_samples)``
            as returned by ``librosa.load(mono=False)``. The channel axis is
            taken to be the shorter one.

    Returns:
        1-D mono audio array. Returned unchanged if already mono.
    """
    if audio.ndim == 1:
        return audio

    channel_axis = 0 if audio.shape[0] < audio.shape[1] else 1
    mono_audio = np.mean(audio, axis=channel_axis)
    logger.debug("Converted %d-channel audio to mono.", audio.shape[channel_axis])
    return mono_audio


def resample_audio(audio: np.ndarray, original_sample_rate: int, target_sample_rate: int | None = None) -> np.ndarray:
    """Resample audio to a target sample rate.

    Args:
        audio: 1-D mono audio array.
        original_sample_rate: Sample rate the audio was recorded at, in Hz.
        target_sample_rate: Desired output sample rate, in Hz. Defaults to
            ``CONFIG.audio.sample_rate``.

    Returns:
        Audio resampled to ``target_sample_rate``. Returned unchanged if the
        rates already match.
    """
    target_sample_rate = target_sample_rate or CONFIG.audio.sample_rate

    if original_sample_rate == target_sample_rate:
        return audio

    resampled = librosa.resample(audio, orig_sr=original_sample_rate, target_sr=target_sample_rate)
    logger.debug("Resampled audio from %d Hz to %d Hz.", original_sample_rate, target_sample_rate)
    return resampled


def trim_silence(audio: np.ndarray, top_db: int | None = None) -> np.ndarray:
    """Remove leading and trailing silence from an audio signal.

    Args:
        audio: 1-D mono audio array.
        top_db: Silence threshold, in decibels below peak. Defaults to
            ``CONFIG.audio.top_db``.

    Returns:
        Audio with leading/trailing silence removed. If trimming would
        remove the entire signal (e.g. near-silent input), the original
        audio is returned unchanged to avoid producing an empty array.
    """
    top_db = top_db if top_db is not None else CONFIG.audio.top_db

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)

    if trimmed.size == 0:
        logger.warning("Trimming silence would empty the signal; keeping original audio.")
        return audio

    logger.debug("Trimmed silence: %d -> %d samples.", audio.size, trimmed.size)
    return trimmed


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """Scale audio so its maximum absolute amplitude is 1.0.

    Args:
        audio: 1-D mono audio array.

    Returns:
        Peak-normalized audio. Returned unchanged if the signal is silent
        (all zeros), to avoid a division by zero.
    """
    peak_amplitude = np.max(np.abs(audio))

    if peak_amplitude == 0:
        logger.warning("Cannot peak-normalize a silent (all-zero) signal; returning unchanged.")
        return audio

    return audio / peak_amplitude


def adjust_length(audio: np.ndarray, target_length: int | None = None) -> np.ndarray:
    """Pad or truncate audio to an exact number of samples.

    Args:
        audio: 1-D mono audio array.
        target_length: Desired output length, in samples. Defaults to
            ``CONFIG.audio.target_length_samples`` (sample_rate * duration).

    Returns:
        Audio truncated to ``target_length`` if longer, zero-padded on the
        right if shorter, or returned unchanged if already exact.
    """
    target_length = target_length or CONFIG.audio.target_length_samples

    current_length = audio.shape[0]

    if current_length == target_length:
        return audio

    if current_length > target_length:
        return audio[:target_length]

    pad_width = target_length - current_length
    padded = np.pad(audio, (0, pad_width), mode=CONFIG.audio.pad_mode)
    logger.debug("Padded audio from %d to %d samples.", current_length, target_length)
    return padded


def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the full preprocessing chain to a raw audio signal.

    Composes, in order: mono conversion, resampling, silence trimming,
    peak normalization, and length adjustment. This is the single function
    both the offline pipeline and the online inference path should call to
    guarantee identical train/serve preprocessing behavior.

    Args:
        audio: Raw audio samples, mono or multi-channel.
        sample_rate: Sample rate the audio was loaded at, in Hz.

    Returns:
        Fully processed, mono, resampled, trimmed, normalized, fixed-length
        audio array ready to be persisted or fed into feature extraction.
    """
    audio = convert_to_mono(audio)
    audio = resample_audio(audio, sample_rate)
    audio = trim_silence(audio)
    audio = peak_normalize(audio)
    audio = adjust_length(audio)
    return audio
