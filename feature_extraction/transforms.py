"""Pure, single-purpose feature-extraction functions.

Each function turns a fixed-length, preprocessed mono waveform into a 2-D
time-frequency representation. As in :mod:`preprocessing.transforms`, every
function here is pure (no I/O, no global state) so the exact same code path
can be used offline (:mod:`feature_extraction.pipeline`) and online
(:mod:`deployment.app`) without train/serve skew.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.fftpack import dct

from configs.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_power_spectrogram(
    audio: np.ndarray,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    power: float | None = None,
) -> np.ndarray:
    """Compute a power (or magnitude) spectrogram via the short-time Fourier transform.

    Args:
        audio: 1-D mono waveform.
        n_fft: FFT window size. Defaults to ``CONFIG.features.n_fft``.
        hop_length: Number of samples between successive frames. Defaults to
            ``CONFIG.features.hop_length``.
        win_length: Window length. Defaults to ``CONFIG.features.win_length``.
        power: Exponent for the magnitude spectrogram (``1.0`` for magnitude,
            ``2.0`` for power). Defaults to ``CONFIG.features.power``.

    Returns:
        2-D array of shape ``(1 + n_fft / 2, num_frames)``.
    """
    n_fft = n_fft or CONFIG.features.n_fft
    hop_length = hop_length or CONFIG.features.hop_length
    win_length = win_length or CONFIG.features.win_length
    power = power if power is not None else CONFIG.features.power

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return np.abs(stft) ** power


def extract_mel_spectrogram(audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
    """Compute a (optionally log-scaled) mel spectrogram.

    Args:
        audio: 1-D mono waveform.
        sample_rate: Sample rate of ``audio``, in Hz. Defaults to
            ``CONFIG.audio.sample_rate``.

    Returns:
        2-D array of shape ``(n_mels, num_frames)``.
    """
    sample_rate = sample_rate or CONFIG.audio.sample_rate
    cfg = CONFIG.features

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=cfg.power,
    )

    if cfg.log_scale:
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    logger.debug("Extracted mel spectrogram of shape %s.", mel_spectrogram.shape)
    return mel_spectrogram.astype(np.float32)


def extract_mfcc(audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
    """Compute Mel-Frequency Cepstral Coefficients.

    Args:
        audio: 1-D mono waveform.
        sample_rate: Sample rate of ``audio``, in Hz. Defaults to
            ``CONFIG.audio.sample_rate``.

    Returns:
        2-D array of shape ``(n_mfcc, num_frames)``.
    """
    sample_rate = sample_rate or CONFIG.audio.sample_rate
    cfg = CONFIG.features

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )

    logger.debug("Extracted MFCC of shape %s.", mfcc.shape)
    return mfcc.astype(np.float32)


def _linear_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Build a bank of triangular filters spaced linearly in frequency (Hz).

    This is the linear-frequency analogue of a mel filterbank: mel spacing
    concentrates resolution at low frequencies, while a linear spacing gives
    uniform resolution across the whole band — a property LFCCs rely on to
    capture the higher-frequency spectral artifacts common in synthetic speech.

    Args:
        n_filters: Number of triangular filters to construct.
        n_fft: FFT size the filterbank will be applied to.
        sample_rate: Sample rate audio was analyzed at, in Hz.
        fmin: Lowest frequency covered by the filterbank, in Hz.
        fmax: Highest frequency covered by the filterbank, in Hz.

    Returns:
        2-D array of shape ``(n_filters, 1 + n_fft / 2)``.
    """
    num_fft_bins = 1 + n_fft // 2
    fft_frequencies = np.linspace(0, sample_rate / 2, num_fft_bins)

    filter_edges = np.linspace(fmin, fmax, n_filters + 2)
    filterbank = np.zeros((n_filters, num_fft_bins), dtype=np.float32)

    for filter_index in range(n_filters):
        left_edge, center, right_edge = filter_edges[filter_index : filter_index + 3]

        rising_slope = (fft_frequencies - left_edge) / (center - left_edge + 1e-10)
        falling_slope = (right_edge - fft_frequencies) / (right_edge - center + 1e-10)

        filterbank[filter_index] = np.maximum(0.0, np.minimum(rising_slope, falling_slope))

    return filterbank


def extract_lfcc(audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
    """Compute Linear-Frequency Cepstral Coefficients.

    Applies a linearly-spaced triangular filterbank (as opposed to MFCC's
    mel-spaced one) to the power spectrogram, takes the log, then a DCT-II
    to decorrelate the filter energies into cepstral coefficients.

    Args:
        audio: 1-D mono waveform.
        sample_rate: Sample rate of ``audio``, in Hz. Defaults to
            ``CONFIG.audio.sample_rate``.

    Returns:
        2-D array of shape ``(n_lfcc, num_frames)``.
    """
    sample_rate = sample_rate or CONFIG.audio.sample_rate
    cfg = CONFIG.features

    power_spectrogram = compute_power_spectrogram(
        audio,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        power=cfg.power,
    )

    filterbank = _linear_filterbank(
        n_filters=cfg.n_lfcc + 2,
        n_fft=cfg.n_fft,
        sample_rate=sample_rate,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )

    filtered_energies = filterbank @ power_spectrogram
    log_energies = np.log(filtered_energies + 1e-10)

    lfcc = dct(log_energies, type=2, axis=0, norm="ortho")[: cfg.n_lfcc, :]

    logger.debug("Extracted LFCC of shape %s.", lfcc.shape)
    return lfcc.astype(np.float32)


#: Dispatch table mapping a feature-type name (as configured in config.yaml)
#: to the function that computes it.
FEATURE_EXTRACTORS = {
    "mel_spectrogram": extract_mel_spectrogram,
    "mfcc": extract_mfcc,
    "lfcc": extract_lfcc,
}
