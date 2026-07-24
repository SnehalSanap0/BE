# Understanding mel-spectrograms, MFCC, LFCC

## The problem: a raw waveform is a bad shape for a CNN

A raw audio waveform is just one long list of amplitude numbers over time — a 4-second clip at 16kHz is 64,000 numbers in a single row. A CNN designed for images expects a 2-D grid, and more importantly, a raw waveform hides *frequency* information in a form that's very hard for a network to learn directly (a CNN's small local filters see only a tiny time window at a time, with no direct sense of pitch/timbre). Converting to a time-frequency representation first makes the information a CNN is good at exploiting — local 2-D patterns — actually visible.

## Spectrogram — the basic idea

Chop the waveform into short overlapping time windows (a few milliseconds each), and for each window, work out how much energy is present at each frequency (via a Fourier transform). Stack these side by side and you get a 2-D grid: **time along one axis, frequency along the other, and brightness/color representing loudness at that time-frequency point.** That grid is a spectrogram — literally a picture of the sound.

## Mel-spectrogram — a spectrogram warped to match human hearing

Human hearing doesn't perceive frequency linearly — we're far more sensitive to small pitch differences in low frequencies than in high ones. The **mel scale** is a frequency axis that's been re-warped to match that perceptual sensitivity (more resolution where the ear cares more, less where it doesn't). A mel-spectrogram is just a regular spectrogram with its frequency axis remapped this way, and its loudness values usually put on a log scale (matching how loudness is perceived logarithmically, not linearly, too) — `configs/config.yaml`'s `log_scale: true` and `n_mels: 128` control this. This project uses mel-spectrograms as its default feature (`features.type: mel_spectrogram`).

## MFCC — a further-compressed, decorrelated version

**MFCC (Mel-Frequency Cepstral Coefficients)** takes a mel-spectrogram one step further: it applies another transform (a discrete cosine transform) that compresses each time-frame's frequency information down to a small number of coefficients (here, `n_mfcc: 40`) that are largely independent of each other and tend to capture the broad *shape* of the frequency spectrum (closely tied to things like vocal-tract shape) rather than fine-grained detail. Historically the standard feature for speech/speaker recognition before deep learning became dominant, because it's compact and works well with simpler statistical models.

## LFCC — the same idea, without the perceptual warping

**LFCC (Linear-Frequency Cepstral Coefficients)** is computed the same way as MFCC, but on a *linear* frequency scale instead of the mel scale — i.e., it skips the "match human hearing" step. This matters specifically for a *spoof-detection* task (as opposed to normal speech recognition): mel warping is designed around what's perceptually important to a *human listener*, but the acoustic artifacts left behind by a speech synthesizer/vocoder aren't necessarily where human perception is most sensitive — they can hide in frequency regions the mel scale compresses away. LFCC preserves those regions instead, which several anti-spoofing papers report catching synthesis artifacts mel-based features miss. This project supports it (`features.type: lfcc`) as a swappable alternative, even though the current default and documented results use mel-spectrograms.

## Why resize everything to 224×224

EfficientNet-B0 (like most standard CNN architectures) expects a fixed input size, and the raw spectrogram's natural dimensions (frequency bins × time frames) won't naturally be square or match that size. `features.image_size: 224` resizes every feature array to a consistent 224×224 grid so every clip — regardless of exact frame count — produces a uniformly-shaped input the network can batch and process identically.

## Rational summary

- **"Why not just feed the raw waveform into the model?"** — a CNN's local filters need spatially-meaningful 2-D structure to exploit; a spectrogram makes both time and frequency directly visible side by side, which a raw 1-D waveform does not.
- **"Why mel-spectrogram instead of MFCC/LFCC as the default?"** — mel-spectrograms keep more raw detail (they aren't compressed down to a handful of coefficients), which suits a deep CNN that can learn its own compression, whereas MFCC/LFCC were designed for eras where simpler models needed the compression done for them by hand. Mel-spectrograms are also the more common choice in recent deep-learning audio classification literature.
- **"Isn't losing perceptual information (LFCC skipping the mel warp) a downside?"** — not for this task specifically: what matters here is catching *synthesis artifacts*, not matching human perceptual priorities, so LFCC's linear scale is a deliberate, task-appropriate trade-off rather than a lesser version of MFCC.
