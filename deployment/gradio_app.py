"""Gradio demo: upload or record a clip, get back a prediction and a Grad-CAM overlay.

Sits alongside :mod:`deployment.app` (the Flask JSON API) rather than
replacing it -- both wrap :class:`deployment.inference_service.InferenceService`,
so predictions are guaranteed identical between the two. This module exists
for a interactive, linkable demo (e.g. hosted on Hugging Face Spaces) that
needs no HTTP client to try, which the Flask API alone doesn't provide.

Real and fake clips are shown side by side (not one at a time): the layout
mirrors real-vs-fake comparison, since the point of a Grad-CAM demo is
contrasting *where* the model looks on a genuine clip versus a synthetic one
sharing the same content, not just explaining one clip in isolation.

Two models, not one: this project has separate checkpoints per language
(``checkpoints/best_model.pt`` for Hindi, ``checkpoints/english_smoke/best_model.pt``
for English), trained under different configs (different ``features.image_size``
in particular -- 224 vs 128). A checkpoint must be fed features shaped exactly
as it was trained on, so this module keeps one :class:`InferenceService` per
language and a language selector routes each "Analyze" to the right one --
running an English clip through the Hindi model (or vice versa) silently
produces meaningless predictions instead of an error, which is why this
matters and isn't just a config nicety.
"""

from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr
import numpy as np

from dataset.deepfake_dataset import LABEL_TO_INDEX
from deployment.inference_service import InferenceService
from explainability.visualizer import render_colormap_legend
from utils.logger import get_logger

logger = get_logger(__name__)

_OTHER_LABEL = {"real": "fake", "fake": "real"}
_ROOT_DIR = Path(__file__).resolve().parents[1]

#: Transcript-paired raw data roots to pull one-click example pairs from,
#: tagged with which model should serve that language. Each root must contain
#: real/ and fake/ subdirectories plus the pair_transcripts.csv sidecar
#: written by paired ingestion (see preprocessing.hf_ingestion.ingest_paired_sources
#: / ingest_wavefake_paired) -- unpaired sources (e.g. data/raw/hindi, whose
#: real and fake clips share no transcript link) are deliberately excluded,
#: since an unmatched real/fake pair would compare unrelated content instead
#: of the same utterance's real vs. synthesized version.
_PAIRED_SOURCE_ROOTS: tuple[tuple[str, str, str], ...] = (
    ("data/smoke_seaspoof_paired/raw/hindi", "Hindi (SEA-Spoof)", "Hindi"),
    ("data/english_smoke/raw/english", "English (WaveFake)", "English"),
)

#: Per-language model config: checkpoint path (relative to project root) and
#: the image_size that checkpoint was trained with (see module docstring).
#:
#: NOTE: "Hindi" deliberately points at the smoke_seaspoof_paired checkpoint,
#: not checkpoints/best_model.pt. best_model.pt is the project's original
#: pooled multi-source model (Kathbath/Common Voice/IndicSynth/SEA-Spoof,
#: pooled random split -- see docs/research_log.md's 2026-07-23 "suspiciously
#: perfect" entry) with a documented asymmetric bias toward missing spoofed
#: clips on sources it wasn't trained on. It was never trained on any
#: transcript-paired data, so testing it against paired example clips (from
#: smoke_seaspoof_paired or paper_seaspoof_paired) compares it to data it has
#: no relationship to -- an apples-to-oranges mismatch, not evidence of a new
#: bug, but misleading for a demo. smoke_seaspoof_paired/best_model.pt is
#: trained on exactly the data its matching examples come from.
_LANGUAGE_MODELS: dict[str, dict[str, object]] = {
    "Hindi": {"checkpoint": "checkpoints/smoke_seaspoof_paired/best_model.pt", "image_size": 128},
    "English": {"checkpoint": "checkpoints/english_smoke/best_model.pt", "image_size": 128},
}

_MAX_EXAMPLE_PAIRS_PER_SOURCE = 2


def _load_paired_examples(language_root: Path, limit: int) -> list[tuple[Path, Path, str]]:
    """Read a language root's pair-transcript sidecar and resolve matched real/fake pairs.

    Args:
        language_root: Directory containing ``real/``, ``fake/`` subdirectories
            and a ``pair_transcripts.csv`` sidecar (see
            ``preprocessing.hf_ingestion._append_pair_transcripts``).
        limit: Maximum number of pairs to return.

    Returns:
        ``(real_path, fake_path, transcript)`` tuples for pairs where both
        files still exist on disk, in sidecar encounter order.
    """
    sidecar_path = language_root / "pair_transcripts.csv"
    if not sidecar_path.exists():
        return []

    real_dir = language_root / "real"
    fake_dir = language_root / "fake"

    by_transcript: dict[str, list[str]] = {}
    with sidecar_path.open(newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            by_transcript.setdefault(row["transcript"], []).append(row["filename"])

    pairs: list[tuple[Path, Path, str]] = []
    for transcript, filenames in by_transcript.items():
        real_match = next((real_dir / name for name in filenames if (real_dir / name).exists()), None)
        fake_match = next((fake_dir / name for name in filenames if (fake_dir / name).exists()), None)
        if real_match is not None and fake_match is not None:
            pairs.append((real_match, fake_match, transcript))
        if len(pairs) >= limit:
            break
    return pairs


def _existing_example_pairs() -> list[list[str]]:
    """Build one-click example rows (real path, fake path, language) for every available paired source."""
    rows: list[list[str]] = []
    for relative_root, source_label, language in _PAIRED_SOURCE_ROOTS:
        language_root = _ROOT_DIR / relative_root
        pairs = _load_paired_examples(language_root, limit=_MAX_EXAMPLE_PAIRS_PER_SOURCE)
        if not pairs:
            logger.debug("No paired examples found under %s (%s).", language_root, source_label)
            continue
        for real_path, fake_path, transcript in pairs:
            logger.debug("Example pair from %s: transcript=%s", source_label, transcript)
            rows.append([str(real_path), str(fake_path), language])
    return rows


def _build_language_services() -> dict[str, InferenceService]:
    """Load one InferenceService per language whose checkpoint actually exists on disk."""
    services: dict[str, InferenceService] = {}
    for language, model_config in _LANGUAGE_MODELS.items():
        checkpoint_path = _ROOT_DIR / model_config["checkpoint"]
        if not checkpoint_path.exists():
            logger.warning("Skipping '%s' model: no checkpoint at %s.", language, checkpoint_path)
            continue
        services[language] = InferenceService(
            checkpoint_path=checkpoint_path,
            feature_image_size=model_config["image_size"],
        )
        logger.info("Loaded '%s' model from %s.", language, checkpoint_path)
    return services


def create_demo(services: dict[str, InferenceService] | None = None) -> gr.Blocks:
    """Construct the Gradio demo, loading each language's model exactly once.

    Args:
        services: Pre-constructed ``{language: InferenceService}`` mapping to
            use instead of loading fresh ones. Primarily for tests, which can
            inject stubs. Defaults to :func:`_build_language_services`.

    Returns:
        A configured, unlaunched :class:`gradio.Blocks` app.

    Raises:
        RuntimeError: If no language has an available checkpoint.
    """
    services = services if services is not None else _build_language_services()
    if not services:
        raise RuntimeError(
            "No trained checkpoints found for any language. Expected at least one of: "
            + ", ".join(str(cfg["checkpoint"]) for cfg in _LANGUAGE_MODELS.values())
        )
    language_choices = list(services)
    legend_image = render_colormap_legend()

    def analyze_clip(language: str, audio_path: str | None) -> tuple:
        """Run one clip through the selected language's model. Blanks (not an error) if no clip is loaded."""
        if audio_path is None:
            return None, None, None, None, ""
        service = services.get(language)
        if service is None:
            raise gr.Error(f"No trained model available for '{language}'.")

        try:
            result = service.predict(Path(audio_path))
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        other_label = _OTHER_LABEL[result.predicted_label]
        label_probs = {
            result.predicted_label: result.confidence,
            other_label: 1.0 - result.confidence,
        }
        caption = (
            f"Most influential region: **{result.peak_region_caption}** — drove the "
            f"'{result.predicted_label}' prediction (confidence {result.confidence:.1%})."
        )
        processed_audio_output = (result.processed_sample_rate, result.processed_audio)
        return label_probs, processed_audio_output, result.spectrogram_image, result.overlay_image, caption

    def analyze_both(
        language: str, real_audio_path: str | None, fake_audio_path: str | None
    ) -> tuple:
        real_label, real_processed, real_spec, real_overlay, real_caption = analyze_clip(language, real_audio_path)
        fake_label, fake_processed, fake_spec, fake_overlay, fake_caption = analyze_clip(language, fake_audio_path)
        return (
            real_label, real_processed, real_spec, real_overlay, real_caption,
            fake_label, fake_processed, fake_spec, fake_overlay, fake_caption,
        )

    with gr.Blocks(title="Explainable Audio Deepfake Detection") as demo:
        gr.Markdown(
            "# Explainable Audio Deepfake Detection for Bilingual Speech\n"
            "EfficientNet-B0 over mel-spectrograms, with Grad-CAM explanations. "
            "Real and fake clips are shown side by side so you can compare where the model "
            "looks on genuine vs. synthetic speech. Pick a matched example pair below, or "
            "upload/record your own clip and select the matching language model. "
            "See the project README for cross-source generalization caveats."
        )

        language_selector = gr.Radio(
            choices=language_choices,
            value=language_choices[0],
            label="Model / language",
            info="Each language has its own trained checkpoint -- pick the one matching your clip's language.",
        )

        with gr.Row():
            with gr.Column(scale=1):
                real_audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Real clip")
            with gr.Column(scale=1):
                fake_audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Fake clip")

        analyze_button = gr.Button("Analyze both", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                real_processed_output = gr.Audio(label="Processed audio (real clip)", interactive=False)
            with gr.Column(scale=1):
                fake_processed_output = gr.Audio(label="Processed audio (fake clip)", interactive=False)

        gr.Markdown("**Grad-CAM legend**")
        gr.Image(
            value=legend_image,
            show_label=False,
            interactive=False,
            buttons=[],
            container=False,
            height=70,
        )

        with gr.Row():
            with gr.Column(scale=1):
                real_label_output = gr.Label(num_top_classes=len(LABEL_TO_INDEX), label="Prediction (real clip)")
                with gr.Row():
                    real_spectrogram_output = gr.Image(label="Spectrogram (real clip)")
                    real_gradcam_output = gr.Image(label="Grad-CAM (real clip)")
                real_caption_output = gr.Markdown()
            with gr.Column(scale=1):
                fake_label_output = gr.Label(num_top_classes=len(LABEL_TO_INDEX), label="Prediction (fake clip)")
                with gr.Row():
                    fake_spectrogram_output = gr.Image(label="Spectrogram (fake clip)")
                    fake_gradcam_output = gr.Image(label="Grad-CAM (fake clip)")
                fake_caption_output = gr.Markdown()

        example_pairs = [row for row in _existing_example_pairs() if row[2] in services]
        outputs = [
            real_label_output, real_processed_output, real_spectrogram_output, real_gradcam_output, real_caption_output,
            fake_label_output, fake_processed_output, fake_spectrogram_output, fake_gradcam_output, fake_caption_output,
        ]
        if example_pairs:
            gr.Examples(
                examples=example_pairs,
                inputs=[real_audio_input, fake_audio_input, language_selector],
                outputs=outputs,
                fn=lambda real_path, fake_path, language: analyze_both(language, real_path, fake_path),
                run_on_click=True,
                label="Try a matched real/fake pair (same transcript) -- also sets the matching language model",
            )

        analyze_button.click(
            fn=analyze_both,
            inputs=[language_selector, real_audio_input, fake_audio_input],
            outputs=outputs,
        )

    logger.info("Gradio demo created for languages: %s.", ", ".join(services))
    return demo


if __name__ == "__main__":
    app = create_demo()
    app.launch()
