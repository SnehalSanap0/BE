"""Run the English WaveFake smoke pipeline end to end.

Uses ``configs/config.english_smoke.yaml`` by default, which writes all
artifacts into dedicated ``english_smoke`` subfolders. Data: LJSpeech
(real) + 7 neural-vocoder fakes (WF1-WF7) from the WaveFake benchmark
(Frank & Schonherr, 2021), via the community HF parquet mirror
``ajaykarthick/wavefake-audio``.

Ingestion is transcript-paired (by shared ``audio_id``), like SEA-Spoof:
each real clip is matched with exactly one fake variant (deterministically
rotated across WF1-WF7 so every vocoder is represented), and splits are
assigned by ``assign_grouped_splits`` so a pair's real+fake clips never land
in different partitions -- the same leakage protection the SEA-Spoof paired
runs rely on. See ``preprocessing.hf_ingestion.ingest_wavefake_paired`` and
docs/research_log.md for the design rationale.

Usage:
    python -m scripts.run_english_smoke
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = ROOT_DIR / "configs" / "config.english_smoke.yaml"
os.environ.setdefault("DEEPFAKE_CONFIG_PATH", str(SMOKE_CONFIG_PATH))

import torch  # noqa: E402

from configs.config import CONFIG  # noqa: E402
from dataset.dataloader import build_dataloader  # noqa: E402
from dataset.split_assigner import assign_grouped_splits  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from feature_extraction.pipeline import FeatureExtractionPipeline  # noqa: E402
from models.model_factory import build_model  # noqa: E402
from preprocessing.hf_ingestion import _resolve_hf_token, ingest_wavefake_paired  # noqa: E402
from preprocessing.metadata_builder import build_initial_metadata  # noqa: E402
from preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.metadata_manager import MetadataManager  # noqa: E402

logger = get_logger(__name__)

_FAKE_VOCODERS = tuple(f"WF{i}" for i in range(1, 8))


def _ensure_raw_pairs() -> None:
    """Ingest WaveFake pairs (transcript/utterance-id matched) up to the configured target."""
    ingest_wavefake_paired(
        dataset=CONFIG.ingestion.real[0].dataset,
        split=CONFIG.ingestion.real[0].split,
        id_field="audio_id",
        audio_field=CONFIG.ingestion.real[0].audio_field,
        real_value="R",
        fake_values=_FAKE_VOCODERS,
        target_pairs=CONFIG.ingestion.real[0].target_count,
        language=CONFIG.ingestion.language,
        raw_dir=CONFIG.paths.raw_dir,
        hf_token=_resolve_hf_token(CONFIG.ingestion.hf_token_env_var),
        real_tag=CONFIG.ingestion.real[0].source_tag,
        fake_tag=CONFIG.ingestion.fake[0].source_tag,
    )


def main() -> None:
    """Run ingestion, preprocessing, features, split assignment, training, and test evaluation."""
    logger.info("English smoke config: %s", os.environ["DEEPFAKE_CONFIG_PATH"])
    logger.info("English smoke artifacts root: %s", CONFIG.paths.data_dir)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    _ensure_raw_pairs()

    manager = MetadataManager(CONFIG.paths.metadata_file)
    manager = build_initial_metadata(manager)
    PreprocessingPipeline(metadata_manager=manager).run(bootstrap_if_empty=False)

    manager = MetadataManager(CONFIG.paths.metadata_file)
    FeatureExtractionPipeline(metadata_manager=manager).run()

    manager = MetadataManager(CONFIG.paths.metadata_file)
    manager = assign_grouped_splits(manager, group_column="transcript", force_reassign=True)

    train_loader = build_dataloader("train", metadata_manager=manager)
    val_loader = build_dataloader("val", metadata_manager=manager)
    test_loader = build_dataloader("test", metadata_manager=manager)

    model = build_model()
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader)
    trainer.fit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_state = trainer.checkpoint_manager.load(CONFIG.paths.checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    evaluator = Evaluator(model, test_loader, device)
    metrics = evaluator.evaluate()

    CONFIG.paths.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = CONFIG.paths.output_dir / "smoke_test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    print(json.dumps(metrics, indent=2))
    logger.info("Saved English smoke metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
