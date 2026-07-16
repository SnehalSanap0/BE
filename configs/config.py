"""Typed configuration loader for the deepfake-audio-detector project.

Design rationale:
    Every other module imports the singleton :data:`CONFIG` from this file
    instead of reading ``configs/config.yaml`` directly or hardcoding values.
    This keeps configuration a single injectable object (Dependency
    Inversion) — tests and scripts can construct an alternate ``Config`` from
    a different YAML file (e.g. ``config.test.yaml``) without touching any
    downstream code.

Path handling:
    Every path in ``config.yaml`` is relative to the project root. This
    module resolves them to absolute :class:`pathlib.Path` objects exactly
    once, at load time, so no other module ever concatenates path strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Load variables from a .env file at the project root (e.g. HF_TOKEN) into
# the process environment before anything reads os.environ. Silently a
# no-op if no .env file exists, which keeps this safe in CI/production
# environments that inject real environment variables directly.
load_dotenv(ROOT_DIR / ".env")
DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used throughout the pipeline, resolved to absolute paths."""

    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    feature_dir: Path
    metadata_dir: Path
    metadata_file: Path
    log_dir: Path
    checkpoint_dir: Path
    output_dir: Path
    explanation_dir: Path

    def create_directories(self) -> None:
        """Create every directory this config points to, if missing.

        Returns:
            None.
        """
        directories = (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.feature_dir,
            self.metadata_dir,
            self.log_dir,
            self.checkpoint_dir,
            self.output_dir,
            self.explanation_dir,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AudioConfig:
    """Signal-level parameters governing raw-audio preprocessing."""

    sample_rate: int
    mono: bool
    target_duration_seconds: int
    remove_silence: bool
    top_db: int
    normalize: bool
    pad_mode: str
    supported_formats: tuple[str, ...]

    @property
    def target_length_samples(self) -> int:
        """Number of samples a processed clip must contain.

        Returns:
            ``sample_rate * target_duration_seconds`` as an integer sample count.
        """
        return self.sample_rate * self.target_duration_seconds


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset composition and split parameters."""

    languages: tuple[str, ...]
    labels: tuple[str, ...]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int


@dataclass(frozen=True)
class HFSourceConfig:
    """A single HuggingFace source dataset to stream one label's clips from."""

    dataset: str
    config_name: str
    split: str
    label: str
    audio_field: str


@dataclass(frozen=True)
class IngestionConfig:
    """Bounded streaming-ingestion parameters for pulling raw clips from HuggingFace.

    This exists so the project never needs to download the full ~1TB of
    upstream IndicSUPERB/IndicSynth data — it streams at most
    ``samples_per_class`` examples per label straight into
    ``data/raw/<language>/<label>/``.
    """

    language: str
    samples_per_class: int
    hf_token_env_var: str
    real: HFSourceConfig
    fake: HFSourceConfig


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-extraction parameters shared by mel-spectrogram, MFCC and LFCC."""

    type: str
    n_mels: int
    n_mfcc: int
    n_lfcc: int
    n_fft: int
    hop_length: int
    win_length: int
    fmin: int
    fmax: int
    power: float
    log_scale: bool
    image_size: int


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for the EfficientNet(+BiLSTM) classifier."""

    backbone: str
    pretrained: bool
    num_classes: int
    use_bilstm: bool
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_bidirectional: bool
    dropout: float
    freeze_backbone: bool


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and training-loop hyperparameters."""

    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    mixed_precision: bool
    gradient_clip_norm: float
    num_workers: int
    checkpoint_metric: str
    resume: bool


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation metric selection and decision threshold."""

    metrics: tuple[str, ...]
    threshold: float


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Grad-CAM explanation parameters."""

    method: str
    target_layer: str
    overlay_alpha: float
    colormap: str


@dataclass(frozen=True)
class DeploymentConfig:
    """Flask inference-service parameters."""

    host: str
    port: int
    debug: bool
    max_upload_size_mb: int
    allowed_extensions: tuple[str, ...]
    model_checkpoint: Path


@dataclass(frozen=True)
class Config:
    """Root configuration object aggregating every subsystem's settings."""

    paths: PathsConfig
    audio: AudioConfig
    dataset: DatasetConfig
    ingestion: IngestionConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    explainability: ExplainabilityConfig
    deployment: DeploymentConfig
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _resolve_path(root: Path, relative: str) -> Path:
    """Resolve a config-file path string against the project root.

    Args:
        root: Absolute project root directory.
        relative: Path string as written in the YAML file, relative to ``root``.

    Returns:
        The absolute, resolved :class:`pathlib.Path`.
    """
    return (root / relative).resolve()


def load_config(config_path: Path | None = None) -> Config:
    """Load and validate the project configuration from a YAML file.

    Args:
        config_path: Path to a YAML config file. Defaults to
            ``configs/config.yaml`` alongside this module.

    Returns:
        A fully populated, immutable :class:`Config` instance.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        KeyError: If a required section or key is missing from the YAML file.
    """
    resolved_path = config_path or DEFAULT_CONFIG_PATH

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file_handle:
        raw: dict[str, Any] = yaml.safe_load(file_handle)

    paths_raw = raw["paths"]
    paths = PathsConfig(
        data_dir=_resolve_path(ROOT_DIR, paths_raw["data_dir"]),
        raw_dir=_resolve_path(ROOT_DIR, paths_raw["raw_dir"]),
        processed_dir=_resolve_path(ROOT_DIR, paths_raw["processed_dir"]),
        feature_dir=_resolve_path(ROOT_DIR, paths_raw["feature_dir"]),
        metadata_dir=_resolve_path(ROOT_DIR, paths_raw["metadata_dir"]),
        metadata_file=_resolve_path(ROOT_DIR, paths_raw["metadata_file"]),
        log_dir=_resolve_path(ROOT_DIR, paths_raw["log_dir"]),
        checkpoint_dir=_resolve_path(ROOT_DIR, paths_raw["checkpoint_dir"]),
        output_dir=_resolve_path(ROOT_DIR, paths_raw["output_dir"]),
        explanation_dir=_resolve_path(ROOT_DIR, paths_raw["explanation_dir"]),
    )
    paths.create_directories()

    audio_raw = raw["audio"]
    audio = AudioConfig(
        sample_rate=audio_raw["sample_rate"],
        mono=audio_raw["mono"],
        target_duration_seconds=audio_raw["target_duration_seconds"],
        remove_silence=audio_raw["remove_silence"],
        top_db=audio_raw["top_db"],
        normalize=audio_raw["normalize"],
        pad_mode=audio_raw["pad_mode"],
        supported_formats=tuple(audio_raw["supported_formats"]),
    )

    dataset_raw = raw["dataset"]
    dataset = DatasetConfig(
        languages=tuple(dataset_raw["languages"]),
        labels=tuple(dataset_raw["labels"]),
        train_ratio=dataset_raw["train_ratio"],
        val_ratio=dataset_raw["val_ratio"],
        test_ratio=dataset_raw["test_ratio"],
        random_seed=dataset_raw["random_seed"],
    )

    ingestion_raw = raw["ingestion"]

    def _hf_source(source_raw: dict[str, Any]) -> HFSourceConfig:
        return HFSourceConfig(
            dataset=source_raw["dataset"],
            config_name=source_raw["config_name"],
            split=source_raw["split"],
            label=source_raw["label"],
            audio_field=source_raw["audio_field"],
        )

    ingestion = IngestionConfig(
        language=ingestion_raw["language"],
        samples_per_class=ingestion_raw["samples_per_class"],
        hf_token_env_var=ingestion_raw["hf_token_env_var"],
        real=_hf_source(ingestion_raw["real"]),
        fake=_hf_source(ingestion_raw["fake"]),
    )

    features_raw = raw["features"]
    features = FeatureConfig(
        type=features_raw["type"],
        n_mels=features_raw["n_mels"],
        n_mfcc=features_raw["n_mfcc"],
        n_lfcc=features_raw["n_lfcc"],
        n_fft=features_raw["n_fft"],
        hop_length=features_raw["hop_length"],
        win_length=features_raw["win_length"],
        fmin=features_raw["fmin"],
        fmax=features_raw["fmax"],
        power=features_raw["power"],
        log_scale=features_raw["log_scale"],
        image_size=features_raw["image_size"],
    )

    model_raw = raw["model"]
    model = ModelConfig(
        backbone=model_raw["backbone"],
        pretrained=model_raw["pretrained"],
        num_classes=model_raw["num_classes"],
        use_bilstm=model_raw["use_bilstm"],
        lstm_hidden_size=model_raw["lstm_hidden_size"],
        lstm_num_layers=model_raw["lstm_num_layers"],
        lstm_bidirectional=model_raw["lstm_bidirectional"],
        dropout=model_raw["dropout"],
        freeze_backbone=model_raw["freeze_backbone"],
    )

    training_raw = raw["training"]
    training = TrainingConfig(
        batch_size=training_raw["batch_size"],
        num_epochs=training_raw["num_epochs"],
        learning_rate=training_raw["learning_rate"],
        weight_decay=training_raw["weight_decay"],
        optimizer=training_raw["optimizer"],
        scheduler=training_raw["scheduler"],
        early_stopping_patience=training_raw["early_stopping_patience"],
        early_stopping_min_delta=training_raw["early_stopping_min_delta"],
        mixed_precision=training_raw["mixed_precision"],
        gradient_clip_norm=training_raw["gradient_clip_norm"],
        num_workers=training_raw["num_workers"],
        checkpoint_metric=training_raw["checkpoint_metric"],
        resume=training_raw["resume"],
    )

    evaluation_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        metrics=tuple(evaluation_raw["metrics"]),
        threshold=evaluation_raw["threshold"],
    )

    explainability_raw = raw["explainability"]
    explainability = ExplainabilityConfig(
        method=explainability_raw["method"],
        target_layer=explainability_raw["target_layer"],
        overlay_alpha=explainability_raw["overlay_alpha"],
        colormap=explainability_raw["colormap"],
    )

    deployment_raw = raw["deployment"]
    deployment = DeploymentConfig(
        host=deployment_raw["host"],
        port=deployment_raw["port"],
        debug=deployment_raw["debug"],
        max_upload_size_mb=deployment_raw["max_upload_size_mb"],
        allowed_extensions=tuple(deployment_raw["allowed_extensions"]),
        model_checkpoint=_resolve_path(ROOT_DIR, deployment_raw["model_checkpoint"]),
    )

    return Config(
        paths=paths,
        audio=audio,
        dataset=dataset,
        ingestion=ingestion,
        features=features,
        model=model,
        training=training,
        evaluation=evaluation,
        explainability=explainability,
        deployment=deployment,
        raw=raw,
    )


# Process-wide singleton. Import this from every other module:
#     from configs.config import CONFIG
CONFIG: Config = load_config()
