"""Improved model: EfficientNet-B0 spatial features -> BiLSTM -> Dense -> logits.

Architecture rationale:
    The baseline classifier (:class:`models.baseline_classifier.EfficientNetClassifier`)
    global-average-pools EfficientNet's spatial feature map into a single
    vector, discarding *where* within the spectrogram (i.e. *when*, since
    one spatial axis of a spectrogram is time) any given pattern occurred.
    This model instead keeps EfficientNet's spatial feature map
    ``(batch, channels, height, width)``, treats the width axis (time) as a
    sequence, and feeds the per-timestep channel/height-flattened feature
    vectors through a bidirectional LSTM. This lets the model learn temporal
    dynamics — e.g. a synthesis artifact that only appears in short bursts,
    or an unnatural transition between phonemes — that pure spatial pooling
    cannot represent.
"""

from __future__ import annotations

import torch
from torch import nn

from configs.config import CONFIG
from models.backbone import build_efficientnet_backbone
from utils.logger import get_logger

logger = get_logger(__name__)


class EfficientNetBiLSTM(nn.Module):
    """EfficientNet-B0 spatial features fed through a BiLSTM temporal head.

    Attributes:
        backbone: The pretrained EfficientNet-B0 feature extractor (used via
            ``forward_features`` to keep spatial resolution).
        lstm: Bidirectional LSTM consuming the per-timestep flattened
            channel/height feature vectors.
        classifier: Dropout + linear layer mapping pooled LSTM outputs to class logits.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        lstm_hidden_size: int | None = None,
        lstm_num_layers: int | None = None,
        lstm_bidirectional: bool | None = None,
        dropout: float | None = None,
        pretrained: bool | None = None,
        freeze_backbone: bool | None = None,
        image_size: int | None = None,
    ) -> None:
        """Initialize the EfficientNet+BiLSTM model.

        Args:
            num_classes: Number of output classes. Defaults to ``CONFIG.model.num_classes``.
            lstm_hidden_size: Hidden size per LSTM direction. Defaults to
                ``CONFIG.model.lstm_hidden_size``.
            lstm_num_layers: Number of stacked LSTM layers. Defaults to
                ``CONFIG.model.lstm_num_layers``.
            lstm_bidirectional: Whether the LSTM processes the sequence in
                both directions. Defaults to ``CONFIG.model.lstm_bidirectional``.
            dropout: Dropout probability before the final linear layer, and
                between stacked LSTM layers. Defaults to ``CONFIG.model.dropout``.
            pretrained: Whether to load ImageNet-pretrained backbone weights.
                Defaults to ``CONFIG.model.pretrained``.
            freeze_backbone: Whether to freeze backbone weights. Defaults to
                ``CONFIG.model.freeze_backbone``.
            image_size: Spatial size of the input feature maps, used only to
                probe the backbone's output shape at construction time.
                Defaults to ``CONFIG.features.image_size``.
        """
        super().__init__()

        num_classes = num_classes or CONFIG.model.num_classes
        lstm_hidden_size = lstm_hidden_size or CONFIG.model.lstm_hidden_size
        lstm_num_layers = lstm_num_layers or CONFIG.model.lstm_num_layers
        lstm_bidirectional = (
            CONFIG.model.lstm_bidirectional if lstm_bidirectional is None else lstm_bidirectional
        )
        dropout = dropout if dropout is not None else CONFIG.model.dropout
        image_size = image_size or CONFIG.features.image_size

        self.backbone = build_efficientnet_backbone(pretrained=pretrained, freeze=freeze_backbone)

        lstm_input_size = self._probe_lstm_input_size(image_size)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        lstm_output_size = lstm_hidden_size * (2 if lstm_bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_size, num_classes),
        )

        logger.info(
            "Constructed EfficientNetBiLSTM: lstm_input=%d, hidden=%d, layers=%d, "
            "bidirectional=%s, output classes=%d.",
            lstm_input_size,
            lstm_hidden_size,
            lstm_num_layers,
            lstm_bidirectional,
            num_classes,
        )

    def _probe_lstm_input_size(self, image_size: int) -> int:
        """Determine the LSTM's expected input size from the backbone's actual output shape.

        Runs a single dummy forward pass instead of hardcoding EfficientNet-B0's
        known ``(1280, 7, 7)`` output, so this model keeps working if the
        backbone or ``image_size`` configuration changes.

        Args:
            image_size: Spatial size of a dummy square input.

        Returns:
            Flattened per-timestep feature size: ``channels * height``.
        """
        was_training = self.backbone.training
        self.backbone.eval()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, image_size, image_size)
            spatial_features = self.backbone.forward_features(dummy_input)

        self.backbone.train(was_training)

        _, channels, height, _ = spatial_features.shape
        return channels * height

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a batch of feature maps through the backbone, BiLSTM and classifier head.

        Args:
            features: Batch of single-channel feature maps, shape
                ``(batch, 1, image_size, image_size)``.

        Returns:
            Unnormalized class logits, shape ``(batch, num_classes)``.
        """
        spatial_features = self.backbone.forward_features(features)  # (B, C, H, W)

        batch_size, channels, height, width = spatial_features.shape
        # Treat the width axis (time, since spectrogram width = time frames)
        # as the sequence dimension; flatten channel+height into the
        # per-timestep feature vector.
        sequence = spatial_features.permute(0, 3, 1, 2).reshape(batch_size, width, channels * height)

        lstm_output, _ = self.lstm(sequence)  # (B, width, lstm_output_size)
        pooled_output = lstm_output.mean(dim=1)  # temporal mean pooling

        return self.classifier(pooled_output)
