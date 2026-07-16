"""Early stopping callback for the training loop."""

from __future__ import annotations

from utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Attributes:
        patience: Number of consecutive non-improving epochs tolerated
            before signaling that training should stop.
        min_delta: Minimum change in the monitored metric to qualify as an improvement.
        mode: ``"max"`` if higher metric values are better (e.g. F1),
            ``"min"`` if lower is better (e.g. loss).
    """

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max") -> None:
        """Initialize the early stopping tracker.

        Args:
            patience: Consecutive non-improving epochs tolerated before stopping.
            min_delta: Minimum improvement magnitude to reset the patience counter.
            mode: ``"max"`` or ``"min"``, matching the monitored metric's direction.

        Raises:
            ValueError: If ``mode`` is not ``"max"`` or ``"min"``.
        """
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best_value: float | None = None
        self._num_bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        """Check whether ``value`` improves on the best value seen so far.

        Args:
            value: The latest monitored metric value.

        Returns:
            ``True`` if this is the first value seen, or if it improves on
            the best recorded value by more than ``min_delta``.
        """
        if self._best_value is None:
            return True
        if self.mode == "max":
            return value > self._best_value + self.min_delta
        return value < self._best_value - self.min_delta

    def step(self, value: float) -> bool:
        """Record one epoch's monitored metric value.

        Args:
            value: The latest monitored metric value.

        Returns:
            ``True`` if training should stop now (patience exhausted),
            ``False`` otherwise.
        """
        if self._is_improvement(value):
            logger.debug("Early stopping: metric improved to %.6f.", value)
            self._best_value = value
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            logger.debug(
                "Early stopping: no improvement (%d/%d).", self._num_bad_epochs, self.patience
            )

        should_stop = self._num_bad_epochs >= self.patience
        if should_stop:
            logger.info("Early stopping triggered after %d epochs without improvement.", self.patience)
        return should_stop

    @property
    def best_value(self) -> float | None:
        """Best monitored metric value recorded so far, or ``None`` if none recorded."""
        return self._best_value
