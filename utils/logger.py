"""Centralized logging factory for the deepfake-audio-detector project.

Every module in this project obtains its logger through :func:`get_logger`
instead of calling ``logging.getLogger`` directly. This guarantees a single,
consistent log format and a single set of handlers (console + rotating file)
across the entire codebase, which is what makes cross-module log correlation
possible when debugging a full preprocessing/training run.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file before rotation
_BACKUP_COUNT = 5

_configured_root = False


def _configure_root_logger(log_dir: Path, level: int) -> None:
    """Attach console and rotating-file handlers to the root logger once.

    Args:
        log_dir: Directory where the rotating log file will be written.
        level: Minimum severity level for the root logger.

    Returns:
        None.
    """
    global _configured_root

    if _configured_root:
        return

    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = RotatingFileHandler(
        filename=log_dir / "deepfake_audio_detector.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configured_root = True


def get_logger(name: str, log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped logger backed by shared console/file handlers.

    Args:
        name: Logical name for the logger, typically ``__name__`` of the
            calling module.
        log_dir: Directory to write rotating log files to. Required on the
            first call in a process; ignored on subsequent calls since the
            root logger is only configured once. Defaults to ``None``, which
            falls back to ``configs.config.LOG_DIR`` when unset.
        level: Minimum severity level this logger will emit.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    if log_dir is None:
        from configs.config import CONFIG

        log_dir = CONFIG.paths.log_dir

    _configure_root_logger(log_dir=log_dir, level=level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger
