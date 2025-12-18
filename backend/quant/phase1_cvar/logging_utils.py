from __future__ import annotations

import logging
from pathlib import Path

def setup_logger(
    name: str = "quant-platform",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Creates a logger that logs to console and optionally to a file.
    Safe to call multiple times (won't duplicate handlers).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger