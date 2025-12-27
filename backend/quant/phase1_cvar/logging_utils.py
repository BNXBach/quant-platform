from __future__ import annotations

import logging
from pathlib import Path
import sys

def setup_logger(
    name: str = "quant-platform",
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Creates a logger that logs to console and optionally to a file.
    Safe to call multiple times (won't duplicate handlers).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger