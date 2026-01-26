"""
Logging utilities for DexGraspNet2.

Provides standardized logging configuration for training and inference.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging for DexGraspNet2.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional file path for log output.
        format_string: Custom format string.

    Returns:
        Root logger instance.
    """
    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger for dexgraspnet2
    logger = logging.getLogger("dexgraspnet2")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (usually __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"dexgraspnet2.{name}")
