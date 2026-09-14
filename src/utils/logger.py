"""
Centralized logging configuration for MedAI system.
"""

import os
import logging
import logging.handlers
from pathlib import Path


def setup_logger(name: str = __name__, log_file: str = "logs/medai.log") -> logging.Logger:
    """
    Configure and return a logger with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Path to log file
    
    Returns:
        Configured Logger instance
    """
    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation (10MB max, keep 5 backups)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except PermissionError:
        logger.warning(f"Cannot write to log file: {log_file}")

    return logger
