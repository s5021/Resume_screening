"""
Logging configuration
"""
import logging
import sys
from pathlib import Path
from src.config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT
def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    level = getattr(logging, LOG_LEVEL)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
