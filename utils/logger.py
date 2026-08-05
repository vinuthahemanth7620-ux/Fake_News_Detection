"""
utils/logger.py
===============
Responsibility:
- Centralized logging configuration for app-wide event tracking and debugging.
"""

import logging
import sys


def get_logger(name: str = "fake_news_detection") -> logging.Logger:
    """
    Configures and returns a Logger instance with standard formatting.

    Args:
        name (str): Name of the logger instance. Defaults to "fake_news_detection".

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

