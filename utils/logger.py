"""
utils/logger.py
===============
Responsibility:
- Centralized logging configuration for app-wide event tracking and debugging.

TODO:
- Configure Python logging logger instance.
- Define log format and file/console handlers.
"""

import logging

def get_logger(name: str = __name__) -> logging.Logger:
    """Returns a configured logger instance."""
    logger = logging.getLogger(name)
    # TODO: Add handlers and formatting
    return logger
