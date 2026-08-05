"""
utils/helper.py
===============
Responsibility:
- General utility functions (file verification, directory management, text formatting).
"""

import os
from utils.logger import get_logger

logger = get_logger(__name__)


def verify_file_exists(file_path: str) -> bool:
    """
    Verifies whether a file exists at the given path.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if file exists and is a file, False otherwise.
    """
    if os.path.exists(file_path) and os.path.isfile(file_path):
        logger.info(f"File verified successfully: {file_path}")
        return True
    logger.error(f"File verification failed. File not found: {file_path}")
    return False


def ensure_directory_exists(dir_path: str) -> None:
    """
    Ensures that a directory exists, creating it if missing.

    Args:
        dir_path (str): Directory path to verify or create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def truncate_text(text: str, max_len: int = 150) -> str:
    """
    Truncates text to maximum length with trailing ellipsis.

    Args:
        text (str): String to truncate.
        max_len (int): Maximum length allowed. Defaults to 150.

    Returns:
        str: Truncated text string.
    """
    if not isinstance(text, str):
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."

