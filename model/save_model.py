"""
model/save_model.py
===================
Responsibility:
- Utility functions for serializing (saving) and deserializing (loading) trained models
  and vectorizers using Joblib.
"""

import os
import sys
from typing import Any
import joblib

# Ensure project root directory is in sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger
from utils.helper import ensure_directory_exists, verify_file_exists

logger = get_logger(__name__)


def save_artifact(obj: Any, file_path: str) -> None:
    """
    Saves a generic Python object to disk using Joblib.

    Args:
        obj (Any): Object to serialize (e.g. model, vectorizer).
        file_path (str): Output destination file path.

    Raises:
        ValueError: If object is None.
        IOError: If serialization fails.
    """
    if obj is None:
        logger.error("Attempted to save a None object artifact.")
        raise ValueError("Object to save cannot be None.")

    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            ensure_directory_exists(dir_name)

        joblib.dump(obj, file_path)
        logger.info(f"Successfully saved artifact to '{file_path}'.")
    except Exception as e:
        logger.error(f"Failed to save artifact to '{file_path}': {e}", exc_info=True)
        raise IOError(f"Failed to save artifact to {file_path}") from e


def load_artifact(file_path: str) -> Any:
    """
    Loads a serialized Python object from disk using Joblib.

    Args:
        file_path (str): Input file path.

    Returns:
        Any: Deserialized object.

    Raises:
        FileNotFoundError: If file does not exist.
        IOError: If deserialization fails.
    """
    if not verify_file_exists(file_path):
        logger.error(f"Artifact file not found: '{file_path}'")
        raise FileNotFoundError(f"Artifact file not found at '{file_path}'")

    try:
        obj = joblib.load(file_path)
        logger.info(f"Successfully loaded artifact from '{file_path}'.")
        return obj
    except Exception as e:
        logger.error(f"Failed to load artifact from '{file_path}': {e}", exc_info=True)
        raise IOError(f"Failed to load artifact from {file_path}") from e


def save_model(model: Any, file_path: str = "model/model.pkl") -> None:
    """
    Saves trained Machine Learning model artifact to disk.

    Args:
        model (Any): Trained classifier model object.
        file_path (str): Target pickle/joblib file path. Defaults to "model/model.pkl".
    """
    logger.info(f"Saving trained model to '{file_path}'...")
    save_artifact(model, file_path)


def save_vectorizer(vectorizer: Any, file_path: str = "model/vectorizer.pkl") -> None:
    """
    Saves fitted TF-IDF vectorizer artifact to disk.

    Args:
        vectorizer (Any): Fitted TF-IDF vectorizer object.
        file_path (str): Target pickle/joblib file path. Defaults to "model/vectorizer.pkl".
    """
    logger.info(f"Saving vectorizer to '{file_path}'...")
    save_artifact(vectorizer, file_path)
