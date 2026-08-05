"""
presenter/presenter.py
======================
Responsibility:
- Main Presenter module/class orchestrating request validation, text preprocessing invocation, model inference, and output formatting.
- Ensures View logic never communicates directly with Model layer.
"""

import os
import sys
from typing import Dict, Any, Tuple

# Ensure project root directory is in sys.path when executed directly
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from utils.logger import get_logger
from presenter.validation import validate_news_input
from model.predictor import predict_news as model_predict_news

logger = get_logger(__name__)


def validate_input(text: str) -> Tuple[bool, str]:
    """
    Validates article text input.

    Args:
        text (str): Raw news article text input.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    logger.info("Validating news article input text...")
    return validate_news_input(text)


def process_prediction(text: str) -> Dict[str, Any]:
    """
    Delegates text prediction to the Model layer predictor module.

    Args:
        text (str): Validated news article text.

    Returns:
        Dict[str, Any]: Raw prediction result from Model layer.
    """
    logger.info("Delegating article prediction to Model predictor module...")
    return model_predict_news(text)


def format_prediction_result(prediction_data: Dict[str, Any], original_text: str = "") -> Dict[str, Any]:
    """
    Formats raw prediction result into a structured payload for View rendering.

    Args:
        prediction_data (Dict[str, Any]): Model prediction output dictionary.
        original_text (str): Original news article text submitted by user.

    Returns:
        Dict[str, Any]: Formatted result dictionary for View layer.
    """
    logger.info("Formatting prediction result for View layer rendering...")
    prediction_label = prediction_data.get("prediction", "UNKNOWN")
    confidence = prediction_data.get("confidence", 0.0)
    label_code = prediction_data.get("label", 0)

    is_real = (prediction_label == "REAL")
    badge_class = "success" if is_real else "danger"
    status_text = "Real News" if is_real else "Fake News"

    excerpt = original_text.strip()
    if len(excerpt) > 300:
        excerpt = excerpt[:297] + "..."

    return {
        "prediction": prediction_label,
        "status_text": status_text,
        "label": label_code,
        "confidence": confidence,
        "badge_class": badge_class,
        "is_real": is_real,
        "excerpt": excerpt,
        "original_text": original_text
    }


def predict_news(text: str) -> Dict[str, Any]:
    """
    Main presenter function coordinating validation, model inference, and output formatting.

    Args:
        text (str): Input news article text.

    Returns:
        Dict[str, Any]: Response dictionary with 'success' status, 'error' message, and 'result'.
    """
    is_valid, error_msg = validate_input(text)
    if not is_valid:
        logger.warning(f"Input validation failed: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "result": None
        }

    try:
        raw_result = process_prediction(text)
        formatted_result = format_prediction_result(raw_result, original_text=text)
        return {
            "success": True,
            "error": None,
            "result": formatted_result
        }
    except Exception as e:
        logger.error(f"Error during prediction processing: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Prediction error: {e}",
            "result": None
        }


class NewsPresenter:
    """Presenter class handling request orchestration following MVP pattern."""

    def __init__(self, predictor=None, validator=None) -> None:
        """Initializes NewsPresenter instance with optional custom predictor/validator dependencies."""
        self.predictor = predictor
        self.validator = validator

    def validate_input(self, text: str) -> Tuple[bool, str]:
        """Validates article input text."""
        if self.validator:
            return self.validator(text)
        return validate_input(text)

    def process_prediction(self, text: str) -> Dict[str, Any]:
        """Processes prediction via predictor instance or module function."""
        if self.predictor and hasattr(self.predictor, "predict"):
            return self.predictor.predict(text)
        return process_prediction(text)

    def format_prediction_result(self, prediction_data: Dict[str, Any], original_text: str = "") -> Dict[str, Any]:
        """Formats prediction result for View."""
        return format_prediction_result(prediction_data, original_text=original_text)

    def handle_prediction(self, article_text: str) -> Dict[str, Any]:
        """
        Validates input, requests prediction from Model, and formats output for View.

        Args:
            article_text (str): Input news article text.

        Returns:
            Dict[str, Any]: Structured prediction output payload.
        """
        return predict_news(article_text)
