"""
model/predictor.py
==================
Responsibility:
- Loads serialized trained model (model.pkl) and TF-IDF vectorizer (vectorizer.pkl).
- Accepts raw news article text input.
- Validates and preprocesses text input using preprocessing pipeline.
- Transforms text into TF-IDF numerical features.
- Computes prediction ("REAL" or "FAKE"), class label (1 or 0), and confidence score.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional

# Ensure project root directory is in sys.path when executed directly
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from utils.logger import get_logger
from model.save_model import load_artifact
from model.preprocessing import clean_text

logger = get_logger(__name__)

# Module-level artifact singletons for performance optimization
_LOADED_MODEL: Optional[Any] = None
_LOADED_VECTORIZER: Optional[Any] = None


def load_model(file_path: str = "model/model.pkl") -> Any:
    """
    Loads and returns the trained Machine Learning model from disk.

    Args:
        file_path (str): Path to model pickle/joblib file. Defaults to "model/model.pkl".

    Returns:
        Any: Loaded classifier model.
    """
    global _LOADED_MODEL
    if _LOADED_MODEL is None or file_path != "model/model.pkl":
        logger.info(f"Loading model artifact from '{file_path}'...")
        try:
            _LOADED_MODEL = load_artifact(file_path)
        except Exception as e:
            msg = f"Model file error for '{file_path}': {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
    return _LOADED_MODEL


def load_vectorizer(file_path: str = "model/vectorizer.pkl") -> Any:
    """
    Loads and returns the fitted TF-IDF vectorizer from disk.

    Args:
        file_path (str): Path to vectorizer pickle/joblib file. Defaults to "model/vectorizer.pkl".

    Returns:
        Any: Loaded TF-IDF vectorizer.
    """
    global _LOADED_VECTORIZER
    if _LOADED_VECTORIZER is None or file_path != "model/vectorizer.pkl":
        logger.info(f"Loading vectorizer artifact from '{file_path}'...")
        try:
            _LOADED_VECTORIZER = load_artifact(file_path)
        except Exception as e:
            msg = f"Vectorizer file error for '{file_path}': {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
    return _LOADED_VECTORIZER



def preprocess_input(text: str) -> str:
    """
    Validates input and preprocesses raw text using NLP text cleaning pipeline.

    Args:
        text (str): Raw input news article text.

    Returns:
        str: Preprocessed and cleaned text string.

    Raises:
        TypeError: If text is None or not a string.
        ValueError: If text is empty or contains only whitespace.
    """
    if text is None:
        logger.error("Validation error: Input text is None.")
        raise TypeError("Input text cannot be None.")

    if not isinstance(text, str):
        logger.error(f"Validation error: Input text must be a string, got {type(text).__name__}.")
        raise TypeError(f"Input text must be a string, got {type(text).__name__}.")

    if not text.strip():
        logger.error("Validation error: Input text is empty or contains only whitespace.")
        raise ValueError("Input text cannot be empty or whitespace only.")

    cleaned = clean_text(text)
    return cleaned


def vectorize_text(text: str, vectorizer: Optional[Any] = None) -> Any:
    """
    Preprocesses input text and converts it into TF-IDF numerical feature vector.

    Args:
        text (str): Input news article text.
        vectorizer (Optional[Any]): Fitted TF-IDF vectorizer instance. Defaults to None.

    Returns:
        Any: Transformed TF-IDF feature matrix.
    """
    cleaned_text = preprocess_input(text)
    if vectorizer is None:
        vectorizer = load_vectorizer()

    logger.info("Transforming preprocessed text into TF-IDF feature vector...")
    return vectorizer.transform([cleaned_text])


def get_prediction_confidence(
    text: str,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None
) -> Tuple[int, float]:
    """
    Computes predicted class label (0 or 1) and confidence percentage using predict_proba.

    Args:
        text (str): Input news article text.
        model (Optional[Any]): Trained model instance. Defaults to None.
        vectorizer (Optional[Any]): Fitted vectorizer instance. Defaults to None.

    Returns:
        Tuple[int, float]: Tuple containing (predicted_label, confidence_percentage).
    """
    vectorized_text = vectorize_text(text, vectorizer=vectorizer)

    if model is None:
        model = load_model()

    logger.info("Computing prediction probabilities with model.predict_proba()...")
    try:
        probabilities = model.predict_proba(vectorized_text)[0]
        predicted_label = int(model.predict(vectorized_text)[0])
    except AttributeError as e:
        msg = f"Incompatible model estimator ({e}). Please retrain the model by running 'python -m model.train_model'."
        logger.error(msg)
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Prediction failed due to model error: {e}. Please retrain the model using 'python -m model.train_model'."
        logger.error(msg)
        raise RuntimeError(msg) from e

    confidence_percentage = round(float(probabilities[predicted_label]) * 100.0, 2)
    logger.info(f"Calculated prediction: label={predicted_label}, confidence={confidence_percentage}%")

    return predicted_label, confidence_percentage



def predict_news(
    text: str,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Generates prediction dictionary for input news text.

    Output format:
    {
        "prediction": "REAL" or "FAKE",
        "label": 1 or 0,
        "confidence": 98.76
    }

    Args:
        text (str): Input news article text.
        model (Optional[Any]): Trained model instance. Defaults to None.
        vectorizer (Optional[Any]): Fitted vectorizer instance. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary with prediction, label, and confidence.
    """
    label, confidence = get_prediction_confidence(text, model=model, vectorizer=vectorizer)
    prediction_str = "REAL" if label == 1 else "FAKE"

    result = {
        "prediction": prediction_str,
        "label": label,
        "confidence": confidence
    }
    logger.info(f"Prediction result: {result}")
    return result


class NewsPredictor:
    """Class responsible for model inference."""

    def __init__(
        self,
        model_path: str = "model/model.pkl",
        vectorizer_path: str = "model/vectorizer.pkl"
    ) -> None:
        """
        Initializes NewsPredictor instance with optional model and vectorizer file paths.

        Args:
            model_path (str): Path to trained model pickle file.
            vectorizer_path (str): Path to fitted vectorizer pickle file.
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None

    def load_artifacts(self) -> None:
        """Loads serialized model and vectorizer from disk."""
        logger.info("Loading predictor model and vectorizer artifacts...")
        self.model = load_model(self.model_path)
        self.vectorizer = load_vectorizer(self.vectorizer_path)

    def predict(self, news_text: str) -> Dict[str, Any]:
        """
        Predicts whether a news article is Real or Fake.

        Args:
            news_text (str): Input news article text.

        Returns:
            Dict[str, Any]: Result dictionary containing prediction label and confidence.
        """
        if self.model is None or self.vectorizer is None:
            self.load_artifacts()
        return predict_news(news_text, model=self.model, vectorizer=self.vectorizer)


def main() -> None:
    """
    CLI entry point for user article prediction.
    Asks user to enter a news article from terminal and prints prediction results.
    """
    print("\n--- Fake News Detector CLI ---")
    try:
        user_input = input("Enter a news article text to verify: ")
        result = predict_news(user_input)

        print("\nPrediction:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Label: {result['label']}\n")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print(f"\nError: {e}\n")


if __name__ == '__main__':
    main()
