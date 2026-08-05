"""
model/train_model.py
====================
Responsibility:
- Loads cleaned dataset (dataset/cleaned_news.csv).
- Validates presence of 'text' and 'label' columns.
- Prepares feature vector (X = text) and label vector (y = label).
- Converts text into numerical TF-IDF features using TfidfVectorizer.
- Splits dataset into train and test sets with stratification.
- Trains a Logistic Regression classification model.
- Evaluates model performance (Accuracy, Precision, Recall, F1-Score,
  Confusion Matrix, Classification Report).
- Saves model.pkl and vectorizer.pkl artifacts.
"""

import os
import sys
from typing import Tuple, Dict, Any

# Ensure project root directory is in sys.path when executed directly
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

import pandas as pd
import sklearn
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils.logger import get_logger
from utils.helper import verify_file_exists
from model.save_model import (
    save_model as save_model_artifact,
    save_vectorizer as save_vectorizer_artifact
)

logger = get_logger(__name__)


def log_environment_versions() -> None:
    """
    Logs the versions of core packages (Python, scikit-learn, joblib, pandas)
    used during model training for reproducibility and debugging.
    """
    logger.info("==========================================")
    logger.info("       ENVIRONMENT & PACKAGE VERSIONS     ")
    logger.info("==========================================")
    logger.info(f"Python version:       {sys.version.split()[0]}")
    logger.info(f"scikit-learn version: {sklearn.__version__}")
    logger.info(f"joblib version:       {joblib.__version__}")
    logger.info(f"pandas version:       {pd.__version__}")
    logger.info("==========================================")



def load_clean_dataset(file_path: str = "dataset/cleaned_news.csv") -> pd.DataFrame:
    """
    Loads and validates the cleaned news dataset.

    Args:
        file_path (str): File path to cleaned_news.csv. Defaults to "dataset/cleaned_news.csv".

    Returns:
        pd.DataFrame: Validated pandas DataFrame containing 'text' and 'label'.

    Raises:
        FileNotFoundError: If dataset file is missing.
        ValueError: If mandatory 'text' or 'label' columns are missing.
    """
    logger.info(f"Loading cleaned dataset from '{file_path}'...")
    if not verify_file_exists(file_path):
        logger.error(f"Dataset file missing at '{file_path}'.")
        raise FileNotFoundError(f"Cleaned dataset not found at '{file_path}'")

    try:
        df = pd.read_csv(file_path)

        required_columns = {'text', 'label'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            msg = f"Dataset is missing required column(s): {missing_columns}"
            logger.error(msg)
            raise ValueError(msg)

        # Ensure text entries are string and handle null values
        df['text'] = df['text'].fillna('').astype(str)

        logger.info(f"Dataset loaded successfully with {len(df)} records.")
        return df

    except Exception as e:
        logger.error(f"Error loading clean dataset from '{file_path}': {e}", exc_info=True)
        raise


def create_tfidf_vectorizer(
    max_features: int = 5000,
    stop_words: str = "english",
    lowercase: bool = True
) -> TfidfVectorizer:
    """
    Instantiates and configures a TF-IDF Vectorizer.

    Args:
        max_features (int): Maximum number of features. Defaults to 5000.
        stop_words (str): Language for stop words. Defaults to 'english'.
        lowercase (bool): Whether to convert text to lowercase. Defaults to True.

    Returns:
        TfidfVectorizer: Configured vectorizer object.
    """
    logger.info(
        f"Creating TfidfVectorizer (max_features={max_features}, "
        f"stop_words='{stop_words}', lowercase={lowercase})..."
    )
    return TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        lowercase=lowercase
    )


def split_dataset(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:
    """
    Splits the feature and target arrays into training and test sets using stratification.

    Args:
        X: Feature data (text strings or sparse matrix).
        y: Label series/array.
        test_size (float): Portion of dataset allocated to testing. Defaults to 0.2.
        random_state (int): Seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[Any, Any, Any, Any]: X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting dataset (test_size={test_size}, random_state={random_state}, stratify=y)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        logger.info(
            f"Dataset split complete: Training samples = {len(X_train)}, "
            f"Testing samples = {len(X_test)}"
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}", exc_info=True)
        raise


def train_model(
    X_train,
    y_train,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Trains a Logistic Regression model on training features and labels.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        max_iter (int): Maximum iterations for solver convergence. Defaults to 1000.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        LogisticRegression: Trained model instance.
    """
    logger.info(
        f"Training Logistic Regression model (max_iter={max_iter}, random_state={random_state})..."
    )
    try:
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        model.fit(X_train, y_train)
        logger.info("Logistic Regression model training completed successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training Logistic Regression model: {e}", exc_info=True)
        raise


def evaluate_model(model: LogisticRegression, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluates trained model on test dataset and logs all standard evaluation metrics.

    Args:
        model (LogisticRegression): Fitted classifier model.
        X_test: Test feature matrix.
        y_test: True test target labels.

    Returns:
        Dict[str, Any]: Dictionary containing all evaluation metrics.
    """
    logger.info("Evaluating model performance on test set...")
    try:
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info("==========================================")
        logger.info("       MODEL EVALUATION RESULTS           ")
        logger.info("==========================================")
        logger.info(f"Accuracy:          {acc:.4f}")
        logger.info(f"Precision:         {prec:.4f}")
        logger.info(f"Recall:            {rec:.4f}")
        logger.info(f"F1-Score:          {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{report}")
        logger.info("==========================================")

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {e}", exc_info=True)
        raise


def save_model(model: Any, file_path: str = "model/model.pkl") -> None:
    """
    Wrapper function to save trained model artifact to disk.

    Args:
        model (Any): Trained classifier model object.
        file_path (str): Target path. Defaults to "model/model.pkl".
    """
    save_model_artifact(model, file_path)


def save_vectorizer(vectorizer: Any, file_path: str = "model/vectorizer.pkl") -> None:
    """
    Wrapper function to save fitted vectorizer artifact to disk.

    Args:
        vectorizer (Any): Fitted TF-IDF vectorizer object.
        file_path (str): Target path. Defaults to "model/vectorizer.pkl".
    """
    save_vectorizer_artifact(vectorizer, file_path)


def run_training_pipeline(
    dataset_path: str = "dataset/cleaned_news.csv",
    model_path: str = "model/model.pkl",
    vectorizer_path: str = "model/vectorizer.pkl"
) -> Tuple[LogisticRegression, TfidfVectorizer, Dict[str, Any]]:
    """
    Orchestrates end-to-end dataset loading, vectorization, dataset splitting,
    model training, model evaluation, and artifact serialization.

    Args:
        dataset_path (str): Input path for cleaned CSV. Defaults to "dataset/cleaned_news.csv".
        model_path (str): Output path for model.pkl. Defaults to "model/model.pkl".
        vectorizer_path (str): Output path for vectorizer.pkl. Defaults to "model/vectorizer.pkl".

    Returns:
        Tuple[LogisticRegression, TfidfVectorizer, Dict[str, Any]]:
            Trained model, fitted vectorizer, and evaluation metrics dictionary.
    """
    logger.info("========== Starting ML Model Training Pipeline ==========")
    log_environment_versions()

    # 1. Load dataset
    df = load_clean_dataset(dataset_path)

    # 2. Extract X (text) and y (label)
    X = df['text']
    y = df['label']

    # 3. Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)

    # 4. Create vectorizer and transform text
    vectorizer = create_tfidf_vectorizer(max_features=5000, stop_words='english', lowercase=True)
    logger.info("Fitting TF-IDF Vectorizer on training text data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    logger.info("Transforming testing text data with TF-IDF Vectorizer...")
    X_test_vec = vectorizer.transform(X_test)

    # 5. Train model
    model = train_model(X_train_vec, y_train, max_iter=1000, random_state=42)

    # 6. Evaluate model
    metrics = evaluate_model(model, X_test_vec, y_test)

    # 7. Save model and vectorizer artifacts
    save_model(model, model_path)
    save_vectorizer(vectorizer, vectorizer_path)

    logger.info("========== ML Model Training Pipeline Completed Successfully ==========")
    return model, vectorizer, metrics


def main() -> None:
    """Entry point for training script execution."""
    try:
        run_training_pipeline()
    except Exception as e:
        logger.critical(f"Training pipeline execution failed: {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
