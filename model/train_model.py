"""
model/train_model.py
====================
Responsibility:
- Loads cleaned dataset (dataset/cleaned_news.csv).
- Prepares feature vector (X = text) and label vector (y = label).
- Splits data into train and test sets with stratification BEFORE vectorization (preventing data leakage).
- Evaluates 36 TF-IDF feature engineering configurations using 5-fold Stratified Cross-Validation:
    - max_features: [5000, 10000, 20000]
    - ngram_range: [(1,1), (1,2), (1,3)]
    - min_df: [2, 5]
    - max_df: [0.9, 0.95]
- Uses the best-performing classifier: LinearSVC(C=1.0).
- Selects the feature configuration with the highest mean 5-fold CV F1-score.
- Generates artifact files in artifacts/:
    - artifacts/feature_comparison.csv
    - artifacts/cross_validation_results.csv
    - artifacts/confusion_matrix.png
    - artifacts/model_metrics.json
- Saves best fitted vectorizer to model/vectorizer.pkl and best model to model/model.pkl.
"""

import os
import sys
import json
import itertools
from typing import Tuple, Dict, Any, List

# Ensure project root directory is in sys.path when executed directly
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

import pandas as pd
import numpy as np
import sklearn
import joblib
import matplotlib
matplotlib.use('Agg')  # Headless backend for rendering plots without display
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from utils.logger import get_logger
from utils.helper import verify_file_exists, ensure_directory_exists
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


def split_dataset(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:
    """
    Splits feature and target arrays into training and test sets using stratification.

    Args:
        X: Raw text data series.
        y: Label series.
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


def optimize_feature_engineering(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series
) -> List[Dict[str, Any]]:
    """
    Evaluates 36 TF-IDF parameter configurations using 5-fold Stratified Cross-Validation
    with LinearSVC(C=1.0).

    Args:
        X_train (pd.Series): Training raw text.
        y_train (pd.Series): Training labels.
        X_test (pd.Series): Testing raw text.
        y_test (pd.Series): Testing labels.

    Returns:
        List[Dict[str, Any]]: Evaluation metrics for all 36 feature configurations.
    """
    max_features_list = [5000, 10000, 20000]
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    min_dfs = [2, 5]
    max_dfs = [0.9, 0.95]

    combinations = list(itertools.product(max_features_list, ngram_ranges, min_dfs, max_dfs))
    logger.info(f"========== Starting Feature Engineering Optimization ({len(combinations)} configurations) ==========")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for idx, (max_feat, ngram_range, min_df, max_df) in enumerate(combinations, start=1):
        config_name = f"Config_{idx:02d} (mf={max_feat}, ngram={ngram_range}, min_df={min_df}, max_df={max_df})"
        logger.info(f"Evaluating [{idx}/{len(combinations)}] {config_name}...")

        try:
            # Fit vectorizer ON TRAINING SET ONLY to prevent data leakage
            vectorizer = TfidfVectorizer(
                max_features=max_feat,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                lowercase=True
            )
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            classifier = LinearSVC(C=1.0, random_state=42, max_iter=2000, dual='auto')

            # 5-fold Stratified Cross Validation
            cv_scores = cross_val_score(classifier, X_train_vec, y_train, cv=cv, scoring='f1', n_jobs=2)

            mean_cv_f1 = float(np.mean(cv_scores))
            std_cv_f1 = float(np.std(cv_scores))

            # Train on full X_train_vec and evaluate on X_test_vec
            classifier.fit(X_train_vec, y_train)
            y_pred = classifier.predict(X_test_vec)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            results.append({
                "config_id": f"Config_{idx:02d}",
                "max_features": max_feat,
                "ngram_range": str(ngram_range),
                "min_df": min_df,
                "max_df": max_df,
                "cv_scores": [float(s) for s in cv_scores],
                "mean_cv_f1": mean_cv_f1,
                "std_cv_f1": std_cv_f1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "classification_report": report,
                "vectorizer": vectorizer,
                "model": classifier
            })

            logger.info(
                f"[{config_name}] Mean CV F1: {mean_cv_f1:.4f} ± {std_cv_f1:.4f} | "
                f"Test Acc: {acc:.4f} | Test F1: {f1:.4f}"
            )

        except Exception as e:
            logger.error(f"Error evaluating feature configuration {config_name}: {e}", exc_info=True)
            raise

    return results


def extract_top_features(model: Any, vectorizer: TfidfVectorizer, top_n: int = 15) -> Dict[str, Any]:
    """Extracts top positive (Real) and negative (Fake) weighted features from LinearSVC."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_positive_indices = np.argsort(coefs)[-top_n:][::-1]
    top_negative_indices = np.argsort(coefs)[:top_n]

    return {
        "top_positive_real_features": [
            {"feature": str(feature_names[i]), "weight": round(float(coefs[i]), 4)} for i in top_positive_indices
        ],
        "top_negative_fake_features": [
            {"feature": str(feature_names[i]), "weight": round(float(coefs[i]), 4)} for i in top_negative_indices
        ]
    }


def generate_artifacts(
    feature_results: List[Dict[str, Any]],
    best_result: Dict[str, Any],
    artifacts_dir: str = "artifacts"
) -> None:
    """
    Generates feature comparison CSV, cross validation results CSV, confusion matrix PNG,
    and metrics JSON files in artifacts/.
    """
    logger.info(f"Generating artifact files in '{artifacts_dir}'...")
    ensure_directory_exists(artifacts_dir)

    # 1. Generate feature_comparison.csv
    comp_rows = []
    for res in feature_results:
        comp_rows.append({
            "Config_ID": res["config_id"],
            "max_features": res["max_features"],
            "ngram_range": res["ngram_range"],
            "min_df": res["min_df"],
            "max_df": res["max_df"],
            "CV_Mean_F1": round(res["mean_cv_f1"], 4),
            "CV_Std_F1": round(res["std_cv_f1"], 4),
            "Test_Accuracy": round(res["accuracy"], 4),
            "Test_Precision": round(res["precision"], 4),
            "Test_Recall": round(res["recall"], 4),
            "Test_F1_Score": round(res["f1_score"], 4)
        })
    df_comp = pd.DataFrame(comp_rows)
    df_comp.sort_values(by="CV_Mean_F1", ascending=False, inplace=True)
    comp_path = os.path.join(artifacts_dir, "feature_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    logger.info(f"Successfully saved feature comparison CSV to '{comp_path}'.")

    # 2. Generate cross_validation_results.csv
    cv_rows = []
    for res in feature_results:
        scores = res["cv_scores"]
        cv_rows.append({
            "Config_ID": res["config_id"],
            "max_features": res["max_features"],
            "ngram_range": res["ngram_range"],
            "min_df": res["min_df"],
            "max_df": res["max_df"],
            "Fold_1_F1": round(scores[0], 4),
            "Fold_2_F1": round(scores[1], 4),
            "Fold_3_F1": round(scores[2], 4),
            "Fold_4_F1": round(scores[3], 4),
            "Fold_5_F1": round(scores[4], 4),
            "Mean_CV_F1": round(res["mean_cv_f1"], 4)
        })
    df_cv = pd.DataFrame(cv_rows)
    df_cv.sort_values(by="Mean_CV_F1", ascending=False, inplace=True)
    cv_path = os.path.join(artifacts_dir, "cross_validation_results.csv")
    df_cv.to_csv(cv_path, index=False)
    logger.info(f"Successfully saved cross validation results CSV to '{cv_path}'.")

    # 3. Extract top features for best configuration
    top_features = extract_top_features(best_result["model"], best_result["vectorizer"], top_n=15)

    # 4. Generate model_metrics.json
    metrics_json = {
        "best_classifier": "LinearSVC (C=1.0)",
        "best_feature_configuration": {
            "config_id": best_result["config_id"],
            "max_features": best_result["max_features"],
            "ngram_range": best_result["ngram_range"],
            "min_df": best_result["min_df"],
            "max_df": best_result["max_df"],
            "mean_cv_f1": round(best_result["mean_cv_f1"], 4),
            "std_cv_f1": round(best_result["std_cv_f1"], 4)
        },
        "best_test_metrics": {
            "accuracy": round(best_result["accuracy"], 4),
            "precision": round(best_result["precision"], 4),
            "recall": round(best_result["recall"], 4),
            "f1_score": round(best_result["f1_score"], 4),
            "confusion_matrix": best_result["confusion_matrix"].tolist()
        },
        "top_features": top_features
    }
    json_path = os.path.join(artifacts_dir, "model_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)
    logger.info(f"Successfully saved model metrics JSON to '{json_path}'.")

    # 5. Generate updated confusion_matrix.png
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=best_result["confusion_matrix"],
            display_labels=["FAKE (0)", "REAL (1)"]
        )
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        plt.title(
            f"Confusion Matrix - Best Feature Config ({best_result['config_id']})\n"
            f"LinearSVC | mf={best_result['max_features']} ngram={best_result['ngram_range']}",
            fontsize=10,
            fontweight="bold"
        )
        plt.tight_layout()
        png_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plt.savefig(png_path, dpi=300)
        plt.close()
        logger.info(f"Successfully saved confusion matrix plot to '{png_path}'.")
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix image: {e}", exc_info=True)


def save_model(model: Any, file_path: str = "model/model.pkl") -> None:
    """Wrapper function to save trained model artifact to disk."""
    save_model_artifact(model, file_path)


def save_vectorizer(vectorizer: Any, file_path: str = "model/vectorizer.pkl") -> None:
    """Wrapper function to save fitted vectorizer artifact to disk."""
    save_vectorizer_artifact(vectorizer, file_path)


def run_training_pipeline(
    dataset_path: str = "dataset/cleaned_news.csv",
    model_path: str = "model/model.pkl",
    vectorizer_path: str = "model/vectorizer.pkl"
) -> Tuple[Any, TfidfVectorizer, Dict[str, Any]]:
    """
    Orchestrates end-to-end dataset loading, dataset splitting, feature engineering
    optimization across 36 TF-IDF parameter configurations, winning vectorizer & model selection,
    artifact generation, and artifact serialization.

    Returns:
        Tuple[Any, TfidfVectorizer, Dict[str, Any]]: Best model, vectorizer, and metrics dict.
    """
    logger.info("========== Starting Feature Engineering Optimization Pipeline ==========")
    log_environment_versions()

    # 1. Load dataset
    df = load_clean_dataset(dataset_path)
    X = df['text']
    y = df['label']

    # 2. Split dataset before vectorizer fitting to prevent data leakage
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)

    # 3. Optimize feature engineering across 36 parameter configurations
    feature_results = optimize_feature_engineering(X_train, y_train, X_test, y_test)

    # 4. Select the feature configuration with the highest mean 5-fold CV F1-score
    best_result = max(feature_results, key=lambda res: (res["mean_cv_f1"], res["f1_score"]))
    best_model = best_result["model"]
    best_vectorizer = best_result["vectorizer"]

    logger.info("==========================================")
    logger.info("   OPTIMIZED FEATURE CONFIGURATION RESULT ")
    logger.info("==========================================")
    logger.info(f"Selected Config ID:   {best_result['config_id']}")
    logger.info(f"max_features:         {best_result['max_features']}")
    logger.info(f"ngram_range:          {best_result['ngram_range']}")
    logger.info(f"min_df:               {best_result['min_df']}")
    logger.info(f"max_df:               {best_result['max_df']}")
    logger.info(f"5-Fold CV Mean F1:    {best_result['mean_cv_f1']:.4f} ± {best_result['std_cv_f1']:.4f}")
    logger.info(f"Test Accuracy:        {best_result['accuracy']:.4f}")
    logger.info(f"Test Precision:       {best_result['precision']:.4f}")
    logger.info(f"Test Recall:          {best_result['recall']:.4f}")
    logger.info(f"Test F1-Score:        {best_result['f1_score']:.4f}")
    logger.info(f"Confusion Matrix:\n{best_result['confusion_matrix']}")
    logger.info(f"Classification Report:\n{best_result['classification_report']}")
    logger.info("==========================================")

    # 6. Calibrate winning LinearSVC model for probability scoring (predict_proba support)
    logger.info("Calibrating winning LinearSVC model probabilities with CalibratedClassifierCV...")
    X_train_vec = best_vectorizer.transform(X_train)
    calibrated_model = CalibratedClassifierCV(estimator=best_model, cv=5)
    calibrated_model.fit(X_train_vec, y_train)

    # 7. Save winning vectorizer and calibrated model to disk
    save_model(calibrated_model, model_path)
    save_vectorizer(best_vectorizer, vectorizer_path)

    logger.info("========== Feature Engineering Optimization Completed Successfully ==========")
    return calibrated_model, best_vectorizer, best_result



def main() -> None:
    """Entry point for training script execution."""
    try:
        run_training_pipeline()
    except Exception as e:
        logger.critical(f"Feature engineering pipeline execution failed: {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
