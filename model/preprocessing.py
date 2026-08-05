"""
model/preprocessing.py
======================
Responsibility:
- Load, merge, label, and shuffle Fake and True news datasets.
- Reusable text cleaning functions for NLP processing.
- Removes HTML, URLs, punctuation, numbers, special characters, and extra whitespace.
- Applies lowercasing, tokenization, NLTK stopword removal, and PorterStemmer.
- Preprocesses all dataset articles and saves output to dataset/cleaned_news.csv.
"""

import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from utils.logger import get_logger
from utils.helper import verify_file_exists, ensure_directory_exists

logger = get_logger(__name__)

# Module-level singletons for performance
_STEMMER = PorterStemmer()
_STOP_WORDS = None


def _download_nltk_resources() -> None:
    """Downloads required NLTK datasets if not already present."""
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            if resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            else:
                nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            logger.info(f"Downloading NLTK resource: '{resource}'")
            nltk.download(resource, quiet=True)


def get_stop_words() -> set[str]:
    """
    Returns cached set of English stopwords, downloading resources if needed.

    Returns:
        set[str]: Set of English stop words.
    """
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _download_nltk_resources()
        _STOP_WORDS = set(stopwords.words('english'))
    return _STOP_WORDS


def convert_to_lowercase(text: str) -> str:
    """Converts text to lowercase."""
    return text.lower()


def remove_html_tags(text: str) -> str:
    """Removes HTML tags from text."""
    return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text: str) -> str:
    """Removes HTTP/HTTPS URLs and www domain links from text."""
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)


def remove_punctuation(text: str) -> str:
    """Removes standard punctuation marks from text."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text: str) -> str:
    """Removes numeric digits from text."""
    return re.sub(r'\d+', ' ', text)


def remove_special_characters(text: str) -> str:
    """Removes special non-alphabetic characters from text."""
    return re.sub(r'[^a-zA-Z\s]', ' ', text)


def remove_extra_whitespace(text: str) -> str:
    """Collapses multiple spaces into single spaces and strips leading/trailing spaces."""
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes text into a list of word tokens.

    Args:
        text (str): Input text string.

    Returns:
        list[str]: List of word tokens.
    """
    _download_nltk_resources()
    try:
        return word_tokenize(text)
    except Exception as e:
        logger.warning(f"NLTK tokenization fallback to split(): {e}")
        return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Filters out English stopwords from token list.

    Args:
        tokens (list[str]): List of word tokens.

    Returns:
        list[str]: Tokens with stopwords removed.
    """
    stop_words = get_stop_words()
    return [word for word in tokens if word not in stop_words]


def apply_porter_stemmer(tokens: list[str]) -> list[str]:
    """
    Applies PorterStemmer to reduce words to their root stems.

    Args:
        tokens (list[str]): List of word tokens.

    Returns:
        list[str]: Stemmed word tokens.
    """
    return [_STEMMER.stem(word) for word in tokens]


def clean_text(text: str) -> str:
    """
    Applies complete NLP preprocessing pipeline to a raw text string.

    Pipeline steps:
    1. Lowercasing
    2. HTML tag removal
    3. URL removal
    4. Punctuation removal
    5. Number removal
    6. Special character removal
    7. Extra whitespace removal
    8. Tokenization
    9. Stopwords removal
    10. Porter stemming

    Args:
        text (str): Raw input text article.

    Returns:
        str: Cleaned and processed text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = convert_to_lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)

    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    stemmed_tokens = apply_porter_stemmer(tokens)

    return " ".join(stemmed_tokens)


def load_and_merge_datasets(
    fake_path: str = "dataset/Fake.csv",
    true_path: str = "dataset/True.csv"
) -> pd.DataFrame:
    """
    Verifies existence of datasets, assigns labels (Fake=0, True=1),
    merges into a single DataFrame, shuffles with random_state=42,
    and retains only 'text' and 'label' columns.

    Args:
        fake_path (str): File path to Fake.csv.
        true_path (str): File path to True.csv.

    Returns:
        pd.DataFrame: Merged and shuffled DataFrame containing 'text' and 'label'.
    """
    logger.info("Verifying dataset files...")
    if not verify_file_exists(fake_path):
        raise FileNotFoundError(f"Fake dataset missing at {fake_path}")
    if not verify_file_exists(true_path):
        raise FileNotFoundError(f"True dataset missing at {true_path}")

    try:
        logger.info("Loading datasets into DataFrames...")
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)

        df_fake['label'] = 0
        df_true['label'] = 1

        logger.info(f"Loaded Fake news: {len(df_fake)} rows | True news: {len(df_true)} rows.")

        df_merged = pd.concat([df_fake, df_true], ignore_index=True)

        if 'text' not in df_merged.columns:
            raise KeyError("Dataset missing mandatory 'text' column.")

        df_merged = df_merged[['text', 'label']]

        # Shuffle dataset using random_state=42
        df_shuffled = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Merged and shuffled dataset. Total records: {len(df_shuffled)}.")

        return df_shuffled

    except Exception as e:
        logger.error(f"Error loading and merging datasets: {e}", exc_info=True)
        raise


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies clean_text preprocessing to every news article in DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'text' column.

    Returns:
        pd.DataFrame: DataFrame with cleaned text.
    """
    logger.info("Applying preprocessing pipeline to all news articles...")
    try:
        df['text'] = df['text'].fillna('')
        df['text'] = df['text'].apply(clean_text)

        initial_len = len(df)
        df = df[df['text'].str.strip() != ''].reset_index(drop=True)
        logger.info(f"Preprocessing finished. {len(df)}/{initial_len} articles retained.")

        return df
    except Exception as e:
        logger.error(f"Error during dataset text preprocessing: {e}", exc_info=True)
        raise


def save_cleaned_dataset(
    df: pd.DataFrame,
    output_path: str = "dataset/cleaned_news.csv"
) -> None:
    """
    Saves cleaned DataFrame to target CSV path.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        output_path (str): File destination path.
    """
    try:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            ensure_directory_exists(dir_name)

        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved cleaned dataset to '{output_path}'.")
    except Exception as e:
        logger.error(f"Error saving cleaned dataset to '{output_path}': {e}", exc_info=True)
        raise


def run_preprocessing_pipeline(
    fake_path: str = "dataset/Fake.csv",
    true_path: str = "dataset/True.csv",
    output_path: str = "dataset/cleaned_news.csv"
) -> pd.DataFrame:
    """
    Runs full data loading, text preprocessing, and file saving pipeline.

    Args:
        fake_path (str): Path to Fake.csv dataset.
        true_path (str): Path to True.csv dataset.
        output_path (str): Output CSV path for cleaned dataset.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    logger.info("========== Starting Data Preprocessing Pipeline ==========")
    _download_nltk_resources()
    df = load_and_merge_datasets(fake_path, true_path)
    df_cleaned = preprocess_dataset(df)
    save_cleaned_dataset(df_cleaned, output_path)
    logger.info("========== Data Preprocessing Pipeline Completed ==========")
    return df_cleaned


if __name__ == "__main__":
    run_preprocessing_pipeline()

