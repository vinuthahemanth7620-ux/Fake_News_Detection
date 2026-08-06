import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from model.predictor import get_prediction_confidence, predict_news, NewsPredictor


class DummyModel:
    """Mock model that implements neither predict_proba nor decision_function."""
    def predict(self, X):
        return np.array([1])


@pytest.fixture
def sample_data():
    X = ["Breaking news real story", "Fake clickbait scam report"]
    y = [1, 0]
    vec = TfidfVectorizer().fit(X)
    return X, y, vec


def test_predict_proba_models(sample_data):
    X, y, vec = sample_data
    for clf_cls in [LogisticRegression, RandomForestClassifier, MultinomialNB]:
        clf = clf_cls().fit(vec.transform(X), y)
        label, conf = get_prediction_confidence("Breaking news real story", model=clf, vectorizer=vec)
        assert label in (0, 1)
        assert 0.0 <= conf <= 100.0


def test_decision_function_models(sample_data):
    X, y, vec = sample_data
    for clf_cls in [LinearSVC, PassiveAggressiveClassifier]:
        clf = clf_cls().fit(vec.transform(X), y)
        label, conf = get_prediction_confidence("Breaking news real story", model=clf, vectorizer=vec)
        assert label in (0, 1)
        assert 50.0 <= conf <= 100.0


def test_fallback_model(sample_data):
    _, _, vec = sample_data
    dummy = DummyModel()
    label, conf = get_prediction_confidence("Test text", model=dummy, vectorizer=vec)
    assert label == 1
    assert conf == 100.0


def test_predict_news_structure(sample_data):
    X, y, vec = sample_data
    clf = LinearSVC().fit(vec.transform(X), y)
    res = predict_news("Breaking news real story", model=clf, vectorizer=vec)
    assert "prediction" in res
    assert res["prediction"] in ("REAL", "FAKE")
    assert "label" in res
    assert "confidence" in res
    assert isinstance(res["confidence"], float)

