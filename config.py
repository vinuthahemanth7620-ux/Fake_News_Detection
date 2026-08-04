"""
config.py
=========
Central configuration settings for the Fake News Detection application.

Responsibility:
- Defines environment settings (Development, Production, Testing).
- Secret keys and security configs.
- Paths to dataset files, trained model files, and vectorizer files.

TODO:
- Define Config base class and environment-specific subclasses.
- Set dataset paths (dataset/Fake.csv, dataset/True.csv).
- Set model file paths (model/model.pkl, model/vectorizer.pkl).
"""
import os

class Config:
    """Base configuration settings."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key-change-in-prod')
    DEBUG = False
    TESTING = False
    DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
