"""
presenter/routes.py
===================
Responsibility:
- Defines Flask Blueprints and HTTP endpoints for the application.
- Delegates incoming requests to Presenter logic.
- Renders HTML Views or returns JSON responses.
"""

import os
import sys
from typing import Any
from flask import Blueprint, render_template, request, jsonify, render_template_string

# Ensure project root directory is in sys.path when executed directly
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from utils.logger import get_logger
from presenter.presenter import predict_news
from utils.helper import verify_file_exists

logger = get_logger(__name__)

# Define Blueprint for presenter routes
presenter_bp = Blueprint('presenter', __name__)


@presenter_bp.route('/', methods=['GET'])
def index() -> Any:
    """
    GET /
    Renders the main home page view with article submission form.
    """
    logger.info("Serving home page GET /")
    return render_template('index.html')


@presenter_bp.route('/predict', methods=['POST'])
def predict() -> Any:
    """
    POST /predict
    Accepts news article text from form data or JSON payload,
    validates input, calls presenter.predict_news(), and renders result.html.
    """
    logger.info("Processing news article classification request POST /predict")
    
    # Retrieve article text from form or json body
    if request.is_json and request.get_json():
        payload = request.get_json()
        news_text = payload.get('news_text') or payload.get('text') or payload.get('article_text', '')
    else:
        news_text = request.form.get('news_text') or request.form.get('text') or request.form.get('article_text', '')

    response_data = predict_news(news_text)

    if not response_data["success"]:
        logger.warning(f"Prediction input validation or processing failed: {response_data['error']}")
        return render_template(
            'index.html',
            error=response_data["error"],
            news_text=news_text
        ), 400

    result_payload = response_data["result"]
    logger.info(
        f"Prediction completed successfully: {result_payload['prediction']} "
        f"({result_payload['confidence']}%)"
    )
    return render_template('result.html', result=result_payload, news_text=news_text)


@presenter_bp.route('/about', methods=['GET'])
def about() -> Any:
    """
    GET /about
    Displays information about the Fake News Detector project.
    """
    logger.info("Serving project information page GET /about")
    try:
        return render_template('about.html')
    except Exception:
        # Fallback template rendering if about.html is not present
        about_html = """
        {% extends "base.html" %}
        {% block title %}Fake News Detector - About{% endblock %}
        {% block content %}
        <div class="row justify-content-center">
            <div class="col-md-8">
                <h1 class="text-center mb-4">About Fake News Detector</h1>
                <p class="lead text-center">An end-to-end Machine Learning web application built with Python, Flask, and scikit-learn following MVP architecture.</p>
                <div class="card p-4 shadow-sm mt-4">
                    <h5>Key Features:</h5>
                    <ul>
                        <li>TF-IDF Text Vectorization and Logistic Regression classification.</li>
                        <li>High accuracy trained model on comprehensive news datasets.</li>
                        <li>Clean Presenter-driven architecture isolating UI from Model logic.</li>
                    </ul>
                    <div class="text-center mt-3">
                        <a href="{{ url_for('presenter.index') }}" class="btn btn-primary">Back to Home</a>
                    </div>
                </div>
            </div>
        </div>
        {% endblock %}
        """
        return render_template_string(about_html)


@presenter_bp.route('/health', methods=['GET'])
def health() -> Any:
    """
    GET /health
    Health check endpoint returning application and model status in JSON format.

    Returns:
        JSON response with 'status' and 'model_loaded'.
    """
    logger.info("Checking application health GET /health")
    model_exists = verify_file_exists("model/model.pkl")
    vectorizer_exists = verify_file_exists("model/vectorizer.pkl")
    model_loaded = model_exists and vectorizer_exists

    status_str = "healthy" if model_loaded else "unhealthy"
    status_code = 200 if model_loaded else 503

    return jsonify({
        "status": status_str,
        "model_loaded": model_loaded
    }), status_code
