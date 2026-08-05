"""
app.py
======
Main application entry point for the Fake News Detection Flask web app.
Follows the MVP (Model-View-Presenter) architecture.

Responsibility:
- Initializes the Flask application instance.
- Configures application settings from config.py.
- Registers Presenter Blueprints and routes.
- Implements request logging middleware.
- Starts the Flask development server.
"""

import os
import sys

# Ensure project root directory is in sys.path when executed directly
_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from flask import Flask, request, Response
from config import Config
from presenter.routes import presenter_bp
from utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> Flask:
    """
    Application factory initializing Flask app instance, configuration,
    middleware logging, and Blueprints.

    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(_ROOT_DIR, 'view', 'templates'),
        static_folder=os.path.join(_ROOT_DIR, 'view', 'static')
    )

    # Load configuration
    app.config.from_object(Config)
    logger.info("Flask application configuration loaded from config.Config.")

    # Register Presenter Blueprint
    app.register_blueprint(presenter_bp)
    logger.info("Presenter blueprint registered successfully.")

    # Middleware to log all incoming requests
    @app.before_request
    def log_request_info() -> None:
        """Logs incoming HTTP request details."""
        logger.info(f"Incoming Request: {request.method} {request.path} - Remote IP: {request.remote_addr}")

    @app.after_request
    def log_response_info(response: Response) -> Response:
        """Logs HTTP response status for requests."""
        logger.info(f"Response Sent: {request.method} {request.path} - Status Code: {response.status_code}")
        return response

    return app


app = create_app()

if __name__ == '__main__':
    logger.info("Starting Fake News Detection Flask web server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
