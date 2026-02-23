"""Flask entrypoint with secure defaults for model-serving endpoints."""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# 🔥 Use absolute imports from project root
from src.main.webapp.api.routes import api_bp
from src.main.webapp.config import CONFIG_MAPPING
from src.main.webapp.services.model_service import ModelService


def create_app() -> Flask:
    """Application factory for WSGI servers and tests."""
    env_name = os.getenv("FLASK_ENV", "default")
    config_class = CONFIG_MAPPING.get(env_name, CONFIG_MAPPING["default"])

    app = Flask(__name__)
    app.config.from_object(config_class)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    CORS(app, resources={r"/*": {"origins": app.config["CORS_ORIGINS"]}})

    # Initialize model service safely
    model_service = ModelService(app.config)
    model_service.load_models()
    app.extensions["model_service"] = model_service

    app.register_blueprint(api_bp)

    register_error_handlers(app)
    return app


def register_error_handlers(app: Flask) -> None:
    """Sanitize runtime errors returned by the API."""

    @app.errorhandler(RequestEntityTooLarge)
    def payload_too_large(_: RequestEntityTooLarge):
        return jsonify({"error": "File is too large."}), 413

    @app.errorhandler(404)
    def not_found(_: Exception):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_: Exception):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(Exception)
    def unhandled_error(_: Exception):
        logging.getLogger(__name__).exception("Unhandled server error")
        return jsonify({"error": "Internal server error."}), 500

from src.main.webapp.app import create_app
app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )