from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from image_analysis import AnalysisError, analyze_image

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("dermalens")

MODEL_CARD = {
    "architecture": "Swin Transformer research pipeline",
    "status": "weights_not_distributed",
    "inference_enabled": False,
    "input_shape": "224 × 224 RGB",
    "classes": [
        {"code": "akiec", "name": "Actinic keratoses"},
        {"code": "bcc", "name": "Basal cell carcinoma"},
        {"code": "bkl", "name": "Benign keratosis-like lesions"},
        {"code": "df", "name": "Dermatofibroma"},
        {"code": "mel", "name": "Melanoma"},
        {"code": "nv", "name": "Melanocytic nevi"},
        {"code": "vasc", "name": "Vascular lesions"},
    ],
    "intended_use": (
        "Portfolio demonstration of a safe image intake and model-readiness workflow."
    ),
    "limitations": [
        "The trained model weights are not present in this repository.",
        "No diagnostic prediction is generated.",
        (
            "Image quality checks cannot determine whether a lesion "
            "is benign or malignant."
        ),
    ],
}


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config.update(
        TESTING=testing,
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
    )

    @app.after_request
    def set_security_headers(response: Any) -> Any:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self'; "
            "script-src 'self'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; "
            "base-uri 'none'; "
            "form-action 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> str:
        return render_template("index.html", model_card=MODEL_CARD)

    @app.get("/robots.txt")
    def robots() -> Any:
        return app.send_static_file("robots.txt")

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "status": "healthy",
                    "service": "dermalens",
                    "inference_enabled": False,
                }
            ),
            HTTPStatus.OK,
        )

    @app.get("/api/model-card")
    def model_card() -> tuple[Any, int]:
        return jsonify({"model": MODEL_CARD}), HTTPStatus.OK

    @app.post("/api/analyze")
    def analyze() -> tuple[Any, int]:
        uploaded = request.files.get("image")
        if uploaded is None:
            return (
                jsonify(
                    {
                        "error": "missing_image",
                        "message": "Choose a JPEG, PNG, or WebP image.",
                    }
                ),
                HTTPStatus.BAD_REQUEST,
            )

        try:
            result = analyze_image(
                uploaded.stream,
                filename=uploaded.filename or "",
                declared_mime=uploaded.mimetype,
            )
            logger.info(
                "image_analysis_completed format=%s size=%sx%s readiness=%s",
                result["file"]["format"],
                result["file"]["width"],
                result["file"]["height"],
                result["readiness"],
            )
            return (
                jsonify(
                    {
                        "success": True,
                        "analysis": result,
                        "inference": {
                            "performed": False,
                            "reason": (
                                "Model weights are not distributed; no medical "
                                "classification was generated."
                            ),
                        },
                        "disclaimer": (
                            "Research and portfolio demonstration only. "
                            "This application does not provide medical advice."
                        ),
                    }
                ),
                HTTPStatus.OK,
            )
        except AnalysisError as exc:
            return (
                jsonify({"error": "invalid_image", "message": str(exc)}),
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )

    @app.errorhandler(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    def request_too_large(_: Exception) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "error": "request_too_large",
                    "message": "Image exceeds the 8 MB upload limit.",
                }
            ),
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> Any:
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "error": "http_error",
                        "message": error.description,
                    }
                ),
                error.code or HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        return error

    @app.errorhandler(Exception)
    def unexpected_error(error: Exception) -> tuple[Any, int]:
        logger.exception("unhandled_request_error", exc_info=error)
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "error": "internal_error",
                        "message": "The image could not be analyzed.",
                    }
                ),
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        return render_template("error.html"), HTTPStatus.INTERNAL_SERVER_ERROR

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
