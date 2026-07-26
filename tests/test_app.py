from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from app import create_app


def image_bytes(
    *,
    size: tuple[int, int] = (512, 512),
    color: tuple[int, int, int] = (150, 90, 80),
    image_format: str = "PNG",
) -> BytesIO:
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, image_format)
    stream.seek(0)
    return stream


@pytest.fixture()
def client():
    return create_app(testing=True).test_client()


def test_unknown_route_preserves_not_found_status(client):
    response = client.get("/missing")

    assert response.status_code == 404


def test_home_is_accessible_and_honest(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"DERMALENS" in response.data
    assert b"No classification performed" in response.data
    assert b"Not a medical device" in response.data
    assert response.headers["X-Frame-Options"] == "DENY"


def test_health_explicitly_disables_inference(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "healthy",
        "service": "dermalens",
        "inference_enabled": False,
    }


def test_robots_allows_public_crawlers(client):
    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert b"User-agent: *" in response.data


def test_model_card_documents_missing_weights(client):
    response = client.get("/api/model-card")
    model = response.get_json()["model"]

    assert response.status_code == 200
    assert model["status"] == "weights_not_distributed"
    assert model["inference_enabled"] is False
    assert len(model["classes"]) == 7
    assert len(model["limitations"]) == 3


def test_valid_image_returns_quality_report_without_prediction(client):
    response = client.post(
        "/api/analyze",
        data={"image": (image_bytes(), "sample.png", "image/png")},
        content_type="multipart/form-data",
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["analysis"]["file"]["format"] == "PNG"
    assert data["analysis"]["file"]["width"] == 512
    assert len(data["analysis"]["checks"]) == 4
    assert data["inference"]["performed"] is False
    assert "medical advice" in data["disclaimer"]
    assert response.headers["Cache-Control"] == "no-store"


def test_missing_image_is_rejected(client):
    response = client.post("/api/analyze")

    assert response.status_code == 400
    assert response.get_json()["error"] == "missing_image"


def test_fake_image_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={"image": (BytesIO(b"not an image"), "fake.png", "image/png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert "readable image" in response.get_json()["message"]


def test_wrong_mime_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={"image": (image_bytes(), "sample.txt", "text/plain")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert "JPEG, PNG, and WebP" in response.get_json()["message"]


def test_path_like_filename_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={"image": (image_bytes(), "../sample.png", "image/png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert "filename" in response.get_json()["message"]


def test_tiny_image_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={
            "image": (
                image_bytes(size=(64, 64)),
                "tiny.png",
                "image/png",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert "96 × 96" in response.get_json()["message"]


def test_request_size_is_bounded(client):
    response = client.post(
        "/api/analyze",
        data=b"x" * (8 * 1024 * 1024 + 1),
        content_type="application/octet-stream",
    )

    assert response.status_code == 413
    assert response.get_json()["error"] == "request_too_large"
