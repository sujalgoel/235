"""HTTP-layer smoke tests with the pipeline mocked out so we don't load
multi-gigabyte models in CI."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Build a TestClient with pipeline.initialize / get_status stubbed."""
    with patch("src.api.app.pipeline") as p:
        p.initialize.return_value = None
        p.get_status.return_value = {
            "initialized": True,
            "modules": {"image": True, "text": True},
            "device": "cpu",
        }
        from src.api.app import app

        with TestClient(app) as c:
            yield c


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "RealityCheck AI"
    assert body["status"] == "operational"


def test_health_reports_modules(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["modules"] == {"image": True, "text": True}


def test_text_endpoint_rejects_short_input(client):
    r = client.post("/api/v1/analyze/text", data={"text": "short"})
    assert r.status_code == 422  # FastAPI validation error


def test_text_endpoint_rejects_oversize_input(client):
    r = client.post("/api/v1/analyze/text", data={"text": "x" * 1001})
    assert r.status_code == 422


def test_oversize_image_returns_413(client, tmp_path):
    # Build a payload bigger than the 10 MiB limit; the streaming reader
    # should reject before allocating the whole buffer.
    big = b"\x00" * (11 * 1024 * 1024)
    bio = "this bio is definitely long enough to clear the validator"
    files = {"image": ("big.png", big, "image/png")}
    r = client.post(
        "/api/v1/analyze/profile",
        data={"bio_text": bio},
        files=files,
    )
    assert r.status_code == 413
