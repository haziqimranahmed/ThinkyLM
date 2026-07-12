"""
ThinkyLM — API Tests

Tests the FastAPI application using TestClient.
Does not require a running server — uses in-process testing.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the ThinkyLM API."""
    from api.main import app
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "message" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "total_params" in data
    assert "vocab_size" in data
    assert "num_layers" in data
    assert data["total_params"] > 0


def test_generate_endpoint_basic(client):
    payload = {
        "prompt": "The nature of consciousness is",
        "max_new_tokens": 10,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert "tokens_generated" in data
    assert "model_loaded" in data
    assert "disclaimer" in data


def test_generate_endpoint_prompt_required(client):
    """Empty prompt should be rejected."""
    payload = {"prompt": ""}
    response = client.post("/generate", json=payload)
    assert response.status_code == 422  # Validation error


def test_generate_endpoint_prompt_too_long(client):
    payload = {"prompt": "x" * 3000}
    response = client.post("/generate", json=payload)
    assert response.status_code == 422


def test_generate_endpoint_invalid_temperature(client):
    payload = {"prompt": "hello", "temperature": -1.0}
    response = client.post("/generate", json=payload)
    assert response.status_code == 422


def test_analyse_argument_endpoint(client):
    payload = {
        "argument": "All humans are mortal, therefore Socrates is mortal.",
        "max_new_tokens": 15,
        "temperature": 0.8,
    }
    response = client.post("/analyse-argument", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "argument" in data
    assert "analysis" in data
    assert "tokens_generated" in data


def test_analyse_argument_empty_raises(client):
    payload = {"argument": ""}
    response = client.post("/analyse-argument", json=payload)
    assert response.status_code == 422


def test_api_has_no_crash_without_checkpoint(client):
    """API must not crash if no checkpoint is loaded; returns helpful message."""
    response = client.get("/health")
    data = response.json()
    # Should always return 200 with a message, never 500
    assert response.status_code == 200
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 10
