import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_cors_allows_configured_localhost(client):
    """Origin in CORS_ALLOW_ORIGINS receives Access-Control-Allow-Origin."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_untrusted_origin(client):
    """Untrusted origin does not receive Access-Control-Allow-Origin."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://malicious-site.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") is None


def test_cors_allows_vercel_and_hf_origins(client):
    """Vercel and Hugging Face domains match regex and receive Access-Control-Allow-Origin."""
    response_vercel = client.options(
        "/health",
        headers={
            "Origin": "https://vectorless-front-8gpg.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response_vercel.status_code == 200
    assert response_vercel.headers.get("access-control-allow-origin") == "https://vectorless-front-8gpg.vercel.app"

    response_hf = client.options(
        "/health",
        headers={
            "Origin": "https://amir-rag-space.hf.space",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response_hf.status_code == 200
    assert response_hf.headers.get("access-control-allow-origin") == "https://amir-rag-space.hf.space"

