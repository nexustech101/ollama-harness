"""Native API smoke tests using FastAPI's synchronous test client."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api import ServerConfig, create_app
from domain.events import VERSION


def _client(tmp_path: Path, **kwargs) -> TestClient:
    config = ServerConfig(
        model="m",
        base_url="http://127.0.0.1:1",
        workspace=tmp_path,
        provider="ollama",
        **kwargs,
    )
    return TestClient(create_app(config))


def test_native_health_reports_version_and_no_paths(tmp_path):
    client = _client(tmp_path)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == VERSION
    assert body["provider"] == "ollama"
    assert "workspace" not in body
    assert str(tmp_path) not in response.text


def test_native_health_does_not_require_auth(tmp_path):
    """Health is a liveness endpoint; it must stay usable by orchestrators."""
    client = _client(tmp_path, api_key="secret")
    assert client.get("/api/v1/health").status_code == 200
