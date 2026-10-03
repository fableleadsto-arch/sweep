from fastapi.testclient import TestClient

from sweep_core import __version__
from sweep_core.api import create_app
from sweep_core.config import Settings


def _client(settings: Settings | None = None) -> TestClient:
    return TestClient(create_app(settings))


class TestHealth:
    def test_health_ok(self):
        response = _client(Settings(env="ci", log_level="WARNING")).get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["env"] == "ci"
        assert isinstance(body["native_available"], bool)

    def test_version_endpoint(self):
        response = _client().get("/version")
        assert response.status_code == 200
        assert response.json() == {"name": "sweep", "version": __version__}

    def test_openapi_documented(self):
        response = _client().get("/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"] == "Sweep"
