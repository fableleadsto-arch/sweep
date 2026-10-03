from fastapi.testclient import TestClient
import pytest

from companion.config import BrainSettings, get_settings
from companion.main import app as companion_app
from companion.orchestrator import safe_math_eval, ToolError


@pytest.fixture(autouse=True)
def clean_overrides():
    original = dict(companion_app.dependency_overrides)
    companion_app.dependency_overrides.clear()
    yield
    companion_app.dependency_overrides.clear()
    companion_app.dependency_overrides.update(original)


def test_companion_api_fails_closed_without_token():
    settings = BrainSettings(_env_file=None, brain_service_token="")
    companion_app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(companion_app)
    assert client.post("/api/brain/execute", json={"code": "result = 1"}).status_code == 503


def test_companion_requires_token_and_execution_opt_in():
    settings = BrainSettings(_env_file=None, brain_service_token="test-only", allow_code_execution=False)
    companion_app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(companion_app)
    payload = {"code": "result = 1"}
    assert client.post("/api/brain/execute", json=payload).status_code == 401
    response = client.post("/api/brain/execute", json=payload,
                           headers={"Authorization": "Bearer test-only"})
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert "disabled" in response.json()["error"]


def test_web_auth_and_public_lightweight_health(monkeypatch):
    from app.config import Settings
    from app import main
    settings = Settings(_env_file=None, sweep_api_token="test-only")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(main.app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/platforms").status_code == 401
    assert client.get("/api/platforms", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/platforms", headers={"Authorization": "Bearer test-only"}).status_code == 200
    settings.sweep_api_token = ""
    assert client.get("/api/platforms").status_code == 503


@pytest.mark.parametrize("expression", ["9 ** 999999999", "9 ** (9 ** 9)", "1e999", "True + 1",
                                          "(-1) ** 0.5", "round(1, bad=2)", "sqrt()", "1+" * 2000])
def test_math_resource_limits(expression):
    with pytest.raises(ToolError):
        safe_math_eval(expression)


def test_math_normal_expressions():
    assert safe_math_eval("2 ** 10 + sqrt(16)") == 1028


def test_controller_preserves_url_case(tmp_path):
    from sweep.skills import _browser_target, SkillContext
    from sweep.store import ControllerStore
    ctx = SkillContext(store=ControllerStore(tmp_path))
    assert _browser_target("https://example.com/Case?Key=Value", ctx) == "https://example.com/Case?Key=Value"
