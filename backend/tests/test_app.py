from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import inspect

from app.config import Settings
from app.db.session import get_engine
from app.main import create_app

FAKE_KEY = "sk-test-DO-NOT-LEAK-1234567890"


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0", "database": "ok"}


def test_schema_initialized(client: TestClient) -> None:
    tables = set(inspect(get_engine()).get_table_names())
    assert {"experiments", "test_cases", "model_runs", "evaluation_results"} <= tables


def test_cors_allows_local_frontend(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_models_without_keys_are_unavailable(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(anthropic_models=["claude-a"], judge_provider="anthropic", judge_model="claude-j")
    with TestClient(create_app(settings)) as client:
        body = client.get("/models").json()
    assert body["models"] == [
        {"provider": "anthropic", "model_id": "claude-a", "available": False, "reason": "ANTHROPIC_API_KEY is not set",
         "sample": False}
    ]
    assert body["judge"]["available"] is False
    assert body["judge"]["reason"] == "ANTHROPIC_API_KEY is not set"


def test_models_with_key_never_expose_it(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(
        openai_api_key=FAKE_KEY,
        openai_models=["gpt-a", "gpt-b"],
        gemini_models=["gemini-a"],
        judge_provider="openai",
        judge_model="gpt-judge",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/models")
    assert FAKE_KEY not in response.text
    body = response.json()
    availability = {(m["provider"], m["model_id"]): m["available"] for m in body["models"]}
    assert availability == {("openai", "gpt-a"): True, ("openai", "gpt-b"): True, ("gemini", "gemini-a"): False}
    assert body["judge"] == {"provider": "openai", "model_id": "gpt-judge", "available": True, "reason": None}


def test_settings_parse_comma_separated_env(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("ANTHROPIC_MODELS", "model-a, model-b,,")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test,http://b.test")
    settings = Settings(_env_file=None)
    assert settings.anthropic_models == ["model-a", "model-b"]
    assert settings.anthropic_api_key is None
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_sample_dataset_endpoint_and_validation(client: TestClient) -> None:
    sample = client.get("/datasets/sample").json()
    assert sample["cases"][0]["id"] == "returns_001"

    ok = client.post("/datasets/validate", json=sample).json()
    assert ok["valid"] is True and ok["case_count"] == len(sample["cases"]) and len(ok["dataset_hash"]) == 64

    created = client.post("/experiments", json={"name": "x", "system_prompt": "p", "cases": sample["cases"]}).json()
    assert created["dataset_hash"] == ok["dataset_hash"]  # the UI matches baselines by this hash

    bad = client.post("/datasets/validate", json={"cases": [{"id": "a", "input": "q"}, {"id": "a", "input": "q"}]}).json()
    assert bad["valid"] is False
    assert any("expected_answer" in e for e in bad["errors"])
    assert client.post("/datasets/validate", json="not a dataset").json()["valid"] is False


def test_evalforge_anthropic_key_alias(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("EVALFORGE_ANTHROPIC_API_KEY", "sk-alias")
    assert Settings(_env_file=None).api_key("anthropic").get_secret_value() == "sk-alias"  # type: ignore[union-attr]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-primary")
    assert Settings(_env_file=None).api_key("anthropic").get_secret_value() == "sk-primary"  # type: ignore[union-attr]
