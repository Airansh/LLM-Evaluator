from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.db.models import ModelRun, RunStatus
from app.db.session import session_factory
from app.main import create_app

from tests.fakes import FakeFactory, FakeJudge, FakeProvider, fail, ok

FIVE_CASES = [
    {"id": f"case_{i}", "input": f"Question {i}?", "context": "Returns within 30 days." if i != 4 else None,
     "expected_answer": "Within 30 days.", "required_facts": ["30 days"] if i != 3 else []}
    for i in range(5)
]


def _settings(make_settings: Callable[..., Settings], **extra: Any) -> Settings:
    return make_settings(
        openai_api_key="sk-test-openai-key-value",
        anthropic_api_key="sk-test-anthropic-key-value",
        openai_models=["gpt-6-luna", "gpt-6-sol"],
        anthropic_models=["claude-haiku-4-5"],
        judge_provider="anthropic",
        judge_model="claude-sonnet-5",
        **extra,
    )


def _create(client: TestClient, cases: list[dict[str, Any]] = FIVE_CASES) -> str:
    response = client.post("/experiments", json={"name": "t", "system_prompt": "Be helpful.", "cases": cases})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_five_cases_times_two_models_creates_ten_results(make_settings: Callable[..., Settings]) -> None:
    openai = FakeProvider("openai")
    judge = FakeJudge()
    app = create_app(settings := _settings(make_settings), FakeFactory(settings, {"openai": openai}, judge))
    with TestClient(app) as client:
        experiment_id = _create(client)
        response = client.post(
            f"/experiments/{experiment_id}/run",
            json={"models": [{"provider": "openai", "model_id": "gpt-6-luna"},
                             {"provider": "openai", "model_id": "gpt-6-sol"}]},
        )
        assert response.status_code == 202, response.text
        assert [r["status"] for r in response.json()["runs"]] == ["pending", "pending"]

        body = client.get(f"/experiments/{experiment_id}").json()

    assert body["status"] == "completed"
    assert len(openai.calls) == 10 and len(judge.calls) == 10
    for run in body["runs"]:
        agg = run["aggregate"]
        assert run["status"] == "completed" and agg["total_cases"] == 5 and agg["responses_ok"] == 5
        assert agg["correctness"] == {"mean": 80.0, "n": 5}
        assert agg["groundedness"]["n"] == 4  # case_4 has no context -> N/A
        assert agg["required_facts"]["n"] == 4  # case_3 has no facts -> N/A
        assert run["judge_model_id"] == "fake-judge" and run["eval_config"]["judge_model_id"] == "fake-judge"
        # No verified price for these model IDs, so cost is unavailable rather than invented.
        assert agg["candidate_cost"]["total"] is None
    # The candidate saw the system prompt and the case context, not the expected answer.
    _, request = openai.calls[0]
    assert request.system_prompt == "Be helpful." and "Within 30 days." not in request.user_message


def test_failing_model_is_visible_and_other_results_survive(make_settings: Callable[..., Settings]) -> None:
    secret = "sk-test-openai-key-value"

    def openai_script(model_id: str, request, n: int):  # noqa: ANN001, ANN202
        if model_id == "gpt-6-sol":
            return fail("openai", model_id, "auth", f"invalid api key {secret}")
        return ok("openai", model_id, "Within 30 days.")

    anthropic = FakeProvider("anthropic", lambda m, r, n: fail("anthropic", m, "rate_limit") if "Question 2" in r.user_message
                             else ok("anthropic", m, "30 days"))
    settings = _settings(make_settings)
    factory = FakeFactory(settings, {"openai": FakeProvider("openai", openai_script), "anthropic": anthropic})
    with TestClient(create_app(settings, factory)) as client:
        experiment_id = _create(client)
        client.post(
            f"/experiments/{experiment_id}/run",
            json={"models": [{"provider": "openai", "model_id": "gpt-6-luna"},
                             {"provider": "openai", "model_id": "gpt-6-sol"},
                             {"provider": "anthropic", "model_id": "claude-haiku-4-5"}]},
        )
        body = client.get(f"/experiments/{experiment_id}").json()

    runs = {run["model_id"]: run for run in body["runs"]}
    assert body["status"] == "partial"
    assert runs["gpt-6-luna"]["status"] == "completed"
    assert runs["gpt-6-sol"]["status"] == "failed" and runs["gpt-6-sol"]["aggregate"]["candidate_errors"] == 5
    assert runs["claude-haiku-4-5"]["status"] == "partial"
    assert runs["claude-haiku-4-5"]["aggregate"]["responses_ok"] == 4
    # Claude Haiku 4.5 has a verified price; the rate-limited call reported no usage, so 4 of 5 are priced.
    cost = runs["claude-haiku-4-5"]["aggregate"]["candidate_cost"]
    assert cost["n_priced"] == 4 and cost["complete"] is True
    # Rate-limit retries were bounded: 4 good cases + 3 attempts for the failing one.
    assert len(anthropic.calls) == 4 + 3

    with session_factory()() as session:
        errors = [r.error for run in session.query(ModelRun).all() for r in run.results if r.error]
    assert any("[rate_limit]" in e and "after 3 attempts" in e for e in errors)
    assert all(secret not in e for e in errors) and any("[REDACTED]" in e for e in errors)


def test_run_request_validation(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(openai_models=["gpt-6-luna"], judge_provider="anthropic", judge_model="claude-sonnet-5")
    with TestClient(create_app(settings)) as client:
        experiment_id = _create(client)
        url = f"/experiments/{experiment_id}/run"

        missing_key = client.post(url, json={"models": [{"provider": "openai", "model_id": "gpt-6-luna"}]})
        assert missing_key.status_code == 422
        assert missing_key.json()["detail"] == ["openai/gpt-6-luna: OPENAI_API_KEY is not set"]

        unknown = client.post(url, json={"models": [{"provider": "openai", "model_id": "gpt-typo"}]})
        assert "not configured" in unknown.json()["detail"][0]

        sample_off = client.post(url, json={"models": [{"provider": "sample", "model_id": "sample-terse"}]})
        assert "sample mode is disabled" in sample_off.json()["detail"][0]

        duplicate = client.post(url, json={"models": [{"provider": "openai", "model_id": "a"}] * 2})
        assert duplicate.status_code == 422
        assert client.post("/experiments/nope/run", json={"models": [{"provider": "openai", "model_id": "a"}]}).status_code == 404
        # Nothing was queued by the rejected requests.
        assert client.get(f"/experiments/{experiment_id}").json()["runs"] == []


def test_real_model_needs_an_available_judge(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(openai_api_key="sk-x", openai_models=["gpt-6-luna"], judge_provider="anthropic",
                             judge_model="claude-sonnet-5")
    with TestClient(create_app(settings)) as client:
        experiment_id = _create(client)
        response = client.post(f"/experiments/{experiment_id}/run",
                               json={"models": [{"provider": "openai", "model_id": "gpt-6-luna"}]})
    assert response.status_code == 422
    assert response.json()["detail"] == ["openai/gpt-6-luna: judge unavailable: ANTHROPIC_API_KEY is not set"]


def test_sample_mode_end_to_end_is_labeled(make_settings: Callable[..., Settings]) -> None:
    with TestClient(create_app(make_settings(sample_mode=True))) as client:
        models = client.get("/models").json()["models"]
        assert {m["model_id"] for m in models if m["sample"]} == {"sample-extractive", "sample-terse", "sample-flaky"}
        experiment_id = _create(client)
        client.post(f"/experiments/{experiment_id}/run",
                    json={"models": [{"provider": "sample", "model_id": "sample-extractive"}]})
        run = client.get(f"/experiments/{experiment_id}").json()["runs"][0]
    assert run["sample"] is True and run["judge_provider"] == "sample"
    assert run["judge_model_id"] == "sample-heuristic-judge"
    assert run["aggregate"]["responses_ok"] == 5


def test_create_experiment_validation(client: TestClient) -> None:
    dup = client.post("/experiments", json={"name": "t", "system_prompt": "p", "cases": [FIVE_CASES[0]] * 2})
    assert dup.status_code == 422 and "duplicate test case ids" in dup.text
    bad_baseline = client.post(
        "/experiments", json={"name": "t", "system_prompt": "p", "cases": FIVE_CASES, "baseline_experiment_id": "nope"}
    )
    assert bad_baseline.status_code == 422 and "does not exist" in bad_baseline.text


def test_interrupted_runs_are_marked_failed_on_restart(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings()
    with TestClient(create_app(settings)) as client:
        experiment_id = _create(client)
    with session_factory()() as session:
        session.add(ModelRun(experiment_id=experiment_id, provider="openai", model_id="m",
                             status=RunStatus.running, eval_config={}, eval_config_hash="x"))
        session.commit()
    with TestClient(create_app(settings)) as client:
        body = client.get(f"/experiments/{experiment_id}").json()
    assert body["status"] == "failed"
    assert body["runs"][0]["status"] == "failed" and "interrupted" in body["runs"][0]["error"]
