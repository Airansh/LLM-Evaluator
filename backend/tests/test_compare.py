from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from tests.fakes import FakeFactory, FakeJudge, FakeProvider, fail, ok
from tests.test_runs import FIVE_CASES

LUNA = {"provider": "openai", "model_id": "gpt-6-luna"}
SOL = {"provider": "openai", "model_id": "gpt-6-sol"}


class Env:
    def __init__(self, client: TestClient, judge: FakeJudge, provider: FakeProvider) -> None:
        self.client, self.judge, self.provider = client, judge, provider

    def experiment(self, prompt: str = "v1", cases: list[dict[str, Any]] = FIVE_CASES, **extra: Any) -> str:
        body = {"name": prompt, "system_prompt": prompt, "cases": cases, **extra}
        response = self.client.post("/experiments", json=body)
        assert response.status_code == 201, response.text
        return response.json()["id"]

    def run(self, experiment_id: str, *models: dict[str, str]) -> None:
        response = self.client.post(f"/experiments/{experiment_id}/run", json={"models": list(models or [LUNA])})
        assert response.status_code == 202, response.text

    def compare(self, experiment_id: str, **params: str) -> dict[str, Any]:
        response = self.client.get(f"/experiments/{experiment_id}/compare", params=params)
        assert response.status_code == 200, response.text
        return response.json()


def _settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(
        openai_api_key="sk-test", openai_models=["gpt-6-luna", "gpt-6-sol"],
        anthropic_api_key="sk-test-2", judge_provider="anthropic", judge_model="claude-sonnet-5",
    )


@pytest.fixture
def env(make_settings: Callable[..., Settings]) -> Iterator[Env]:
    settings = _settings(make_settings)
    judge, provider = FakeJudge(score=80), FakeProvider("openai")
    with TestClient(create_app(settings, FakeFactory(settings, {"openai": provider}, judge))) as client:
        yield Env(client, judge, provider)


def _metric(model: dict[str, Any], name: str) -> dict[str, Any]:
    return next(m for m in model["metrics"] if m["metric"] == name)


@pytest.mark.parametrize(("candidate_score", "verdict"), [(76.0, "regression"), (77.0, "ok"), (79.0, "ok"), (90.0, "ok")])
def test_controlled_score_drop_trips_three_point_threshold(env: Env, candidate_score: float, verdict: str) -> None:
    baseline = env.experiment("v1")
    env.run(baseline)
    candidate = env.experiment("v2", baseline_experiment_id=baseline)
    env.judge.score = candidate_score
    env.run(candidate)

    body = env.compare(candidate)
    assert body["comparable"] is True and body["prompt_changed"] is True and body["threshold_pp"] == 3.0
    model = body["models"][0]
    correctness = _metric(model, "correctness")
    assert (correctness["paired_cases"], correctness["applicable_cases"]) == (5, 5)
    assert correctness["baseline_mean"] == 80 and correctness["delta_pp"] == pytest.approx(candidate_score - 80)
    assert model["verdict"] == verdict and correctness["verdict"] == verdict
    assert _metric(model, "groundedness")["applicable_cases"] == 4  # case_4 has no context
    assert body["has_regression"] is (verdict == "regression")
    if verdict == "regression":
        assert len(model["largest_drops"]) == 9  # 5 correctness + 4 groundedness per-case drops
    # Regression indicator shows up in the experiment list.
    listed = {e["id"]: e for e in env.client.get("/experiments").json()}
    assert listed[candidate]["regression"] == verdict and listed[baseline]["regression"] is None


def test_threshold_is_configurable(env: Env) -> None:
    baseline = env.experiment()
    env.run(baseline)
    candidate = env.experiment("v2", baseline_experiment_id=baseline, regression_threshold=5)
    env.judge.score = 76
    env.run(candidate)
    assert env.compare(candidate)["models"][0]["verdict"] == "ok"


def test_different_cases_are_not_comparable(env: Env) -> None:
    baseline = env.experiment()
    env.run(baseline)
    changed = [dict(c) for c in FIVE_CASES]
    changed[1]["expected_answer"] = "Something else."
    changed.append({"id": "extra", "input": "Q?", "expected_answer": "A."})
    candidate = env.experiment("v2", cases=changed[1:], baseline_experiment_id=baseline)
    env.run(candidate)

    body = env.compare(candidate)
    assert body["comparable"] is False and body["has_regression"] is None and body["models"] == []
    reasons = " | ".join(body["reasons"])
    assert "do not use the same test cases" in reasons
    assert "only in baseline: case_0" in reasons and "only in candidate: extra" in reasons
    assert "same id, different content: case_1" in reasons


def test_reordered_cases_are_still_comparable(env: Env) -> None:
    baseline = env.experiment()
    env.run(baseline)
    candidate = env.experiment("v2", cases=list(reversed(FIVE_CASES)), baseline_experiment_id=baseline)
    env.run(candidate)
    assert env.compare(candidate)["comparable"] is True


def test_different_model_or_judge_is_not_comparable(env: Env) -> None:
    baseline = env.experiment()
    env.run(baseline, LUNA)
    candidate = env.experiment("v2", baseline_experiment_id=baseline)
    env.judge.model_id = "a-different-judge"
    env.run(candidate, LUNA, SOL)

    body = env.compare(candidate)
    models = {m["model_id"]: m for m in body["models"]}
    assert models["gpt-6-sol"]["verdict"] == "not_comparable"
    assert models["gpt-6-sol"]["reasons"] == ["the baseline has no finished run for this model"]
    assert models["gpt-6-luna"]["verdict"] == "not_comparable"
    assert "judge_model_id: baseline='fake-judge', candidate='a-different-judge'" in models["gpt-6-luna"]["reasons"][0]
    assert body["comparable"] is False and body["has_regression"] is None

    only_sol = env.compare(candidate, model_id="gpt-6-sol")
    assert [m["model_id"] for m in only_sol["models"]] == ["gpt-6-sol"]


def test_missing_scores_refuse_a_verdict(env: Env) -> None:
    baseline = env.experiment()
    env.run(baseline)
    candidate = env.experiment("v2", baseline_experiment_id=baseline)
    env.judge.score = 10  # a huge drop, but one case is missing, so no verdict
    env.provider._script = lambda m, r, n: (
        fail("openai", m, "bad_request") if "Question 2" in r.user_message else ok("openai", m, "30 days")
    )
    env.run(candidate)

    model = env.compare(candidate)["models"][0]
    correctness = _metric(model, "correctness")
    assert correctness["verdict"] == "insufficient_data" and correctness["paired_cases"] == 4
    assert "1 of 5 applicable cases lack a score" in correctness["note"]
    assert model["verdict"] == "insufficient_data"


def test_no_baseline(env: Env) -> None:
    experiment = env.experiment()
    body = env.compare(experiment)
    assert body["comparable"] is False and body["reasons"] == ["no baseline experiment is set for this experiment"]
    assert env.client.get("/experiments/missing/compare").status_code == 404


def test_candidate_without_finished_runs(env: Env) -> None:
    baseline = env.experiment()
    env.run(baseline)
    candidate = env.experiment("v2", baseline_experiment_id=baseline)
    assert env.compare(candidate)["reasons"] == ["the candidate has no finished run"]


def test_results_endpoint_exposes_rationale_errors_and_flags(env: Env) -> None:
    env.provider._script = lambda m, r, n: (
        fail("openai", m, "auth", "bad key") if "Question 1" in r.user_message else ok("openai", m, "No idea.")
    )
    env.judge.score = 40
    experiment = env.experiment()
    env.run(experiment)

    body = env.client.get(f"/experiments/{experiment}/results").json()
    assert body["total_cases"] == 5 and len(body["cases"]) == 5 and body["low_score_threshold"] == 60
    by_id = {c["case_id"]: c for c in body["cases"]}
    failed = by_id["case_1"]["results"][0]
    assert failed["error"] == "[auth] bad key" and failed["flags"] == ["candidate_error"]
    scored = by_id["case_0"]["results"][0]
    assert scored["judge_reason"] == "fake judge" and scored["judge_model_id"] == "fake-judge"
    assert scored["missing_facts"] == ["30 days"]
    assert scored["flags"] == ["missing_facts", "low_correctness", "low_groundedness"]
    assert by_id["case_4"]["context"] is None and by_id["case_4"]["results"][0]["groundedness_score"] is None

    one = env.client.get(f"/experiments/{experiment}/results", params={"case_id": "case_1"}).json()
    assert [c["case_id"] for c in one["cases"]] == ["case_1"]
    assert env.client.get(f"/experiments/{experiment}/results", params={"case_id": "nope"}).status_code == 404


def test_failures_only_filter(env: Env) -> None:
    env.judge.score = 95
    env.provider._script = lambda m, r, n: ok("openai", m, "Within 30 days.")
    experiment = env.experiment()
    env.run(experiment)
    body = env.client.get(f"/experiments/{experiment}/results", params={"failures_only": True}).json()
    assert body["cases"] == []


def test_results_survive_a_restart(make_settings: Callable[..., Settings]) -> None:
    settings = _settings(make_settings)
    factory = FakeFactory(settings, {"openai": FakeProvider("openai")}, FakeJudge(score=70))
    with TestClient(create_app(settings, factory)) as client:
        experiment = client.post("/experiments", json={"name": "p", "system_prompt": "p", "cases": FIVE_CASES}).json()
        client.post(f"/experiments/{experiment['id']}/run", json={"models": [LUNA]})
        before_detail = client.get(f"/experiments/{experiment['id']}").json()
        before_results = client.get(f"/experiments/{experiment['id']}/results").json()

    # A brand-new app (fresh engine, no fakes) reading the same SQLite file.
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/experiments/{experiment['id']}").json() == before_detail
        assert client.get(f"/experiments/{experiment['id']}/results").json() == before_results
        assert [e["id"] for e in client.get("/experiments").json()] == [experiment["id"]]
    assert before_detail["runs"][0]["aggregate"]["correctness"] == {"mean": 70.0, "n": 5}
