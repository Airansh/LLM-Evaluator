"""Live benchmark: claude-haiku-4-5 vs gemini-3.8-flash, judged by claude-sonnet-5, under a hard budget.

Run from backend/:  PYTHONPATH=. python scripts/live_benchmark.py smoke|full
Needs EVALFORGE_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY) and GEMINI_API_KEY. Makes paid API calls.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Claude Code sets ANTHROPIC_BASE_URL for its own connection; the app must call the public API.
os.environ.pop("ANTHROPIC_BASE_URL", None)

from app.config import Settings  # noqa: E402
from app.db.models import EvaluationResult, Experiment  # noqa: E402
from app.db.session import configure_engine, init_db, session_factory  # noqa: E402
from app.evaluation.aggregate import aggregate_run  # noqa: E402
from app.evaluation.pricing import latest_pricing  # noqa: E402
from app.evaluation.runner import execute_runs  # noqa: E402
from app.providers.registry import ProviderFactory  # noqa: E402
from app.schemas import DatasetIn, ExperimentCreate, ModelRef  # noqa: E402
from app.services import create_experiment, prepare_runs, runner_options  # noqa: E402

BUDGET_USD = 0.45  # per provider, under the user's $0.50 cap
# Gemini price could not be verified (pricing page blocked); use a deliberately pessimistic bound.
GEMINI_PESSIMISTIC = {"input": 5.0, "output": 30.0}  # USD per 1M tokens, guard only, never stored
PROMPT = (
    "You are a customer-support assistant for Northwind Outfitters. Answer using only the policy context "
    "provided. If the context does not answer the question, say so and suggest contacting support."
)
MODELS = [ModelRef(provider="anthropic", model_id="claude-haiku-4-5"), ModelRef(provider="gemini", model_id="gemini-3.8-flash")]
DATASET = Path(__file__).resolve().parents[2] / "datasets" / "customer_support.json"


def settings() -> Settings:
    return Settings(
        _env_file=None,
        anthropic_models=["claude-haiku-4-5"],
        gemini_models=["gemini-3.8-flash"],
        openai_models=[],
        judge_provider="anthropic",
        judge_model="claude-sonnet-5",
        sample_mode=False,
        candidate_max_output_tokens=1024,
        judge_max_output_tokens=1500,
        max_concurrency=4,
        database_url="sqlite:///./data/live.db",
    )


def spend(experiment: Experiment) -> dict[str, float]:
    """Actual Anthropic spend (verified prices) and a pessimistic Gemini bound, from reported tokens."""
    anthropic = gemini = 0.0
    for run in experiment.model_runs:
        for r in run.results:
            if run.provider == "anthropic":
                anthropic += r.estimated_cost or 0.0
            else:
                gemini += ((r.input_tokens or 0) * GEMINI_PESSIMISTIC["input"] + (r.output_tokens or 0) * GEMINI_PESSIMISTIC["output"]) / 1e6
            anthropic += r.judge_cost or 0.0  # the judge is Anthropic for every run
    return {"anthropic": anthropic, "gemini_upper_bound": gemini}


async def run(name: str, n_cases: int | None) -> dict[str, float]:
    s = settings()
    assert s.api_key("anthropic") and s.api_key("gemini"), "both keys must be present"
    configure_engine(s.database_url)
    init_db()
    factory, pricing = ProviderFactory(s), latest_pricing()
    cases = DatasetIn.model_validate(json.loads(DATASET.read_text())).cases
    cases = cases[:n_cases] if n_cases else cases
    with session_factory()() as session:
        exp = create_experiment(session, ExperimentCreate(name=name, system_prompt=PROMPT, cases=cases))
        _, plans = prepare_runs(session, factory, pricing, exp.id, MODELS)
        exp_id = exp.id
    await execute_runs(exp_id, plans, session_factory=session_factory(), options=runner_options(s),
                       pricing=pricing, secrets=factory.secrets())
    with session_factory()() as session:
        exp = session.get_one(Experiment, exp_id)
        print(f"\n== {exp.name} ({exp.id}) status={exp.status.value} cases={len(exp.test_cases)}")
        for r in exp.model_runs:
            a = aggregate_run(r.results)
            print(f"{r.provider}/{r.model_id} [{r.status.value}] ok {a.responses_ok}/{a.total_cases} "
                  f"cand_err {a.candidate_errors} judge_err {a.judge_errors} | "
                  f"correct {a.correctness.mean} (n={a.correctness.n}) grounded {a.groundedness.mean} (n={a.groundedness.n}) "
                  f"facts {a.required_facts.mean} (n={a.required_facts.n}) | p50 {a.latency_ms_median} ms | "
                  f"tokens {a.input_tokens}/{a.output_tokens} | cand_cost {a.candidate_cost.total} judge_cost {a.judge_cost.total}")
            for res in sorted(r.results, key=lambda x: x.test_case.position):
                print(f"   {res.test_case.case_id:<13} c={res.correctness_score} g={res.groundedness_score} "
                      f"f={res.required_facts_score} out_tok={res.output_tokens} j_out={res.judge_output_tokens} "
                      f"err={res.error} jerr={res.judge_error}")
        total = spend(exp)
        print("SPEND:", {k: round(v, 5) for k, v in total.items()})
        return total


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "smoke":
        s = asyncio.run(run("LIVE smoke: haiku-4-5 vs gemini-3.8-flash (2 cases)", 2))
        projected = {k: v * 12 / 2 for k, v in s.items()}
        print("PROJECTED full-run spend:", {k: round(v, 4) for k, v in projected.items()},
              "OK" if all(v < BUDGET_USD for v in projected.values()) else "OVER BUDGET - do not run full")
    elif mode == "full":
        asyncio.run(run("LIVE: claude-haiku-4-5 vs gemini-3.8-flash (12 cases, judge claude-sonnet-5)", None))
