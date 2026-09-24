"""SAMPLE MODE setup run: `python -m app.demo [--compare]` (from backend/).

Creates an experiment from datasets/customer_support.json, runs the offline sample
models through the real runner/scoring/persistence path, and prints per-model
aggregates. With --compare it runs a second experiment with a changed prompt against
the first as its baseline and prints the regression comparison.
No API keys are used and no real LLM is called.
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.config import Settings, get_settings
from app.db.models import Experiment
from app.db.session import configure_engine, init_db, session_factory
from app.evaluation.aggregate import RunAggregate, aggregate_run
from app.evaluation.compare import compare_experiments
from app.evaluation.pricing import latest_pricing
from app.evaluation.runner import execute_runs
from app.providers.registry import ProviderFactory
from app.providers.sample_provider import SAMPLE_MODELS
from app.schemas import DatasetIn, ExperimentCreate, ModelRef, TestCaseIn
from app.services import create_experiment, prepare_runs, runner_options

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "customer_support.json"
PROMPT_V1 = (
    "You are a customer-support assistant for Northwind Outfitters. Answer using only the policy context "
    "provided. If the context does not answer the question, say so and suggest contacting support."
)
PROMPT_V2 = PROMPT_V1 + " Keep every answer to one sentence."


def _fmt(value: float | None, suffix: str = "", digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}{suffix}"


def _row(label: str, agg: RunAggregate) -> str:
    cost = agg.candidate_cost.total
    return (
        f"{label:<26} ok {agg.responses_ok:>2}/{agg.total_cases:<2} "
        f"err {agg.candidate_errors:>2}  "
        f"correct {_fmt(agg.correctness.mean):>5} (n={agg.correctness.n:>2})  "
        f"grounded {_fmt(agg.groundedness.mean):>5} (n={agg.groundedness.n:>2})  "
        f"facts {_fmt(agg.required_facts.mean):>5} (n={agg.required_facts.n:>2})  "
        f"p50 {_fmt(agg.latency_ms_median, 'ms', 0):>6}  "
        f"cost {'N/A' if cost is None else f'${cost:.6f}'}"
    )


async def _create_and_run(
    settings: Settings,
    name: str,
    prompt: str,
    cases: list[TestCaseIn],
    models: list[str],
    baseline_id: str | None = None,
) -> str:
    factory = ProviderFactory(settings)
    pricing = latest_pricing()
    with session_factory()() as session:
        experiment = create_experiment(
            session,
            ExperimentCreate(name=name, system_prompt=prompt, cases=cases, baseline_experiment_id=baseline_id),
        )
        _, plans = prepare_runs(
            session, factory, pricing, experiment.id, [ModelRef(provider="sample", model_id=m) for m in models]
        )
        experiment_id = experiment.id
    await execute_runs(
        experiment_id,
        plans,
        session_factory=session_factory(),
        options=runner_options(settings),
        pricing=pricing,
        secrets=factory.secrets(),
    )
    return experiment_id


def _print_experiment(experiment: Experiment) -> None:
    print(f"Experiment {experiment.name!r} {experiment.id}  status={experiment.status.value}  "
          f"cases={len(experiment.test_cases)}")
    for run in experiment.model_runs:
        print("  " + _row(f"{run.provider}/{run.model_id}", aggregate_run(run.results)) + f"  [{run.status.value}]")


async def main(dataset_path: Path, name: str, compare: bool) -> None:
    settings = get_settings().model_copy(update={"sample_mode": True})
    configure_engine(settings.database_url)
    init_db()
    cases = DatasetIn.model_validate(json.loads(dataset_path.read_text())).cases

    print("SAMPLE MODE - offline heuristic models and judge, simulated latency, fictional prices.")
    print("These numbers exercise the pipeline only; they say nothing about real LLMs.")
    print(f"Database: {settings.database_url}\n")

    if not compare:
        experiment_id = await _create_and_run(settings, name, PROMPT_V1, cases, list(SAMPLE_MODELS))
        with session_factory()() as session:
            _print_experiment(session.get_one(Experiment, experiment_id))
        return

    models = ["sample-extractive", "sample-terse"]
    baseline_id = await _create_and_run(settings, f"{name} - prompt v1 (baseline)", PROMPT_V1, cases, models)
    candidate_id = await _create_and_run(
        settings, f"{name} - prompt v2 (one sentence)", PROMPT_V2, cases, models, baseline_id
    )
    with session_factory()() as session:
        baseline = session.get_one(Experiment, baseline_id)
        candidate = session.get_one(Experiment, candidate_id)
        _print_experiment(baseline)
        _print_experiment(candidate)
        comparison = compare_experiments(candidate, baseline)
        print(f"\nComparison vs baseline (threshold {comparison.threshold_pp:g} pp, "
              f"prompt changed: {comparison.prompt_changed})")
        for model in comparison.models:
            print(f"  {model.provider}/{model.model_id}: {model.verdict.upper()}")
            for reason in model.reasons:
                print(f"    - {reason}")
            for metric in model.metrics:
                print(f"    {metric.metric:<13} baseline {_fmt(metric.baseline_mean):>5}  "
                      f"candidate {_fmt(metric.candidate_mean):>5}  "
                      f"delta {_fmt(metric.delta_pp, ' pp'):>8}  paired {metric.paired_cases}/{metric.applicable_cases}"
                      f"  -> {metric.verdict}")
            for info in model.informational:
                print(f"    {info.name:<21} baseline {_fmt(info.baseline, digits=4):>9}  "
                      f"candidate {_fmt(info.candidate, digits=4):>9}  (informational, not a verdict)")
        print(f"\n{comparison.note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--name", default="Sample setup run")
    parser.add_argument("--compare", action="store_true", help="run a prompt v1 vs v2 regression comparison")
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.name, args.compare))
