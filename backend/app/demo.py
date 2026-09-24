"""SAMPLE MODE setup run: `python -m app.demo` (from backend/).

Creates an experiment from datasets/customer_support.json, runs the offline sample
models through the real runner/scoring/persistence path, and prints per-model
aggregates. No API keys are used and no real LLM is called.
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.db.models import Experiment
from app.db.session import configure_engine, init_db, session_factory
from app.evaluation.aggregate import RunAggregate, aggregate_run
from app.evaluation.pricing import latest_pricing
from app.evaluation.runner import execute_runs
from app.providers.registry import ProviderFactory
from app.providers.sample_provider import SAMPLE_MODELS
from app.schemas import DatasetIn, ExperimentCreate, ModelRef
from app.services import create_experiment, prepare_runs, runner_options

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "customer_support.json"
DEFAULT_PROMPT = (
    "You are a customer-support assistant for Northwind Outfitters. Answer using only the policy context "
    "provided. If the context does not answer the question, say so and suggest contacting support."
)


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


async def main(dataset_path: Path, name: str) -> str:
    settings = get_settings().model_copy(update={"sample_mode": True})
    configure_engine(settings.database_url)
    init_db()
    factory = ProviderFactory(settings)
    pricing = latest_pricing()
    dataset = DatasetIn.model_validate(json.loads(dataset_path.read_text()))

    with session_factory()() as session:
        experiment = create_experiment(
            session, ExperimentCreate(name=name, system_prompt=DEFAULT_PROMPT, cases=dataset.cases)
        )
        _, plans = prepare_runs(
            session, factory, pricing, experiment.id, [ModelRef(provider="sample", model_id=m) for m in SAMPLE_MODELS]
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

    with session_factory()() as session:
        experiment = session.get_one(Experiment, experiment_id)
        print("SAMPLE MODE - offline heuristic models and judge, simulated latency, fictional prices.")
        print("These numbers exercise the pipeline only; they say nothing about real LLMs.\n")
        print(f"Experiment {experiment.id}  status={experiment.status.value}  cases={len(experiment.test_cases)}")
        print(f"Database: {settings.database_url}\n")
        for run in experiment.model_runs:
            print(_row(f"{run.provider}/{run.model_id}", aggregate_run(run.results)) + f"  [{run.status.value}]")
    return experiment_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--name", default="Sample setup run")
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.name))
