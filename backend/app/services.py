"""Experiment creation and run preparation, shared by the API and the demo script."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Experiment, ExperimentStatus, ModelRun, RunStatus, TestCase, utcnow
from app.evaluation.fingerprint import config_hash, dataset_hash, prompt_hash
from app.evaluation.judge import Judge
from app.evaluation.pricing import PricingTable
from app.evaluation.prompts import CANDIDATE_TEMPLATE_VERSION, JUDGE_PROMPT_VERSION
from app.evaluation.runner import RunnerOptions, RunPlan
from app.providers.base import Provider
from app.providers.registry import ProviderFactory, Unavailable
from app.schemas import ExperimentCreate, ModelRef

SCORING_VERSION = 1
ACTIVE_RUN_STATUSES = (RunStatus.pending, RunStatus.running)


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class InvalidRequest(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def create_experiment(session: Session, payload: ExperimentCreate) -> Experiment:
    if payload.baseline_experiment_id and session.get(Experiment, payload.baseline_experiment_id) is None:
        raise InvalidRequest([f"baseline experiment {payload.baseline_experiment_id!r} does not exist"])
    experiment = Experiment(
        name=payload.name,
        system_prompt=payload.system_prompt,
        baseline_experiment_id=payload.baseline_experiment_id,
        regression_threshold=payload.regression_threshold,
        dataset_hash=dataset_hash(payload.cases),
        prompt_hash=prompt_hash(payload.system_prompt),
        test_cases=[
            TestCase(
                case_id=case.id,
                position=position,
                input=case.input,
                context=case.context,
                expected_answer=case.expected_answer,
                required_facts=case.required_facts,
                case_metadata=case.metadata,
            )
            for position, case in enumerate(payload.cases)
        ],
    )
    session.add(experiment)
    session.commit()
    return experiment


def eval_config(settings: Settings, judge: Judge) -> dict[str, Any]:
    """Everything that affects scores. Two runs are only comparable if this is identical."""
    return {
        "scoring_version": SCORING_VERSION,
        "candidate_template_version": CANDIDATE_TEMPLATE_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_provider": judge.provider,
        "judge_model_id": judge.model_id,
        "fact_matching": "substring after NFKC + casefold + whitespace collapse",
        "candidate_max_output_tokens": settings.candidate_max_output_tokens,
        "judge_max_output_tokens": settings.judge_max_output_tokens,
        "temperature": settings.temperature,
    }


def prepare_runs(
    session: Session,
    factory: ProviderFactory,
    pricing: PricingTable,
    experiment_id: str,
    models: list[ModelRef],
) -> tuple[Experiment, list[RunPlan]]:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id!r} not found")
    if any(run.status in ACTIVE_RUN_STATUSES for run in experiment.model_runs):
        raise Conflict("this experiment already has a run in progress")

    problems: list[str] = []
    resolved: list[tuple[ModelRef, Provider, Judge]] = []
    for ref in models:
        reason = factory.unavailable_reason(ref.provider, ref.model_id)
        if reason is not None:
            problems.append(f"{ref.provider}/{ref.model_id}: {reason}")
            continue
        try:
            resolved.append((ref, factory.provider(ref.provider), factory.judge_for(ref.provider)))
        except Unavailable as exc:
            problems.append(f"{ref.provider}/{ref.model_id}: {exc}")
    if problems:
        raise InvalidRequest(problems)

    plans: list[RunPlan] = []
    for ref, provider, judge in resolved:
        config = eval_config(factory.settings, judge)
        run = ModelRun(
            experiment=experiment,
            provider=ref.provider,
            model_id=ref.model_id,
            status=RunStatus.pending,
            judge_provider=judge.provider,
            judge_model_id=judge.model_id,
            eval_config=config,
            eval_config_hash=config_hash(config),
            pricing_version=pricing.version,
        )
        session.add(run)
        session.flush()
        plans.append(RunPlan(run_id=run.id, provider=provider, model_id=ref.model_id, judge=judge))
    experiment.status = ExperimentStatus.running
    session.commit()
    return experiment, plans


def runner_options(settings: Settings) -> RunnerOptions:
    return RunnerOptions(
        max_concurrency=settings.max_concurrency,
        max_attempts=settings.max_attempts,
        max_output_tokens=settings.candidate_max_output_tokens,
        temperature=settings.temperature,
    )


def refresh_experiment_status(experiment: Experiment) -> None:
    statuses = [run.status for run in experiment.model_runs]
    if not statuses:
        experiment.status = ExperimentStatus.draft
    elif any(status in ACTIVE_RUN_STATUSES for status in statuses):
        experiment.status = ExperimentStatus.running
    elif all(status == RunStatus.completed for status in statuses):
        experiment.status = ExperimentStatus.completed
    elif all(status == RunStatus.failed for status in statuses):
        experiment.status = ExperimentStatus.failed
    else:
        experiment.status = ExperimentStatus.partial


def recover_interrupted_runs(session: Session) -> int:
    """At startup, runs still pending/running were cut off by a restart; mark them failed."""
    runs = session.scalars(select(ModelRun).where(ModelRun.status.in_(ACTIVE_RUN_STATUSES))).all()
    for run in runs:
        run.status = RunStatus.failed
        run.error = "interrupted: the server stopped before this run finished; results so far are kept"
        run.completed_at = utcnow()
    for experiment in {run.experiment for run in runs}:
        refresh_experiment_status(experiment)
    session.commit()
    return len(runs)
