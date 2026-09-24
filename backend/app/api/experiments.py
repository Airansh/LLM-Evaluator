from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_pricing, get_provider_factory
from app.db.models import EvaluationResult, Experiment, ModelRun
from app.db.session import get_session, session_factory
from app.evaluation.aggregate import aggregate_run
from app.evaluation.compare import ComparisonOut, compare_experiments
from app.evaluation.coverage import missing_facts
from app.evaluation.pricing import PricingTable
from app.evaluation.runner import execute_runs
from app.providers.base import SAMPLE_PROVIDER
from app.providers.registry import ProviderFactory
from app.schemas import (
    CaseOut,
    CaseResultOut,
    ExperimentCreate,
    ExperimentOut,
    ExperimentSummary,
    ResultsOut,
    RunOut,
    RunRequest,
    RunStartedOut,
)
from app.services import Conflict, InvalidRequest, NotFound, create_experiment, prepare_runs, runner_options

router = APIRouter(prefix="/experiments", tags=["experiments"])

# Display-only highlight for the failure inspector; it does not affect scores or verdicts.
LOW_SCORE_THRESHOLD = 60.0


def run_out(run: ModelRun, with_aggregate: bool = False) -> RunOut:
    return RunOut(
        id=run.id,
        provider=run.provider,  # type: ignore[arg-type]
        model_id=run.model_id,
        status=run.status.value,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error=run.error,
        judge_provider=run.judge_provider,
        judge_model_id=run.judge_model_id,
        eval_config=run.eval_config,
        eval_config_hash=run.eval_config_hash,
        pricing_version=run.pricing_version,
        sample=run.provider == SAMPLE_PROVIDER,
        aggregate=aggregate_run(run.results) if with_aggregate else None,
    )


def experiment_out(experiment: Experiment, with_aggregates: bool = True) -> ExperimentOut:
    return ExperimentOut(
        id=experiment.id,
        name=experiment.name,
        system_prompt=experiment.system_prompt,
        created_at=experiment.created_at,
        baseline_experiment_id=experiment.baseline_experiment_id,
        regression_threshold=experiment.regression_threshold,
        status=experiment.status.value,
        dataset_hash=experiment.dataset_hash,
        prompt_hash=experiment.prompt_hash,
        case_count=len(experiment.test_cases),
        runs=[run_out(run, with_aggregates) for run in experiment.model_runs],
    )


@router.post("", response_model=ExperimentOut, status_code=status.HTTP_201_CREATED)
def create(payload: ExperimentCreate, session: Session = Depends(get_session)) -> ExperimentOut:
    try:
        experiment = create_experiment(session, payload)
    except InvalidRequest as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.problems) from exc
    return experiment_out(experiment)


def _get_or_404(session: Session, experiment_id: str) -> Experiment:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"experiment {experiment_id!r} not found")
    return experiment


def _comparison(session: Session, experiment: Experiment, **filters: str | None) -> ComparisonOut:
    baseline = session.get(Experiment, experiment.baseline_experiment_id) if experiment.baseline_experiment_id else None
    return compare_experiments(experiment, baseline, **filters)


@router.get("", response_model=list[ExperimentSummary])
def list_experiments(session: Session = Depends(get_session)) -> list[ExperimentSummary]:
    """Newest first, with a regression indicator for experiments that have a baseline."""
    experiments = session.scalars(select(Experiment).order_by(Experiment.created_at.desc())).all()
    summaries = []
    for experiment in experiments:
        regression: Literal["regression", "ok", "no_verdict"] | None = None
        if experiment.baseline_experiment_id:
            verdict = _comparison(session, experiment).has_regression
            regression = "no_verdict" if verdict is None else ("regression" if verdict else "ok")
        summaries.append(
            ExperimentSummary(
                id=experiment.id,
                name=experiment.name,
                created_at=experiment.created_at,
                status=experiment.status.value,
                case_count=len(experiment.test_cases),
                run_count=len(experiment.model_runs),
                models=sorted({f"{run.provider}/{run.model_id}" for run in experiment.model_runs}),
                baseline_experiment_id=experiment.baseline_experiment_id,
                sample=any(run.provider == SAMPLE_PROVIDER for run in experiment.model_runs),
                regression=regression,
            )
        )
    return summaries


@router.get("/{experiment_id}", response_model=ExperimentOut)
def detail(experiment_id: str, session: Session = Depends(get_session)) -> ExperimentOut:
    return experiment_out(_get_or_404(session, experiment_id))


def _flags(result: EvaluationResult, missing: list[str]) -> list[str]:
    flags = []
    if result.error:
        flags.append("candidate_error")
    if result.judge_error:
        flags.append("judge_error")
    if missing:
        flags.append("missing_facts")
    if result.correctness_score is not None and result.correctness_score < LOW_SCORE_THRESHOLD:
        flags.append("low_correctness")
    if result.groundedness_score is not None and result.groundedness_score < LOW_SCORE_THRESHOLD:
        flags.append("low_groundedness")
    return flags


@router.get("/{experiment_id}/results", response_model=ResultsOut)
def results(
    experiment_id: str,
    case_id: str | None = None,
    failures_only: bool = Query(False, description="Only cases where some result has a flag"),
    session: Session = Depends(get_session),
) -> ResultsOut:
    """Individual results grouped by test case, with judge rationale, errors and failure flags."""
    experiment = _get_or_404(session, experiment_id)
    by_case: dict[int, list[CaseResultOut]] = {case.pk: [] for case in experiment.test_cases}
    for run in experiment.model_runs:
        for result in run.results:
            case = result.test_case
            missing = missing_facts(result.response, case.required_facts) if result.response is not None else []
            by_case[case.pk].append(
                CaseResultOut(
                    run_id=run.id,
                    provider=run.provider,  # type: ignore[arg-type]
                    model_id=run.model_id,
                    sample=run.provider == SAMPLE_PROVIDER,
                    run_status=run.status.value,
                    response=result.response,
                    correctness_score=result.correctness_score,
                    groundedness_score=result.groundedness_score,
                    required_facts_score=result.required_facts_score,
                    missing_facts=missing,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    estimated_cost=result.estimated_cost,
                    judge_provider=run.judge_provider,
                    judge_model_id=run.judge_model_id,
                    judge_reason=result.judge_reason,
                    judge_input_tokens=result.judge_input_tokens,
                    judge_output_tokens=result.judge_output_tokens,
                    judge_cost=result.judge_cost,
                    error=result.error,
                    judge_error=result.judge_error,
                    flags=_flags(result, missing),
                )
            )

    cases = []
    for case in experiment.test_cases:
        if case_id is not None and case.case_id != case_id:
            continue
        case_results = by_case[case.pk]
        if failures_only and not any(r.flags for r in case_results):
            continue
        cases.append(
            CaseOut(
                case_id=case.case_id,
                input=case.input,
                context=case.context,
                expected_answer=case.expected_answer,
                required_facts=list(case.required_facts),
                metadata=dict(case.case_metadata),
                results=case_results,
            )
        )
    if case_id is not None and not cases and not failures_only:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"test case {case_id!r} not found in this experiment")
    return ResultsOut(
        experiment_id=experiment.id,
        total_cases=len(experiment.test_cases),
        low_score_threshold=LOW_SCORE_THRESHOLD,
        cases=cases,
    )


@router.get("/{experiment_id}/compare", response_model=ComparisonOut)
def compare(
    experiment_id: str,
    provider: str | None = None,
    model_id: str | None = None,
    session: Session = Depends(get_session),
) -> ComparisonOut:
    """Paired, same-model comparison against the baseline; `comparable: false` explains why not."""
    return _comparison(session, _get_or_404(session, experiment_id), provider=provider, model_id=model_id)


@router.post("/{experiment_id}/run", response_model=RunStartedOut, status_code=status.HTTP_202_ACCEPTED)
def run(
    experiment_id: str,
    payload: RunRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    factory: ProviderFactory = Depends(get_provider_factory),
    pricing: PricingTable = Depends(get_pricing),
) -> RunStartedOut:
    """Validate the selection, create pending model runs, and execute them in the background."""
    try:
        experiment, plans = prepare_runs(session, factory, pricing, experiment_id, payload.models)
    except NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Conflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except InvalidRequest as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.problems) from exc

    background.add_task(
        execute_runs,
        experiment.id,
        plans,
        session_factory=session_factory(),
        options=runner_options(factory.settings),
        pricing=pricing,
        secrets=factory.secrets(),
        sleep=factory.sleep,
    )
    plan_ids = {plan.run_id for plan in plans}
    return RunStartedOut(
        experiment_id=experiment.id,
        status=experiment.status.value,
        runs=[run_out(r) for r in experiment.model_runs if r.id in plan_ids],
    )
