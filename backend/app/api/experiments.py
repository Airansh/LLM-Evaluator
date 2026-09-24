from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_pricing, get_provider_factory
from app.db.models import Experiment, ModelRun
from app.db.session import get_session, session_factory
from app.evaluation.aggregate import aggregate_run
from app.evaluation.pricing import PricingTable
from app.evaluation.runner import execute_runs
from app.providers.base import SAMPLE_PROVIDER
from app.providers.registry import ProviderFactory
from app.schemas import ExperimentCreate, ExperimentOut, RunOut, RunRequest, RunStartedOut
from app.services import Conflict, InvalidRequest, NotFound, create_experiment, prepare_runs, runner_options

router = APIRouter(prefix="/experiments", tags=["experiments"])


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


@router.get("/{experiment_id}", response_model=ExperimentOut)
def detail(experiment_id: str, session: Session = Depends(get_session)) -> ExperimentOut:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"experiment {experiment_id!r} not found")
    return experiment_out(experiment)


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
