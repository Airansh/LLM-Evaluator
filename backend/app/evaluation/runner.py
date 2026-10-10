"""Bounded async execution of every (test case × model run) pair.

Each result is persisted as soon as it is scored, so one failing call (or a crash midway)
never discards the results that already completed.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import EvaluationResult, Experiment, ModelRun, RunStatus, utcnow
from app.evaluation.coverage import required_facts_score
from app.evaluation.judge import Judge, JudgeCase
from app.evaluation.pricing import PricingTable
from app.evaluation.prompts import build_candidate_message
from app.evaluation.retry import Sleep, complete_with_retry
from app.providers.base import CompletionRequest, Provider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    provider: Provider
    model_id: str
    judge: Judge


@dataclass(frozen=True)
class RunnerOptions:
    max_concurrency: int
    max_attempts: int
    max_output_tokens: int
    temperature: float | None


@dataclass(frozen=True)
class _Case:
    pk: int
    case_id: str
    input: str
    context: str | None
    expected_answer: str
    required_facts: list[str]


def redact(text: str | None, secrets: list[str]) -> str | None:
    if text is None:
        return None
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


async def execute_runs(
    experiment_id: str,
    plans: list[RunPlan],
    *,
    session_factory: sessionmaker[Session],
    options: RunnerOptions,
    pricing: PricingTable,
    secrets: list[str],
    sleep: Sleep = asyncio.sleep,
) -> None:
    # Imported here to avoid a cycle (services imports this module).
    from app.services import refresh_experiment_status

    with session_factory() as session:
        experiment = session.get_one(Experiment, experiment_id)
        system_prompt = experiment.system_prompt
        cases = [
            _Case(c.pk, c.case_id, c.input, c.context, c.expected_answer, list(c.required_facts))
            for c in experiment.test_cases
        ]
        for plan in plans:
            run = session.get_one(ModelRun, plan.run_id)
            run.status, run.started_at = RunStatus.running, utcnow()
        session.commit()

    semaphore = asyncio.Semaphore(options.max_concurrency)

    async def evaluate(plan: RunPlan, case: _Case) -> None:
        async with semaphore:
            try:
                fields = await _evaluate_case(plan, case, system_prompt, options, pricing, sleep)
            except Exception as exc:  # noqa: BLE001 - persist, don't lose the rest of the run
                logger.exception("unexpected error evaluating %s on %s", case.case_id, plan.model_id)
                fields = {"error": f"[internal] {type(exc).__name__}: {exc}"}
        fields["error"] = redact(fields.get("error"), secrets)
        fields["judge_error"] = redact(fields.get("judge_error"), secrets)
        with session_factory() as session:
            session.add(EvaluationResult(model_run_id=plan.run_id, test_case_pk=case.pk, **fields))
            session.commit()

    try:
        await asyncio.gather(*(evaluate(plan, case) for plan in plans for case in cases))
    finally:
        with session_factory() as session:
            for plan in plans:
                _finalize_run(session.get_one(ModelRun, plan.run_id))
            refresh_experiment_status(session.get_one(Experiment, experiment_id))
            session.commit()


async def _evaluate_case(
    plan: RunPlan,
    case: _Case,
    system_prompt: str,
    options: RunnerOptions,
    pricing: PricingTable,
    sleep: Sleep,
) -> dict[str, Any]:
    request = CompletionRequest(
        system_prompt=system_prompt,
        user_message=build_candidate_message(case.input, case.context),
        max_output_tokens=options.max_output_tokens,
        temperature=options.temperature,
    )
    response, attempts = await complete_with_retry(
        lambda: plan.provider.complete(plan.model_id, request), max_attempts=options.max_attempts, sleep=sleep
    )
    fields: dict[str, Any] = {
        "latency_ms": response.latency_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "estimated_cost": pricing.estimate(
            plan.provider.name, plan.model_id, response.input_tokens, response.output_tokens
        ),
    }
    if not response.ok:
        suffix = f" (after {attempts} attempts)" if attempts > 1 else ""
        fields["error"] = f"[{response.error_kind}] {response.error}{suffix}"
        return fields

    text = response.text or ""
    fields["response"] = text
    fields["required_facts_score"] = required_facts_score(text, case.required_facts)

    outcome = await plan.judge.evaluate(JudgeCase(case.input, case.context, case.expected_answer), text)
    fields.update(
        correctness_score=outcome.correctness,
        groundedness_score=outcome.groundedness,
        judge_reason=outcome.reason,
        judge_error=outcome.error,
        judge_input_tokens=outcome.input_tokens,
        judge_output_tokens=outcome.output_tokens,
        judge_cost=pricing.estimate(
            plan.judge.provider, plan.judge.model_id, outcome.input_tokens, outcome.output_tokens
        ),
    )
    return fields


def _finalize_run(run: ModelRun) -> None:
    results = run.results
    candidate_errors = sum(1 for r in results if r.error is not None)
    judge_errors = sum(1 for r in results if r.error is None and r.judge_error is not None)
    expected = len(run.experiment.test_cases)

    if not results or candidate_errors == len(results):
        run.status = RunStatus.failed
        run.error = run.error or ("no results were recorded" if not results else "every request failed")
    elif candidate_errors or judge_errors or len(results) < expected:
        run.status = RunStatus.partial
    else:
        run.status = RunStatus.completed
    if len(results) < expected:
        run.error = f"{expected - len(results)} of {expected} cases have no result"
    run.completed_at = utcnow()
