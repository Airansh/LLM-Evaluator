"""Per-run aggregates over successful, scored cases, always reported with denominators."""

from collections.abc import Iterable
from statistics import mean, median

from pydantic import BaseModel

from app.db.models import EvaluationResult


class MetricSummary(BaseModel):
    mean: float | None
    n: int  # number of cases contributing to the mean


class CostSummary(BaseModel):
    total: float | None  # None when no result had a measurable cost
    n_priced: int
    complete: bool  # True when every result that reported usage also had a price


class RunAggregate(BaseModel):
    total_cases: int
    responses_ok: int
    candidate_errors: int
    judge_errors: int
    correctness: MetricSummary
    groundedness: MetricSummary
    required_facts: MetricSummary
    latency_ms_mean: float | None
    latency_ms_median: float | None
    input_tokens: int | None
    output_tokens: int | None
    candidate_cost: CostSummary
    judge_cost: CostSummary


def _metric(values: Iterable[float | None]) -> MetricSummary:
    present = [value for value in values if value is not None]
    return MetricSummary(mean=mean(present) if present else None, n=len(present))


def _cost(costs: list[float | None], had_usage: list[bool]) -> CostSummary:
    priced = [cost for cost in costs if cost is not None]
    expected = sum(had_usage)
    return CostSummary(
        total=sum(priced) if priced else None,
        n_priced=len(priced),
        complete=expected > 0 and len(priced) == expected,
    )


def _token_total(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def aggregate_run(results: list[EvaluationResult]) -> RunAggregate:
    ok = [r for r in results if r.error is None]
    latencies = [r.latency_ms for r in ok if r.latency_ms is not None]
    return RunAggregate(
        total_cases=len(results),
        responses_ok=len(ok),
        candidate_errors=len(results) - len(ok),
        judge_errors=sum(1 for r in ok if r.judge_error is not None),
        correctness=_metric(r.correctness_score for r in ok),
        groundedness=_metric(r.groundedness_score for r in ok),
        required_facts=_metric(r.required_facts_score for r in ok),
        latency_ms_mean=mean(latencies) if latencies else None,
        latency_ms_median=median(latencies) if latencies else None,
        input_tokens=_token_total([r.input_tokens for r in results]),
        output_tokens=_token_total([r.output_tokens for r in results]),
        candidate_cost=_cost(
            [r.estimated_cost for r in results],
            [r.input_tokens is not None and r.output_tokens is not None for r in results],
        ),
        judge_cost=_cost(
            [r.judge_cost for r in ok],
            [r.judge_input_tokens is not None and r.judge_output_tokens is not None for r in ok],
        ),
    )
