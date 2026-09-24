"""Baseline comparison and regression detection.

A verdict is only given when the comparison is apples to apples:
- same test cases (content hash) in baseline and candidate experiments,
- same provider/model on both sides,
- identical evaluation config (judge, templates, token limits, ...),
- every applicable case scored on both sides (otherwise no verdict).
The system prompt is expected to differ; that is what a prompt-regression check measures.
"""

from datetime import datetime
from statistics import mean, median
from typing import Any, Literal

from pydantic import BaseModel

from app.db.models import EvaluationResult, Experiment, ModelRun, RunStatus

FINISHED = (RunStatus.completed, RunStatus.partial)
SMALL_SAMPLE = 30

Metric = Literal["correctness", "groundedness"]
MetricVerdict = Literal["regression", "ok", "insufficient_data", "not_applicable"]
ModelVerdict = Literal["regression", "ok", "insufficient_data", "not_comparable"]


class MetricComparison(BaseModel):
    metric: Metric
    applicable_cases: int
    paired_cases: int
    baseline_mean: float | None
    candidate_mean: float | None
    delta_pp: float | None
    verdict: MetricVerdict
    note: str | None = None


class InfoDelta(BaseModel):
    """Reported alongside the verdict but never used to flag a regression."""

    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    paired_cases: int


class CaseDrop(BaseModel):
    case_id: str
    metric: Metric
    baseline: float
    candidate: float
    delta_pp: float


class ModelComparison(BaseModel):
    provider: str
    model_id: str
    verdict: ModelVerdict
    reasons: list[str]
    baseline_run_id: str | None
    candidate_run_id: str | None
    metrics: list[MetricComparison]
    informational: list[InfoDelta]
    largest_drops: list[CaseDrop]


class ComparisonOut(BaseModel):
    experiment_id: str
    baseline_experiment_id: str | None
    threshold_pp: float
    comparable: bool
    reasons: list[str]
    prompt_changed: bool | None
    has_regression: bool | None  # None = no verdict could be reached
    models: list[ModelComparison]
    note: str


def _run_time(run: ModelRun) -> datetime:
    # SQLite returns naive datetimes; compare everything naive (all stored as UTC).
    return (run.started_at or run.completed_at or datetime.min).replace(tzinfo=None)


def latest_finished_runs(experiment: Experiment) -> dict[tuple[str, str], ModelRun]:
    """Most recent completed/partial run per (provider, model_id)."""
    latest: dict[tuple[str, str], ModelRun] = {}
    for run in experiment.model_runs:
        if run.status not in FINISHED:
            continue
        key = (run.provider, run.model_id)
        current = latest.get(key)
        if current is None or _run_time(run) > _run_time(current):
            latest[key] = run
    return latest


def _case_fields(experiment: Experiment) -> dict[str, dict[str, Any]]:
    return {
        c.case_id: {
            "input": c.input,
            "context": c.context,
            "expected_answer": c.expected_answer,
            "required_facts": list(c.required_facts),
            "metadata": dict(c.case_metadata),
        }
        for c in experiment.test_cases
    }


def dataset_differences(baseline: Experiment, candidate: Experiment) -> list[str]:
    if baseline.dataset_hash == candidate.dataset_hash:
        return []
    base, cand = _case_fields(baseline), _case_fields(candidate)
    reasons = ["the two experiments do not use the same test cases"]
    if removed := sorted(base.keys() - cand.keys()):
        reasons.append(f"only in baseline: {', '.join(removed[:10])}{' …' if len(removed) > 10 else ''}")
    if added := sorted(cand.keys() - base.keys()):
        reasons.append(f"only in candidate: {', '.join(added[:10])}{' …' if len(added) > 10 else ''}")
    changed = sorted(case_id for case_id in base.keys() & cand.keys() if base[case_id] != cand[case_id])
    if changed:
        reasons.append(f"same id, different content: {', '.join(changed[:10])}{' …' if len(changed) > 10 else ''}")
    return reasons


def _config_differences(baseline: ModelRun, candidate: ModelRun) -> list[str]:
    if baseline.eval_config_hash == candidate.eval_config_hash:
        return []
    keys = sorted(set(baseline.eval_config) | set(candidate.eval_config))
    diffs = [
        f"{key}: baseline={baseline.eval_config.get(key)!r}, candidate={candidate.eval_config.get(key)!r}"
        for key in keys
        if baseline.eval_config.get(key) != candidate.eval_config.get(key)
    ]
    return ["evaluation settings differ: " + "; ".join(diffs or ["config hash mismatch"])]


def _by_case(run: ModelRun) -> dict[str, EvaluationResult]:
    return {result.test_case.case_id: result for result in run.results}


def _score(result: EvaluationResult | None, metric: Metric) -> float | None:
    if result is None or result.error is not None:
        return None
    return result.correctness_score if metric == "correctness" else result.groundedness_score


def _compare_metric(
    metric: Metric,
    applicable: list[str],
    base: dict[str, EvaluationResult],
    cand: dict[str, EvaluationResult],
    threshold: float,
) -> tuple[MetricComparison, list[CaseDrop]]:
    pairs = [
        (case_id, b, c)
        for case_id in applicable
        if (b := _score(base.get(case_id), metric)) is not None and (c := _score(cand.get(case_id), metric)) is not None
    ]
    base_mean = mean(b for _, b, _ in pairs) if pairs else None
    cand_mean = mean(c for _, _, c in pairs) if pairs else None
    delta = cand_mean - base_mean if base_mean is not None and cand_mean is not None else None
    drops = [CaseDrop(case_id=i, metric=metric, baseline=b, candidate=c, delta_pp=c - b) for i, b, c in pairs if c < b]

    note = None
    if not applicable:
        verdict: MetricVerdict = "not_applicable"
        note = "no case has context" if metric == "groundedness" else "no cases"
    elif len(pairs) < len(applicable):
        verdict = "insufficient_data"
        note = (
            f"{len(applicable) - len(pairs)} of {len(applicable)} applicable cases lack a score in the baseline "
            "or candidate (request or judge error); no verdict is given on incomplete pairs"
        )
    elif delta is not None and delta < -threshold:
        verdict = "regression"
    else:
        verdict = "ok"
    comparison = MetricComparison(
        metric=metric,
        applicable_cases=len(applicable),
        paired_cases=len(pairs),
        baseline_mean=base_mean,
        candidate_mean=cand_mean,
        delta_pp=delta,
        verdict=verdict,
        note=note,
    )
    return comparison, drops


def _info(name: str, base_values: list[float], cand_values: list[float], agg: Any, paired: int) -> InfoDelta:
    b = agg(base_values) if base_values else None
    c = agg(cand_values) if cand_values else None
    return InfoDelta(name=name, baseline=b, candidate=c, delta=None if b is None or c is None else c - b, paired_cases=paired)


def _informational(case_ids: list[str], base: dict[str, EvaluationResult], cand: dict[str, EvaluationResult]) -> list[InfoDelta]:
    ok_pairs = [
        (base[i], cand[i]) for i in case_ids
        if i in base and i in cand and base[i].error is None and cand[i].error is None
    ]

    def paired_values(attr: str) -> tuple[list[float], list[float], int]:
        pairs = [(getattr(b, attr), getattr(c, attr)) for b, c in ok_pairs]
        pairs = [(b, c) for b, c in pairs if b is not None and c is not None]
        return [b for b, _ in pairs], [c for _, c in pairs], len(pairs)

    facts_b, facts_c, facts_n = paired_values("required_facts_score")
    lat_b, lat_c, lat_n = paired_values("latency_ms")
    cost_b, cost_c, cost_n = paired_values("estimated_cost")
    return [
        _info("required_facts_mean", facts_b, facts_c, mean, facts_n),
        _info("latency_ms_median", lat_b, lat_c, median, lat_n),
        _info("candidate_cost_total", cost_b, cost_c, sum, cost_n),
    ]


def compare_model(
    baseline_run: ModelRun | None, candidate_run: ModelRun, candidate: Experiment, threshold: float
) -> ModelComparison:
    reasons: list[str] = []
    if baseline_run is None:
        reasons.append("the baseline has no finished run for this model")
    else:
        reasons.extend(_config_differences(baseline_run, candidate_run))
    if reasons or baseline_run is None:
        return ModelComparison(
            provider=candidate_run.provider, model_id=candidate_run.model_id, verdict="not_comparable",
            reasons=reasons, baseline_run_id=baseline_run.id if baseline_run else None,
            candidate_run_id=candidate_run.id, metrics=[], informational=[], largest_drops=[],
        )

    base, cand = _by_case(baseline_run), _by_case(candidate_run)
    all_cases = [c.case_id for c in candidate.test_cases]
    with_context = [c.case_id for c in candidate.test_cases if c.context is not None]
    correctness, drops_c = _compare_metric("correctness", all_cases, base, cand, threshold)
    groundedness, drops_g = _compare_metric("groundedness", with_context, base, cand, threshold)
    metrics = [correctness, groundedness]

    verdicts = {m.verdict for m in metrics}
    verdict: ModelVerdict
    if "regression" in verdicts:
        verdict = "regression"
    elif "insufficient_data" in verdicts:
        verdict = "insufficient_data"
    else:
        verdict = "ok"
    return ModelComparison(
        provider=candidate_run.provider,
        model_id=candidate_run.model_id,
        verdict=verdict,
        reasons=[],
        baseline_run_id=baseline_run.id,
        candidate_run_id=candidate_run.id,
        metrics=metrics,
        informational=_informational(all_cases, base, cand),
        largest_drops=sorted(drops_c + drops_g, key=lambda d: d.delta_pp)[:10],
    )


def compare_experiments(
    candidate: Experiment, baseline: Experiment | None, provider: str | None = None, model_id: str | None = None
) -> ComparisonOut:
    threshold = candidate.regression_threshold
    note = (
        f"Verdicts use the candidate's threshold of {threshold:g} percentage points on mean judge scores over paired "
        f"cases. With fewer than {SMALL_SAMPLE} cases and non-deterministic models and judges, treat differences as "
        "indicative, not statistically significant."
    )

    def result(comparable: bool, reasons: list[str], models: list[ModelComparison]) -> ComparisonOut:
        verdicts = [m.verdict for m in models]
        has_regression: bool | None
        if "regression" in verdicts:
            has_regression = True
        elif verdicts and all(v == "ok" for v in verdicts):
            has_regression = False
        else:
            has_regression = None
        return ComparisonOut(
            experiment_id=candidate.id,
            baseline_experiment_id=candidate.baseline_experiment_id,
            threshold_pp=threshold,
            comparable=comparable,
            reasons=reasons,
            prompt_changed=None if baseline is None else baseline.prompt_hash != candidate.prompt_hash,
            has_regression=has_regression,
            models=models,
            note=note,
        )

    if candidate.baseline_experiment_id is None:
        return result(False, ["no baseline experiment is set for this experiment"], [])
    if baseline is None:
        return result(False, ["the baseline experiment no longer exists"], [])
    if reasons := dataset_differences(baseline, candidate):
        return result(False, reasons, [])

    candidate_runs = latest_finished_runs(candidate)
    baseline_runs = latest_finished_runs(baseline)
    selected = [
        run for (p, m), run in sorted(candidate_runs.items())
        if (provider is None or p == provider) and (model_id is None or m == model_id)
    ]
    if not selected:
        return result(False, ["the candidate has no finished run" + (" for the selected model" if model_id else "")], [])
    models = [
        compare_model(baseline_runs.get((run.provider, run.model_id)), run, candidate, threshold) for run in selected
    ]
    return result(any(m.verdict != "not_comparable" for m in models), [], models)
