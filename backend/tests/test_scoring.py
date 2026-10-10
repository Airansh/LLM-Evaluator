import pytest

from app.evaluation.aggregate import aggregate_run
from app.evaluation.coverage import missing_facts, required_facts_score
from app.evaluation.pricing import ModelPrice, PricingTable, latest_pricing
from app.evaluation.retry import complete_with_retry
from app.db.models import EvaluationResult

from tests.fakes import fail, no_sleep, ok


# ---- required-fact coverage ----------------------------------------------------------------


def test_coverage_normalizes_case_and_whitespace() -> None:
    assert required_facts_score("Returns within 30   DAYS only.", ["30 days"]) == 100.0


def test_coverage_is_fractional_and_lists_missing() -> None:
    facts = ["$7", "restocking fee", "receipt"]
    response = "A $7 restocking fee applies."
    assert required_facts_score(response, facts) == pytest.approx(200 / 3)
    assert missing_facts(response, facts) == ["receipt"]


def test_coverage_empty_fact_list_is_not_applicable() -> None:
    assert required_facts_score("anything", []) is None


def test_coverage_is_lexical_not_semantic() -> None:
    # "thirty days" means the same thing but is not a lexical match; documented limitation.
    assert required_facts_score("Returns within thirty days.", ["30 days"]) == 0.0
    # Punctuation is kept, so a bare number doesn't match a dollar amount.
    assert required_facts_score("The fee is 7.", ["$7"]) == 0.0


# ---- pricing -------------------------------------------------------------------------------


def _table() -> PricingTable:
    price = ModelPrice(input=2.0, output=10.0, as_of="2026-06-24", source="test")
    return PricingTable(version="test", models={"anthropic/m": price})


def test_price_estimate() -> None:
    assert _table().estimate("anthropic", "m", 1_000_000, 100_000) == pytest.approx(3.0)


def test_missing_price_or_usage_is_unavailable() -> None:
    table = _table()
    assert table.estimate("openai", "unknown", 10, 10) is None
    assert table.estimate("anthropic", "m", None, 10) is None
    assert table.estimate("anthropic", "m", 10, None) is None


def test_shipped_pricing_file_loads_and_has_as_of_dates() -> None:
    table = latest_pricing()
    assert table.version >= "2026-09-24"
    assert table.price("anthropic", "claude-sonnet-5") is not None
    # No price was verified for these, so cost must show as unavailable rather than guessed.
    assert table.price("openai", "gpt-6-luna") is None
    assert all(price.as_of for price in table.models.values())


# ---- retries -------------------------------------------------------------------------------


async def test_retries_transient_then_succeeds() -> None:
    responses = [fail("openai", "m", "rate_limit"), fail("openai", "m", "server"), ok("openai", "m", "hi")]

    async def call():  # noqa: ANN202
        return responses.pop(0)

    result, attempts = await complete_with_retry(call, max_attempts=3, sleep=no_sleep)
    assert result.ok and attempts == 3


async def test_does_not_retry_permanent_errors() -> None:
    calls = 0

    async def call():  # noqa: ANN202
        nonlocal calls
        calls += 1
        return fail("openai", "m", "auth")

    result, attempts = await complete_with_retry(call, max_attempts=3, sleep=no_sleep)
    assert not result.ok and attempts == 1 and calls == 1


async def test_retry_attempts_are_bounded() -> None:
    async def call():  # noqa: ANN202
        return fail("openai", "m", "timeout")

    result, attempts = await complete_with_retry(call, max_attempts=2, sleep=no_sleep)
    assert result.error_kind == "timeout" and attempts == 2


# ---- aggregation ---------------------------------------------------------------------------


def _result(**fields) -> EvaluationResult:  # noqa: ANN003
    return EvaluationResult(model_run_id="r", test_case_pk=1, **fields)


def test_aggregate_ignores_missing_scores_and_reports_denominators() -> None:
    results = [
        _result(correctness_score=90, groundedness_score=None, required_facts_score=100, latency_ms=100,
                input_tokens=10, output_tokens=5, estimated_cost=0.01),
        _result(correctness_score=70, groundedness_score=80, required_facts_score=None, latency_ms=300,
                input_tokens=10, output_tokens=5, estimated_cost=0.02),
        _result(judge_error="malformed", required_facts_score=50, latency_ms=200,
                input_tokens=10, output_tokens=5, estimated_cost=None),
        _result(error="[server] down", latency_ms=5),
    ]
    agg = aggregate_run(results)
    assert (agg.total_cases, agg.responses_ok, agg.candidate_errors, agg.judge_errors) == (4, 3, 1, 1)
    assert agg.correctness.mean == 80 and agg.correctness.n == 2
    assert agg.groundedness.mean == 80 and agg.groundedness.n == 1
    assert agg.required_facts.mean == 75 and agg.required_facts.n == 2
    assert agg.latency_ms_median == 200  # failed calls excluded
    assert agg.candidate_cost.total == pytest.approx(0.03)
    assert agg.candidate_cost.n_priced == 2 and agg.candidate_cost.complete is False


def test_aggregate_with_no_scores_is_none_not_zero() -> None:
    agg = aggregate_run([_result(error="[auth] bad key")])
    assert agg.correctness.mean is None and agg.correctness.n == 0
    assert agg.candidate_cost.total is None
