"""API contracts and dataset validation."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.evaluation.aggregate import RunAggregate
from app.evaluation.text import normalize_text
from app.providers.base import ProviderName

MAX_CASES = 200
MAX_PROMPT_CHARS = 8_000
MAX_FIELD_CHARS = 20_000
MAX_FACTS_PER_CASE = 20
CASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$"

MetadataValue = str | int | float | bool | None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TestCaseIn(_Strict):
    __test__ = False  # not a pytest test class

    id: str = Field(pattern=CASE_ID_PATTERN, description="Unique within the dataset, e.g. returns_001")
    input: str = Field(min_length=1, max_length=MAX_FIELD_CHARS)
    context: str | None = Field(default=None, max_length=MAX_FIELD_CHARS)
    expected_answer: str = Field(min_length=1, max_length=MAX_FIELD_CHARS)
    required_facts: list[str] = Field(default_factory=list, max_length=MAX_FACTS_PER_CASE)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("context")
    @classmethod
    def _blank_context_is_none(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("required_facts")
    @classmethod
    def _validate_facts(cls, facts: list[str]) -> list[str]:
        cleaned = [fact.strip() for fact in facts]
        if any(not fact for fact in cleaned):
            raise ValueError("required_facts must not contain empty strings")
        seen: set[str] = set()
        for fact in cleaned:
            key = normalize_text(fact)
            if key in seen:
                raise ValueError(f"duplicate required fact after normalization: {fact!r}")
            seen.add(key)
        return cleaned


def _check_unique_ids(cases: list[TestCaseIn]) -> list[TestCaseIn]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        if case.id in seen and case.id not in duplicates:
            duplicates.append(case.id)
        seen.add(case.id)
    if duplicates:
        raise ValueError(f"duplicate test case ids: {', '.join(duplicates)}")
    return cases


class DatasetIn(_Strict):
    """An uploaded dataset file: `{name?, description?, cases: [...]}` or a bare list of cases."""

    name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2_000)
    cases: list[TestCaseIn] = Field(min_length=1, max_length=MAX_CASES)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"cases": data}
        return data

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[TestCaseIn]) -> list[TestCaseIn]:
        return _check_unique_ids(cases)


class ExperimentCreate(_Strict):
    name: str = Field(min_length=1, max_length=200)
    system_prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    cases: list[TestCaseIn] = Field(min_length=1, max_length=MAX_CASES)
    baseline_experiment_id: str | None = None
    regression_threshold: float = Field(default=3.0, gt=0, le=100, description="Percentage points")

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[TestCaseIn]) -> list[TestCaseIn]:
        return _check_unique_ids(cases)


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: Literal["ok", "error"]


class ModelInfo(BaseModel):
    provider: ProviderName
    model_id: str
    available: bool
    reason: str | None = Field(default=None, description="Why the model cannot be run, if unavailable")
    sample: bool = Field(default=False, description="Offline sample model; output is not from a real LLM")


class JudgeInfo(BaseModel):
    provider: ProviderName | None
    model_id: str | None
    available: bool
    reason: str | None = None


class ModelsOut(BaseModel):
    models: list[ModelInfo]
    judge: JudgeInfo


class ModelRef(_Strict):
    provider: ProviderName
    model_id: str = Field(min_length=1, max_length=200)


class RunRequest(_Strict):
    models: list[ModelRef] = Field(min_length=1, max_length=10)

    @field_validator("models")
    @classmethod
    def _unique_models(cls, models: list[ModelRef]) -> list[ModelRef]:
        keys = [(m.provider, m.model_id) for m in models]
        if len(set(keys)) != len(keys):
            raise ValueError("each model may be selected only once per run")
        return models


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    provider: ProviderName
    model_id: str
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None
    judge_provider: str | None
    judge_model_id: str | None
    eval_config: dict[str, Any]
    eval_config_hash: str
    pricing_version: str | None
    sample: bool = False
    aggregate: RunAggregate | None = None


class ExperimentOut(BaseModel):
    id: str
    name: str
    system_prompt: str
    created_at: datetime
    baseline_experiment_id: str | None
    regression_threshold: float
    status: str
    dataset_hash: str
    prompt_hash: str
    case_count: int
    runs: list[RunOut]


class RunStartedOut(BaseModel):
    experiment_id: str
    status: str
    runs: list[RunOut]
