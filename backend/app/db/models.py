"""ORM models. `NULL` in a score/usage/cost column means "not available", never zero."""

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ExperimentStatus(str, enum.Enum):
    draft = "draft"
    running = "running"
    completed = "completed"
    partial = "partial"
    failed = "failed"


class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    partial = "partial"
    failed = "failed"


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    system_prompt: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    baseline_experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"), default=None
    )
    regression_threshold: Mapped[float] = mapped_column(Float, default=3.0)
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus, native_enum=False), default=ExperimentStatus.draft
    )
    # SHA-256 of the canonicalized case snapshot / prompt; used for baseline compatibility.
    dataset_hash: Mapped[str] = mapped_column(String(64))
    prompt_hash: Mapped[str] = mapped_column(String(64))

    test_cases: Mapped[list["TestCase"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", order_by="TestCase.position"
    )
    model_runs: Mapped[list["ModelRun"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", order_by="ModelRun.started_at"
    )


class TestCase(Base):
    """Immutable snapshot of one dataset case, owned by one experiment."""

    __tablename__ = "test_cases"
    __table_args__ = (UniqueConstraint("experiment_id", "case_id"),)
    __test__ = False  # not a pytest test class

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    case_id: Mapped[str] = mapped_column(String(64))
    position: Mapped[int] = mapped_column(Integer)
    input: Mapped[str] = mapped_column(Text)
    context: Mapped[str | None] = mapped_column(Text, default=None)
    expected_answer: Mapped[str] = mapped_column(Text)
    required_facts: Mapped[list[str]] = mapped_column(JSON, default=list)
    # `metadata` is reserved on declarative classes.
    case_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    experiment: Mapped[Experiment] = relationship(back_populates="test_cases")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    model_id: Mapped[str] = mapped_column(String(200))
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus, native_enum=False), default=RunStatus.pending)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    # Evaluator identity and settings, recorded so comparisons stay interpretable.
    judge_provider: Mapped[str | None] = mapped_column(String(32), default=None)
    judge_model_id: Mapped[str | None] = mapped_column(String(200), default=None)
    eval_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    eval_config_hash: Mapped[str] = mapped_column(String(64))
    pricing_version: Mapped[str | None] = mapped_column(String(32), default=None)

    experiment: Mapped[Experiment] = relationship(back_populates="model_runs")
    results: Mapped[list["EvaluationResult"]] = relationship(
        back_populates="model_run", cascade="all, delete-orphan"
    )


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (UniqueConstraint("model_run_id", "test_case_pk"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)
    test_case_pk: Mapped[int] = mapped_column(ForeignKey("test_cases.pk", ondelete="CASCADE"), index=True)

    response: Mapped[str | None] = mapped_column(Text, default=None)
    correctness_score: Mapped[float | None] = mapped_column(Float, default=None)
    groundedness_score: Mapped[float | None] = mapped_column(Float, default=None)
    required_facts_score: Mapped[float | None] = mapped_column(Float, default=None)
    latency_ms: Mapped[float | None] = mapped_column(Float, default=None)
    input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    output_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    # Candidate-model cost; evaluation (judge) cost is tracked separately below.
    estimated_cost: Mapped[float | None] = mapped_column(Float, default=None)
    judge_input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    judge_output_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    judge_cost: Mapped[float | None] = mapped_column(Float, default=None)
    judge_reason: Mapped[str | None] = mapped_column(Text, default=None)
    # `error`: the candidate call failed. `judge_error`: the call succeeded but scoring failed.
    error: Mapped[str | None] = mapped_column(Text, default=None)
    judge_error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    model_run: Mapped[ModelRun] = relationship(back_populates="results")
    test_case: Mapped[TestCase] = relationship()
