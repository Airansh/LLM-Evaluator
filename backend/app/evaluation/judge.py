"""Correctness/groundedness judging. Judge scores are estimates, not verified facts."""

import json
import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.evaluation.prompts import JUDGE_SYSTEM_PROMPT, build_judge_message
from app.evaluation.retry import Sleep, complete_with_retry
from app.providers.base import CompletionRequest, Provider, ProviderName

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    correctness: float = Field(ge=0, le=100)
    groundedness: float | None = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=2_000)


@dataclass
class JudgeCase:
    question: str
    context: str | None
    expected_answer: str


@dataclass
class JudgeOutcome:
    correctness: float | None = None
    groundedness: float | None = None
    reason: str | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Judge(Protocol):
    provider: ProviderName
    model_id: str

    async def evaluate(self, case: JudgeCase, response: str) -> JudgeOutcome: ...


def parse_verdict(raw: str, has_context: bool) -> JudgeVerdict:
    """Parse and validate judge output. Raises ValueError on anything malformed.

    Malformed output is surfaced as an evaluation error; it is never turned into a zero score.
    """
    text = _FENCE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("judge response contained no JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge response was not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("judge response JSON was not an object")
    if "groundedness" not in data:
        raise ValueError("judge response is missing 'groundedness'")
    try:
        verdict = JudgeVerdict.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, err["loc"])) or "root" for err in exc.errors())
        raise ValueError(f"judge response failed validation ({fields})") from exc

    if not has_context:
        # Groundedness is N/A without context, whatever the judge said.
        verdict.groundedness = None
    elif verdict.groundedness is None:
        raise ValueError("judge returned null groundedness although context was supplied")
    return verdict


class LLMJudge:
    """Asks a separately configured model for structured scores."""

    def __init__(
        self, provider: Provider, model_id: str, *, max_output_tokens: int, max_attempts: int, sleep: Sleep
    ) -> None:
        self.provider: ProviderName = provider.name
        self.model_id = model_id
        self._backend = provider
        self._max_output_tokens = max_output_tokens
        self._max_attempts = max_attempts
        self._sleep = sleep

    async def evaluate(self, case: JudgeCase, response: str) -> JudgeOutcome:
        request = CompletionRequest(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=build_judge_message(case.question, case.context, case.expected_answer, response),
            max_output_tokens=self._max_output_tokens,
        )
        result, _ = await complete_with_retry(
            lambda: self._backend.complete(self.model_id, request), max_attempts=self._max_attempts, sleep=self._sleep
        )
        outcome = JudgeOutcome(input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        if not result.ok:
            outcome.error = f"judge call failed [{result.error_kind}]: {result.error}"
            return outcome
        try:
            verdict = parse_verdict(result.text or "", has_context=case.context is not None)
        except ValueError as exc:
            outcome.error = str(exc)
            return outcome
        outcome.correctness = verdict.correctness
        outcome.groundedness = verdict.groundedness
        outcome.reason = verdict.reason
        return outcome
