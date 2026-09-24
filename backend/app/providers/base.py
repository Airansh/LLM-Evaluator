"""Normalized provider interface.

Every adapter turns one provider's request/response format into `ProviderResponse`.
Provider-specific parsing (usage fields, error types) stays inside the adapter; the
runner and scorers only ever see the normalized type.
"""

import time
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

ProviderName = Literal["openai", "anthropic", "gemini", "sample"]
# Real providers backed by an API key. "sample" is the labeled offline provider.
PROVIDERS: tuple[ProviderName, ...] = ("openai", "anthropic", "gemini")
SAMPLE_PROVIDER: ProviderName = "sample"

ErrorKind = Literal[
    "rate_limit", "timeout", "connection", "server", "auth", "bad_request", "refused", "empty_response", "unknown"
]
RETRYABLE_ERRORS: frozenset[ErrorKind] = frozenset({"rate_limit", "timeout", "connection", "server"})
MAX_ERROR_CHARS = 500


class CompletionRequest(BaseModel):
    system_prompt: str
    user_message: str
    max_output_tokens: int = Field(default=8192, ge=1)
    # None = provider default. Several current models (e.g. Claude Sonnet 5, OpenAI
    # reasoning models) reject an explicit temperature, so it is opt-in.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ProviderResponse(BaseModel):
    """One normalized completion attempt, including failures.

    Token counts are `None` when the provider did not report them, never 0.
    `latency_ms` is wall-clock time for this single attempt.
    """

    provider: ProviderName
    model_id: str
    text: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    error: str | None = None
    error_kind: ErrorKind | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def retryable(self) -> bool:
        return self.error_kind in RETRYABLE_ERRORS


@runtime_checkable
class Provider(Protocol):
    """An async text-completion backend.

    `complete` must not raise for provider-side failures (rate limits, auth, bad
    requests). It returns a `ProviderResponse` with `error`/`error_kind` set instead,
    so a single failure can be persisted without aborting a whole run.
    """

    name: ProviderName

    async def complete(self, model_id: str, request: CompletionRequest) -> ProviderResponse: ...


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def error_response(
    provider: ProviderName,
    model_id: str,
    start: float,
    kind: ErrorKind,
    message: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> ProviderResponse:
    return ProviderResponse(
        provider=provider,
        model_id=model_id,
        latency_ms=elapsed_ms(start),
        error=message[:MAX_ERROR_CHARS],
        error_kind=kind,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
