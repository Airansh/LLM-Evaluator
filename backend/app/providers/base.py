"""Normalized provider interface.

Every adapter turns one provider's request/response format into `ProviderResponse`.
Provider-specific parsing (usage fields, error types) stays inside the adapter; the
runner and scorers only ever see the normalized type.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

ProviderName = Literal["openai", "anthropic", "gemini"]
PROVIDERS: tuple[ProviderName, ...] = ("openai", "anthropic", "gemini")

ErrorKind = Literal["rate_limit", "timeout", "auth", "bad_request", "server", "unknown"]
RETRYABLE_ERRORS: frozenset[ErrorKind] = frozenset({"rate_limit", "timeout", "server"})


class CompletionRequest(BaseModel):
    system_prompt: str
    user_message: str
    max_output_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ProviderResponse(BaseModel):
    """One normalized completion attempt, including failures.

    Token counts are `None` when the provider did not report them, never 0.
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
