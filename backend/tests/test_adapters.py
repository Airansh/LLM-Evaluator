"""Adapter normalization against stubbed SDK clients (no network)."""

from types import SimpleNamespace as NS
from typing import Any

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import CompletionRequest
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider

REQUEST = CompletionRequest(system_prompt="sys", user_message="hello", max_output_tokens=256)


class Stub:
    """Minimal async method returning a canned value or raising."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.test/v1"))


# ---- Anthropic -----------------------------------------------------------------------------


def _anthropic(result: Any) -> tuple[AnthropicProvider, Stub]:
    stub = Stub(result)
    return AnthropicProvider("k", 10, client=NS(messages=NS(create=stub))), stub


async def test_anthropic_success_skips_thinking_and_omits_temperature() -> None:
    message = NS(
        content=[NS(type="thinking", thinking=""), NS(type="text", text=" No, 30 days. ")],
        usage=NS(input_tokens=42, output_tokens=7),
        stop_reason="end_turn",
    )
    provider, stub = _anthropic(message)
    response = await provider.complete("claude-sonnet-5", REQUEST)
    assert response.ok and response.text == "No, 30 days."
    assert (response.input_tokens, response.output_tokens) == (42, 7)
    assert stub.kwargs["system"] == "sys" and stub.kwargs["max_tokens"] == 256
    assert "temperature" not in stub.kwargs


async def test_anthropic_refusal_is_an_error_with_usage() -> None:
    message = NS(content=[], usage=NS(input_tokens=5, output_tokens=1), stop_reason="refusal")
    response = await _anthropic(message)[0].complete("m", REQUEST)
    assert response.error_kind == "refused" and response.input_tokens == 5


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (anthropic.RateLimitError("slow down", response=_http_response(429), body=None), "rate_limit"),
        (anthropic.InternalServerError("overloaded", response=_http_response(529), body=None), "server"),
        (anthropic.AuthenticationError("bad key", response=_http_response(401), body=None), "auth"),
        (anthropic.NotFoundError("no model", response=_http_response(404), body=None), "bad_request"),
        (anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.test")), "timeout"),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.test")), "connection"),
    ],
)
async def test_anthropic_error_mapping(exc: Exception, kind: str) -> None:
    response = await _anthropic(exc)[0].complete("m", REQUEST)
    assert response.error_kind == kind and not response.ok
    assert response.retryable == (kind in {"rate_limit", "server", "timeout", "connection"})


# ---- OpenAI --------------------------------------------------------------------------------


def _openai(result: Any) -> tuple[OpenAIProvider, Stub]:
    stub = Stub(result)
    return OpenAIProvider("k", 10, client=NS(responses=NS(create=stub))), stub


async def test_openai_success() -> None:
    response_obj = NS(output_text="Yes.", usage=NS(input_tokens=30, output_tokens=12), status="completed", output=[])
    provider, stub = _openai(response_obj)
    response = await provider.complete("gpt-6-luna", REQUEST.model_copy(update={"temperature": 0.0}))
    assert response.ok and response.text == "Yes." and response.output_tokens == 12
    assert stub.kwargs["instructions"] == "sys" and stub.kwargs["input"] == "hello"
    assert stub.kwargs["temperature"] == 0.0 and stub.kwargs["max_output_tokens"] == 256


async def test_openai_refusal_and_incomplete() -> None:
    refusal = NS(output_text="", usage=None, status="completed",
                 output=[NS(content=[NS(type="refusal", refusal="can't help")])])
    response = await _openai(refusal)[0].complete("m", REQUEST)
    assert response.error_kind == "refused" and response.input_tokens is None

    incomplete = NS(output_text="", usage=NS(input_tokens=1, output_tokens=256), status="incomplete",
                    incomplete_details=NS(reason="max_output_tokens"), output=[])
    response = await _openai(incomplete)[0].complete("m", REQUEST)
    assert response.error_kind == "empty_response" and "max_output_tokens" in (response.error or "")


async def test_openai_rate_limit() -> None:
    exc = openai.RateLimitError("429", response=_http_response(429), body=None)
    assert (await _openai(exc)[0].complete("m", REQUEST)).error_kind == "rate_limit"


# ---- Gemini --------------------------------------------------------------------------------


def _gemini(result: Any) -> tuple[GeminiProvider, Stub]:
    stub = Stub(result)
    client = NS(aio=NS(models=NS(generate_content=stub)))
    return GeminiProvider("k", 10, client=client), stub


async def test_gemini_success_counts_thinking_as_output() -> None:
    result = NS(
        candidates=[NS(content=NS(parts=[NS(text="plan", thought=True), NS(text="Answer.", thought=None)]),
                       finish_reason="STOP")],
        usage_metadata=NS(prompt_token_count=20, candidates_token_count=5, thoughts_token_count=30),
        prompt_feedback=None,
    )
    provider, stub = _gemini(result)
    response = await provider.complete("gemini-x", REQUEST)
    assert response.ok and response.text == "Answer."
    assert (response.input_tokens, response.output_tokens) == (20, 35)
    assert stub.kwargs["config"].system_instruction == "sys"


@pytest.mark.parametrize(("code", "kind"), [(429, "rate_limit"), (403, "auth"), (400, "bad_request"), (503, "server")])
async def test_gemini_error_mapping(code: int, kind: str) -> None:
    exc = genai_errors.APIError(code, {"error": {"message": "x", "status": "S"}})
    assert (await _gemini(exc)[0].complete("m", REQUEST)).error_kind == kind


async def test_gemini_timeout() -> None:
    assert (await _gemini(httpx.ReadTimeout("slow"))[0].complete("m", REQUEST)).error_kind == "timeout"
