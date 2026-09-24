import time
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.providers.base import (
    CompletionRequest,
    ErrorKind,
    ProviderName,
    ProviderResponse,
    elapsed_ms,
    error_response,
)


class GeminiProvider:
    """Uses the google-genai SDK. Output tokens = candidate tokens + thinking tokens (both billed)."""

    name: ProviderName = "gemini"

    def __init__(self, api_key: str, timeout_s: float, client: Any = None) -> None:
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),  # the runner owns retries
            ),
        )

    async def complete(self, model_id: str, request: CompletionRequest) -> ProviderResponse:
        config = types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
        )
        start = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=model_id, contents=request.user_message, config=config
            )
        except Exception as exc:  # noqa: BLE001
            kind, detail = _classify(exc)
            return error_response(self.name, model_id, start, kind, detail)
        latency = elapsed_ms(start)

        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", None)
        candidate_tokens = getattr(usage, "candidates_token_count", None)
        thought_tokens = getattr(usage, "thoughts_token_count", None) or 0
        output_tokens = None if candidate_tokens is None else candidate_tokens + thought_tokens

        block_reason = getattr(response.prompt_feedback, "block_reason", None) if response.prompt_feedback else None
        if block_reason:
            return error_response(
                self.name, model_id, start, "refused", f"Prompt blocked: {block_reason}", input_tokens, output_tokens
            )
        text = _extract_text(response)
        if not text:
            finish = response.candidates[0].finish_reason if response.candidates else None
            return error_response(
                self.name, model_id, start, "empty_response",
                f"No text in response (finish_reason={finish})", input_tokens, output_tokens,
            )
        return ProviderResponse(
            provider=self.name,
            model_id=model_id,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
        )


def _extract_text(response: Any) -> str:
    parts = []
    for candidate in (response.candidates or [])[:1]:
        for part in getattr(candidate.content, "parts", None) or []:
            if getattr(part, "text", None) and not getattr(part, "thought", False):
                parts.append(part.text)
    return "".join(parts).strip()


def _classify(exc: Exception) -> tuple[ErrorKind, str]:
    detail = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"
    if isinstance(exc, genai_errors.APIError):
        code = exc.code or 0
        if code == 429:
            return "rate_limit", detail
        if code in (401, 403):
            return "auth", detail
        if code >= 500:
            return "server", detail
        return "bad_request", detail
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", detail
    if isinstance(exc, httpx.TransportError):
        return "connection", detail
    return "unknown", detail
