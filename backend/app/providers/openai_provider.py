import time
from typing import Any

import openai

from app.providers._sdk_errors import classify_sdk_error
from app.providers.base import CompletionRequest, ProviderName, ProviderResponse, elapsed_ms, error_response


class OpenAIProvider:
    """Uses the Responses API. Reported `output_tokens` include any reasoning tokens."""

    name: ProviderName = "openai"

    def __init__(self, api_key: str, timeout_s: float, client: Any = None) -> None:
        self._client = client or openai.AsyncOpenAI(api_key=api_key, max_retries=0, timeout=timeout_s)

    async def complete(self, model_id: str, request: CompletionRequest) -> ProviderResponse:
        params: dict[str, Any] = {
            "model": model_id,
            "instructions": request.system_prompt,
            "input": request.user_message,
            "max_output_tokens": request.max_output_tokens,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature

        start = time.perf_counter()
        try:
            response = await self._client.responses.create(**params)
        except Exception as exc:  # noqa: BLE001
            kind, detail = classify_sdk_error(openai, exc)
            return error_response(self.name, model_id, start, kind, detail)
        latency = elapsed_ms(start)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        text = (response.output_text or "").strip()

        if not text:
            refusal = _find_refusal(response)
            if refusal is not None:
                return error_response(
                    self.name, model_id, start, "refused", f"Model refused: {refusal}", input_tokens, output_tokens
                )
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            return error_response(
                self.name, model_id, start, "empty_response",
                f"No text in response (status={response.status}, reason={reason})", input_tokens, output_tokens,
            )
        return ProviderResponse(
            provider=self.name,
            model_id=model_id,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
        )


def _find_refusal(response: Any) -> str | None:
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return str(getattr(part, "refusal", "") or "refusal")
    return None
