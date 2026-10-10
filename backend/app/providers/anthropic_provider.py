import time
from typing import Any

import anthropic

from app.providers._sdk_errors import classify_sdk_error
from app.providers.base import CompletionRequest, ProviderName, ProviderResponse, elapsed_ms, error_response


class AnthropicProvider:
    name: ProviderName = "anthropic"

    def __init__(self, api_key: str, timeout_s: float, client: Any = None) -> None:
        # SDK retries are disabled; the runner owns retry/backoff so attempts are bounded and visible.
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout_s)

    async def complete(self, model_id: str, request: CompletionRequest) -> ProviderResponse:
        params: dict[str, Any] = {
            "model": model_id,
            "max_tokens": request.max_output_tokens,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_message}],
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature

        start = time.perf_counter()
        try:
            message = await self._client.messages.create(**params)
        except Exception as exc:  # noqa: BLE001 - normalized below, never raised to the runner
            kind, detail = classify_sdk_error(anthropic, exc)
            return error_response(self.name, model_id, start, kind, detail)
        latency = elapsed_ms(start)

        usage = getattr(message, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        # Thinking blocks (if any) are skipped; only the visible answer is evaluated.
        text = "".join(block.text for block in message.content if block.type == "text").strip()

        if message.stop_reason == "refusal":
            return error_response(
                self.name, model_id, start, "refused", "Model declined the request (stop_reason=refusal)",
                input_tokens, output_tokens,
            )
        if not text:
            return error_response(
                self.name, model_id, start, "empty_response",
                f"No text in response (stop_reason={message.stop_reason})", input_tokens, output_tokens,
            )
        return ProviderResponse(
            provider=self.name,
            model_id=model_id,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
        )
