"""Builds providers and judges from settings, and reports what is runnable."""

import asyncio

from app.config import Settings
from app.evaluation.judge import Judge, LLMJudge
from app.evaluation.retry import Sleep
from app.providers.base import PROVIDERS, SAMPLE_PROVIDER, Provider, ProviderName
from app.providers.sample_provider import SAMPLE_MODELS, SampleJudge, SampleProvider
from app.schemas import JudgeInfo, ModelInfo


class Unavailable(Exception):
    """A requested model or judge cannot run; the message says why."""


def key_env_var(provider: ProviderName) -> str:
    return f"{provider.upper()}_API_KEY"


class ProviderFactory:
    def __init__(self, settings: Settings, sleep: Sleep = asyncio.sleep) -> None:
        self.settings = settings
        self.sleep = sleep
        self._providers: dict[ProviderName, Provider] = {}

    # ---- discovery -------------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for provider in PROVIDERS:
            has_key = self.settings.api_key(provider) is not None
            for model_id in self.settings.configured_models(provider):
                models.append(
                    ModelInfo(
                        provider=provider,
                        model_id=model_id,
                        available=has_key,
                        reason=None if has_key else f"{key_env_var(provider)} is not set",
                    )
                )
        if self.settings.sample_mode:
            models.extend(
                ModelInfo(provider=SAMPLE_PROVIDER, model_id=model_id, available=True, sample=True)
                for model_id in SAMPLE_MODELS
            )
        return models

    def unavailable_reason(self, provider: ProviderName, model_id: str) -> str | None:
        for model in self.list_models():
            if model.provider == provider and model.model_id == model_id:
                return model.reason
        if provider == SAMPLE_PROVIDER and not self.settings.sample_mode:
            return "sample mode is disabled (set SAMPLE_MODE=true)"
        return f"{provider}/{model_id} is not configured (add it to {provider.upper()}_MODELS)"

    def judge_info(self) -> JudgeInfo:
        provider, model_id = self.settings.judge_provider, self.settings.judge_model
        reason: str | None = None
        if provider is None or model_id is None:
            reason = "JUDGE_PROVIDER and JUDGE_MODEL must both be set"
        elif provider == SAMPLE_PROVIDER:
            reason = "the sample judge is used automatically for sample models; choose a real judge provider"
        elif self.settings.api_key(provider) is None:
            reason = f"{key_env_var(provider)} is not set"
        return JudgeInfo(provider=provider, model_id=model_id, available=reason is None, reason=reason)

    # ---- construction ----------------------------------------------------------------------

    def provider(self, name: ProviderName) -> Provider:
        if name not in self._providers:
            self._providers[name] = self._build_provider(name)
        return self._providers[name]

    def _build_provider(self, name: ProviderName) -> Provider:
        if name == SAMPLE_PROVIDER:
            return SampleProvider()
        key = self.settings.api_key(name)
        if key is None:
            raise Unavailable(f"{key_env_var(name)} is not set")
        timeout = self.settings.request_timeout_s
        # Imported lazily so an SDK import problem only affects that provider.
        if name == "anthropic":
            from app.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(key.get_secret_value(), timeout)
        if name == "openai":
            from app.providers.openai_provider import OpenAIProvider

            return OpenAIProvider(key.get_secret_value(), timeout)
        from app.providers.gemini_provider import GeminiProvider

        return GeminiProvider(key.get_secret_value(), timeout)

    def judge_for(self, candidate_provider: ProviderName) -> Judge:
        """Sample candidates are scored by the sample judge; real candidates by the configured judge."""
        if candidate_provider == SAMPLE_PROVIDER:
            return SampleJudge()
        info = self.judge_info()
        if not info.available or info.provider is None or info.model_id is None:
            raise Unavailable(f"judge unavailable: {info.reason}")
        return LLMJudge(
            self.provider(info.provider),
            info.model_id,
            max_output_tokens=self.settings.judge_max_output_tokens,
            max_attempts=self.settings.max_attempts,
            sleep=self.sleep,
        )

    def secrets(self) -> list[str]:
        """Configured key values, used to redact any echo of them from stored errors."""
        keys = (self.settings.api_key(provider) for provider in PROVIDERS)
        return [key.get_secret_value() for key in keys if key is not None]
