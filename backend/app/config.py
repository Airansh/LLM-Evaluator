"""Server-side configuration. Nothing in here is ever serialized to API clients directly."""

from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.providers.base import ProviderName

CommaList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    # Works whether uvicorn is started from the repo root or from backend/.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    # Alternative name for hosts that reserve ANTHROPIC_* variables for their own use (e.g. hosted
    # coding-agent environments). A non-empty ANTHROPIC_API_KEY wins if both are set.
    evalforge_anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    openai_models: CommaList = Field(default_factory=list)
    anthropic_models: CommaList = Field(default_factory=list)
    gemini_models: CommaList = Field(default_factory=list)

    judge_provider: ProviderName | None = None
    judge_model: str | None = None

    # Exposes the offline, clearly labeled "sample" provider and judge.
    sample_mode: bool = False

    database_url: str = "sqlite:///./data/evalforge.db"
    cors_origins: CommaList = Field(default_factory=lambda: ["http://localhost:3000"])
    max_concurrency: int = Field(default=4, ge=1, le=32)
    max_attempts: int = Field(default=3, ge=1, le=6, description="Per call, including the first try")
    request_timeout_s: float = Field(default=120.0, gt=0)
    candidate_max_output_tokens: int = Field(default=8192, ge=1)
    judge_max_output_tokens: int = Field(default=8192, ge=1)
    # Blank = provider default. Some models reject an explicit temperature.
    temperature: float | None = Field(default=None, ge=0, le=2)

    @field_validator("openai_models", "anthropic_models", "gemini_models", "cors_origins", mode="before")
    @classmethod
    def _split_commas(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("openai_api_key", "anthropic_api_key", "evalforge_anthropic_api_key", "gemini_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("judge_provider", "judge_model", "temperature", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def api_key(self, provider: ProviderName) -> SecretStr | None:
        """None for providers without a key, including the keyless "sample" provider."""
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key or self.evalforge_anthropic_api_key,
            "gemini": self.gemini_api_key,
        }.get(provider)

    def configured_models(self, provider: ProviderName) -> list[str]:
        return {
            "openai": self.openai_models,
            "anthropic": self.anthropic_models,
            "gemini": self.gemini_models,
        }.get(provider, [])


@lru_cache
def get_settings() -> Settings:
    return Settings()
