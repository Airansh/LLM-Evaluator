from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.providers.base import PROVIDERS, ProviderName
from app.schemas import JudgeInfo, ModelInfo, ModelsOut

router = APIRouter(tags=["models"])


def _key_env_var(provider: ProviderName) -> str:
    return f"{provider.upper()}_API_KEY"


@router.get("/models", response_model=ModelsOut)
def list_models(settings: Settings = Depends(get_settings)) -> ModelsOut:
    """Configured candidate models and whether each can run. Reports key presence only, never keys."""
    models: list[ModelInfo] = []
    for provider in PROVIDERS:
        has_key = settings.api_key(provider) is not None
        for model_id in settings.configured_models(provider):
            models.append(
                ModelInfo(
                    provider=provider,
                    model_id=model_id,
                    available=has_key,
                    reason=None if has_key else f"{_key_env_var(provider)} is not set",
                )
            )

    if settings.judge_provider is None or settings.judge_model is None:
        judge = JudgeInfo(
            provider=settings.judge_provider,
            model_id=settings.judge_model,
            available=False,
            reason="JUDGE_PROVIDER and JUDGE_MODEL must both be set",
        )
    elif settings.api_key(settings.judge_provider) is None:
        judge = JudgeInfo(
            provider=settings.judge_provider,
            model_id=settings.judge_model,
            available=False,
            reason=f"{_key_env_var(settings.judge_provider)} is not set",
        )
    else:
        judge = JudgeInfo(provider=settings.judge_provider, model_id=settings.judge_model, available=True)

    return ModelsOut(models=models, judge=judge)
