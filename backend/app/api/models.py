from fastapi import APIRouter, Depends

from app.api.deps import get_provider_factory
from app.providers.registry import ProviderFactory
from app.schemas import ModelsOut

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsOut)
def list_models(factory: ProviderFactory = Depends(get_provider_factory)) -> ModelsOut:
    """Configured candidate models and whether each can run. Reports key presence only, never keys."""
    return ModelsOut(models=factory.list_models(), judge=factory.judge_info())
