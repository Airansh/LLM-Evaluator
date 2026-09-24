import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import datasets, experiments, health, models
from app.config import Settings, get_settings
from app.db.session import configure_engine, init_db, session_factory
from app.evaluation.pricing import PricingTable, latest_pricing
from app.providers.registry import ProviderFactory
from app.services import recover_interrupted_runs

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    provider_factory: ProviderFactory | None = None,
    pricing: PricingTable | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        configure_engine(settings.database_url)
        init_db()
        with session_factory()() as session:
            if interrupted := recover_interrupted_runs(session):
                logger.warning("marked %d interrupted run(s) as failed", interrupted)
        yield

    app = FastAPI(title="EvalForge", version=__version__, lifespan=lifespan)
    app.state.provider_factory = provider_factory or ProviderFactory(settings)
    app.state.pricing = pricing or latest_pricing()
    app.dependency_overrides[get_settings] = lambda: settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(experiments.router)
    app.include_router(datasets.router)
    return app


app = create_app()
