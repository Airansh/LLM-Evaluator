from fastapi import Request

from app.evaluation.pricing import PricingTable
from app.providers.registry import ProviderFactory


def get_provider_factory(request: Request) -> ProviderFactory:
    """One factory (and so one SDK client per provider) per app, set in `create_app`."""
    factory: ProviderFactory = request.app.state.provider_factory
    return factory


def get_pricing(request: Request) -> PricingTable:
    pricing: PricingTable = request.app.state.pricing
    return pricing
