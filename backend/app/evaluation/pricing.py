"""Estimated cost from a versioned pricing file. Missing prices or usage give None ("unavailable")."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

PRICING_DIR = Path(__file__).resolve().parents[2] / "pricing"


class ModelPrice(BaseModel):
    input: float = Field(ge=0, description="USD per 1M input tokens")
    output: float = Field(ge=0, description="USD per 1M output tokens")
    as_of: str
    source: str
    note: str | None = None


class PricingTable(BaseModel):
    version: str
    currency: str = "USD"
    unit: str = "per_1m_tokens"
    models: dict[str, ModelPrice] = Field(description='Keyed by "provider/model_id"')

    def price(self, provider: str, model_id: str) -> ModelPrice | None:
        return self.models.get(f"{provider}/{model_id}")

    def estimate(
        self, provider: str, model_id: str, input_tokens: int | None, output_tokens: int | None
    ) -> float | None:
        price = self.price(provider, model_id)
        if price is None or input_tokens is None or output_tokens is None:
            return None
        return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def load_pricing(path: Path) -> PricingTable:
    return PricingTable.model_validate(json.loads(path.read_text()))


@lru_cache
def latest_pricing(directory: Path = PRICING_DIR) -> PricingTable:
    """Files are named `<version>.json`; the lexically greatest (ISO date) version wins."""
    files = sorted(directory.glob("*.json"))
    if not files:
        return PricingTable(version="none", models={})
    return load_pricing(files[-1])
