import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATASET = REPO_ROOT / "datasets" / "customer_support.json"


@pytest.fixture
def sample_dataset() -> dict[str, Any]:
    return json.loads(SAMPLE_DATASET.read_text())


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Settings isolated from the developer's real .env and environment."""

    def factory(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "database_url": f"sqlite:///{tmp_path / 'test.db'}",
            "openai_api_key": None,
            "anthropic_api_key": None,
            "gemini_api_key": None,
            "openai_models": [],
            "anthropic_models": [],
            "gemini_models": [],
            "judge_provider": None,
            "judge_model": None,
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return factory


@pytest.fixture
def client(make_settings: Callable[..., Settings]) -> Iterator[TestClient]:
    with TestClient(create_app(make_settings())) as test_client:
        yield test_client
