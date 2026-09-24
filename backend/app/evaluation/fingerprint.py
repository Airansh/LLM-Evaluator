"""Content hashes used to decide whether two experiments/runs are comparable."""

import hashlib
import json
from typing import Any

from app.schemas import TestCaseIn


def _sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def dataset_hash(cases: list[TestCaseIn]) -> str:
    """Order-independent hash of every field that affects generation or scoring."""
    return _sha256(sorted((case.model_dump() for case in cases), key=lambda case: case["id"]))


def prompt_hash(system_prompt: str) -> str:
    return _sha256(system_prompt)


def config_hash(eval_config: dict[str, Any]) -> str:
    return _sha256(eval_config)
