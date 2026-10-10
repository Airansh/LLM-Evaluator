import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError

from app.evaluation.fingerprint import dataset_hash
from app.schemas import DatasetIn, DatasetValidationOut

router = APIRouter(prefix="/datasets", tags=["datasets"])

SAMPLE_DATASET = Path(__file__).resolve().parents[3] / "datasets" / "customer_support.json"


def _error_messages(exc: ValidationError) -> list[str]:
    messages = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        messages.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return messages


@router.get("/sample")
def sample_dataset() -> dict[str, Any]:
    """The bundled synthetic customer-support dataset."""
    if not SAMPLE_DATASET.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="sample dataset file is missing")
    data: dict[str, Any] = json.loads(SAMPLE_DATASET.read_text())
    return data


@router.post("/validate", response_model=DatasetValidationOut)
def validate_dataset(payload: Any = Body(...)) -> DatasetValidationOut:
    """Validate an uploaded dataset without storing it; returns its content hash for baseline matching."""
    try:
        dataset = DatasetIn.model_validate(payload)
    except ValidationError as exc:
        return DatasetValidationOut(valid=False, errors=_error_messages(exc))
    return DatasetValidationOut(
        valid=True, errors=[], case_count=len(dataset.cases), dataset_hash=dataset_hash(dataset.cases)
    )
