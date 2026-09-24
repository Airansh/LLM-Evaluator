from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas import MAX_CASES, MAX_PROMPT_CHARS, DatasetIn, ExperimentCreate


def _case(case_id: str = "c1", **overrides: Any) -> dict[str, Any]:
    case = {"id": case_id, "input": "Question?", "expected_answer": "Answer.", "required_facts": ["fact"]}
    case.update(overrides)
    return case


def _errors(exc: pytest.ExceptionInfo[ValidationError]) -> str:
    return str(exc.value)


def test_sample_dataset_is_valid(sample_dataset: dict[str, Any]) -> None:
    dataset = DatasetIn.model_validate(sample_dataset)
    assert len(dataset.cases) >= 5
    assert dataset.cases[0].id == "returns_001"
    # The sample deliberately includes a no-context case and an empty-facts case.
    assert any(case.context is None for case in dataset.cases)
    assert any(case.required_facts == [] for case in dataset.cases)


def test_bare_list_is_accepted() -> None:
    dataset = DatasetIn.model_validate([_case("a"), _case("b")])
    assert [case.id for case in dataset.cases] == ["a", "b"]


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        DatasetIn.model_validate({"cases": [_case("dup"), _case("ok"), _case("dup")]})
    assert "duplicate test case ids: dup" in _errors(exc)


@pytest.mark.parametrize("bad_id", ["", "   ", "has space", "-leading-dash", "x" * 65])
def test_invalid_ids_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        DatasetIn.model_validate({"cases": [_case(bad_id)]})


@pytest.mark.parametrize("field", ["id", "input", "expected_answer"])
def test_missing_required_field_rejected(field: str) -> None:
    case = _case()
    del case[field]
    with pytest.raises(ValidationError) as exc:
        DatasetIn.model_validate({"cases": [case]})
    assert field in _errors(exc)


def test_blank_input_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetIn.model_validate({"cases": [_case(input="   ")]})


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        DatasetIn.model_validate({"cases": [_case(expected="typo")]})
    assert "expected" in _errors(exc)


def test_empty_and_oversized_datasets_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetIn.model_validate({"cases": []})
    with pytest.raises(ValidationError):
        DatasetIn.model_validate({"cases": [_case(f"c{i}") for i in range(MAX_CASES + 1)]})


def test_blank_context_becomes_none_and_empty_facts_allowed() -> None:
    dataset = DatasetIn.model_validate({"cases": [_case(context="  ", required_facts=[])]})
    assert dataset.cases[0].context is None
    assert dataset.cases[0].required_facts == []


def test_facts_validated() -> None:
    with pytest.raises(ValidationError):
        DatasetIn.model_validate({"cases": [_case(required_facts=["ok", " "])]})
    with pytest.raises(ValidationError) as exc:
        DatasetIn.model_validate({"cases": [_case(required_facts=["30 Days", "30  days"])]})
    assert "duplicate required fact" in _errors(exc)


def test_nested_metadata_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetIn.model_validate({"cases": [_case(metadata={"nested": {"a": 1}})]})


def test_experiment_prompt_length_and_threshold() -> None:
    base = {"name": "exp", "cases": [_case()]}
    ExperimentCreate.model_validate({**base, "system_prompt": "p" * MAX_PROMPT_CHARS})
    with pytest.raises(ValidationError):
        ExperimentCreate.model_validate({**base, "system_prompt": "p" * (MAX_PROMPT_CHARS + 1)})
    with pytest.raises(ValidationError):
        ExperimentCreate.model_validate({**base, "system_prompt": "  "})
    with pytest.raises(ValidationError):
        ExperimentCreate.model_validate({**base, "system_prompt": "p", "regression_threshold": 0})
    assert ExperimentCreate.model_validate({**base, "system_prompt": "p"}).regression_threshold == 3.0
