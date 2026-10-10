"""Required-fact coverage: a lexical check, not a measure of semantic truth."""

from app.evaluation.text import normalize_text


def missing_facts(response: str, required_facts: list[str]) -> list[str]:
    haystack = normalize_text(response)
    return [fact for fact in required_facts if normalize_text(fact) not in haystack]


def required_facts_score(response: str, required_facts: list[str]) -> float | None:
    """Percent (0-100) of required facts found as normalized substrings of the response.

    Returns None (N/A) when the case lists no required facts, rather than 100%.
    """
    if not required_facts:
        return None
    found = len(required_facts) - len(missing_facts(response, required_facts))
    return 100.0 * found / len(required_facts)
