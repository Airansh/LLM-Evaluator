"""Text normalization shared by dataset validation and lexical scoring."""

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """NFKC-normalize, casefold, and collapse runs of whitespace to one space.

    Punctuation is kept on purpose: "$25" and "25" are different facts.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()
