"""SAMPLE MODE: a deterministic offline provider and judge for trying the app without API keys.

Everything produced here is labeled with provider "sample". Outputs are simple text
heuristics over the supplied context (sample-extractive answers in a single sentence when
the system prompt asks for "one sentence"), latency is simulated, and token counts are word
counts. None of it says anything about how a real model would perform.
"""

import asyncio
import hashlib
import re
import time

from app.evaluation.judge import JudgeCase, JudgeOutcome
from app.evaluation.prompts import CONTEXT_HEADER, QUESTION_HEADER
from app.evaluation.text import normalize_text
from app.providers.base import CompletionRequest, ProviderName, ProviderResponse, elapsed_ms, error_response

SAMPLE_MODELS: tuple[str, ...] = ("sample-extractive", "sample-terse", "sample-flaky")
SAMPLE_JUDGE_MODEL = "sample-heuristic-judge"

_WORD = re.compile(r"[a-z0-9$%'-]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from have how i if in is it its my of on or so that the this to "
    "was what when where which will with you your".split()
)


def _words(text: str) -> list[str]:
    return _WORD.findall(normalize_text(text))


def _content_words(text: str) -> set[str]:
    return {word for word in _words(text) if word not in _STOPWORDS}


def _stable_fraction(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _split_message(message: str) -> tuple[str | None, str]:
    if message.startswith(CONTEXT_HEADER):
        context, _, question = message[len(CONTEXT_HEADER) :].partition(f"\n\n{QUESTION_HEADER}\n")
        return context.strip(), question.strip()
    return None, message.removeprefix(QUESTION_HEADER).strip()


class SampleProvider:
    name: ProviderName = "sample"

    def __init__(self, simulate_latency: bool = True) -> None:
        self._simulate_latency = simulate_latency

    async def complete(self, model_id: str, request: CompletionRequest) -> ProviderResponse:
        start = time.perf_counter()
        context, question = _split_message(request.user_message)
        if self._simulate_latency:
            await asyncio.sleep(0.05 + 0.25 * _stable_fraction(model_id, question))
        input_tokens = len(_words(request.system_prompt)) + len(_words(request.user_message))

        if model_id not in SAMPLE_MODELS:
            return error_response(self.name, model_id, start, "bad_request", f"Unknown sample model {model_id!r}")
        if model_id == "sample-flaky" and _stable_fraction("flaky", question) < 0.3:
            # Deterministic persistent failure, to demonstrate partial-run handling.
            return error_response(self.name, model_id, start, "server", "Simulated 503 from sample-flaky")

        if context is None:
            text = "I don't have specific policy information for that; please contact support for help."
        elif model_id == "sample-terse":
            text = _SENTENCE.split(context)[0]
        else:
            # Prompt-sensitive on purpose, so offline prompt-regression demos have something to detect.
            limit = 1 if "one sentence" in request.system_prompt.casefold() else 2
            text = self._extract(context, question, limit)
        return ProviderResponse(
            provider=self.name,
            model_id=model_id,
            text=text,
            input_tokens=input_tokens,
            output_tokens=len(_words(text)),
            latency_ms=elapsed_ms(start),
        )

    @staticmethod
    def _extract(context: str, question: str, limit: int) -> str:
        """Return the (up to `limit`) context sentences sharing the most words with the question."""
        question_words = _content_words(question)
        sentences = [s for s in _SENTENCE.split(context) if s.strip()]
        ranked = sorted(sentences, key=lambda s: -len(question_words & _content_words(s)))
        chosen = [s for s in sentences if s in ranked[:limit]]
        return " ".join(chosen)


class SampleJudge:
    """Heuristic stand-in for an LLM judge: word-overlap scores, clearly not an LLM."""

    provider: ProviderName = "sample"
    model_id = SAMPLE_JUDGE_MODEL

    async def evaluate(self, case: JudgeCase, response: str) -> JudgeOutcome:
        response_words = _content_words(response)
        expected_words = _content_words(case.expected_answer)
        overlap = len(response_words & expected_words)
        precision = overlap / len(response_words) if response_words else 0.0
        recall = overlap / len(expected_words) if expected_words else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        groundedness = None
        if case.context is not None:
            context_words = _content_words(case.context)
            groundedness = round(100 * len(response_words & context_words) / len(response_words)) if response_words else 0

        return JudgeOutcome(
            correctness=round(100 * f1),
            groundedness=groundedness,
            reason=(
                "SAMPLE heuristic judge (not an LLM): correctness = word-overlap F1 with the expected answer; "
                "groundedness = share of response words found in the context."
            ),
            input_tokens=len(response_words) + len(expected_words),
            output_tokens=1,
        )
