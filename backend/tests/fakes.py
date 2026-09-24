"""Scripted fake providers/judges so tests need no network or API keys."""

from collections.abc import Callable

from app.config import Settings
from app.evaluation.judge import Judge, JudgeCase, JudgeOutcome
from app.providers.base import CompletionRequest, ErrorKind, Provider, ProviderName, ProviderResponse
from app.providers.registry import ProviderFactory

Script = Callable[[str, CompletionRequest, int], ProviderResponse]


async def no_sleep(_seconds: float) -> None:
    return None


def ok(provider: ProviderName, model_id: str, text: str, tokens: tuple[int, int] = (100, 20)) -> ProviderResponse:
    return ProviderResponse(
        provider=provider, model_id=model_id, text=text, input_tokens=tokens[0], output_tokens=tokens[1], latency_ms=12.5
    )


def fail(provider: ProviderName, model_id: str, kind: ErrorKind, message: str = "boom") -> ProviderResponse:
    return ProviderResponse(provider=provider, model_id=model_id, latency_ms=3.0, error=message, error_kind=kind)


class FakeProvider:
    """Calls `script(model_id, request, call_number)`; defaults to echoing the question."""

    def __init__(self, name: ProviderName, script: Script | None = None) -> None:
        self.name = name
        self.calls: list[tuple[str, CompletionRequest]] = []
        self._script = script

    async def complete(self, model_id: str, request: CompletionRequest) -> ProviderResponse:
        self.calls.append((model_id, request))
        if self._script is not None:
            return self._script(model_id, request, len(self.calls))
        return ok(self.name, model_id, f"Answer about 30 days. {request.user_message[-40:]}")


class FakeJudge:
    def __init__(self, provider: ProviderName = "anthropic", model_id: str = "fake-judge", score: float = 80) -> None:
        self.provider = provider
        self.model_id = model_id
        self.score = score
        self.calls: list[tuple[JudgeCase, str]] = []

    async def evaluate(self, case: JudgeCase, response: str) -> JudgeOutcome:
        self.calls.append((case, response))
        return JudgeOutcome(
            correctness=self.score,
            groundedness=None if case.context is None else self.score,
            reason="fake judge",
            input_tokens=50,
            output_tokens=10,
        )


class FakeFactory(ProviderFactory):
    def __init__(self, settings: Settings, providers: dict[str, Provider], judge: Judge | None = None) -> None:
        super().__init__(settings, sleep=no_sleep)
        self._fakes = providers
        self._judge = judge or FakeJudge()

    def _build_provider(self, name: ProviderName) -> Provider:
        return self._fakes[name]

    def judge_for(self, candidate_provider: ProviderName) -> Judge:
        if candidate_provider == "sample":
            return super().judge_for(candidate_provider)
        return self._judge
