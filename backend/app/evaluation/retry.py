import random
from collections.abc import Awaitable, Callable

from app.providers.base import ProviderResponse

Sleep = Callable[[float], Awaitable[None]]


def backoff_delay(attempt: int, base_s: float = 1.0, cap_s: float = 8.0) -> float:
    """Exponential backoff with up to 25% jitter. `attempt` is 1-based (delay after attempt N)."""
    delay = min(cap_s, base_s * 2 ** (attempt - 1))
    return delay + random.uniform(0, delay * 0.25)


async def complete_with_retry(
    call: Callable[[], Awaitable[ProviderResponse]], *, max_attempts: int, sleep: Sleep
) -> tuple[ProviderResponse, int]:
    """Retry transient failures (rate limits, timeouts, 5xx). Returns the last response and attempts used."""
    attempt = 1
    while True:
        response = await call()
        if response.ok or not response.retryable or attempt >= max_attempts:
            return response, attempt
        await sleep(backoff_delay(attempt))
        attempt += 1
