"""Error classification for the OpenAI and Anthropic SDKs, which share exception names."""

from types import ModuleType

from app.providers.base import ErrorKind


def classify_sdk_error(sdk: ModuleType, exc: Exception) -> tuple[ErrorKind, str]:
    """Map an SDK exception to a normalized kind, most specific class first."""
    name = type(exc).__name__
    message = getattr(exc, "message", None) or str(exc)
    detail = f"{name}: {message}"

    # APITimeoutError subclasses APIConnectionError in both SDKs, so check it first.
    if isinstance(exc, sdk.APITimeoutError):
        return "timeout", detail
    if isinstance(exc, sdk.APIConnectionError):
        return "connection", detail
    if isinstance(exc, sdk.RateLimitError):
        return "rate_limit", detail
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return "auth", detail
    if isinstance(exc, (sdk.BadRequestError, sdk.NotFoundError, sdk.UnprocessableEntityError)):
        return "bad_request", detail
    if isinstance(exc, sdk.APIStatusError):
        # Includes InternalServerError and Anthropic's 529 "overloaded".
        return ("server" if exc.status_code >= 500 else "bad_request"), f"{detail} (HTTP {exc.status_code})"
    return "unknown", detail
