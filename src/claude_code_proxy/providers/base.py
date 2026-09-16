"""Shared provider contract and errors."""

from collections.abc import AsyncIterator
from typing import Protocol
from ..domain.models import CompletionRequest, CompletionResponse, StreamError, StreamEvent


class ProviderError(Exception):
    def __init__(self, message: str, *, provider: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class UnsupportedOperationError(ProviderError):
    pass


def stream_error_from_exception(
    error: Exception, *, provider: str, expose_message: bool = False
) -> StreamError:
    status_code = getattr(error, "status_code", None)
    error_type, default_message = _public_error(status_code)
    message = str(error) if expose_message else default_message
    return StreamError(
        error_type=error_type,
        message=message,
        status_code=status_code,
        retryable=status_code is None or status_code == 429 or status_code >= 500,
        provider=provider,
        diagnostic=str(error),
    )


def protocol_error(detail: str, *, provider: str | None = None) -> StreamError:
    return StreamError(provider=provider, diagnostic=detail)


def _public_error(status_code: int | None) -> tuple[str, str]:
    known_errors = {
        400: ("invalid_request_error", "Invalid request"),
        401: ("authentication_error", "Authentication failed"),
        402: ("billing_error", "Billing error"),
        403: ("permission_error", "Permission denied"),
        404: ("not_found_error", "Resource not found"),
        409: ("conflict_error", "Request conflict"),
        413: ("request_too_large", "Request too large"),
        429: ("rate_limit_error", "Rate limit exceeded"),
        504: ("timeout_error", "Request timed out"),
        529: ("overloaded_error", "Overloaded"),
    }
    if status_code in known_errors:
        return known_errors[status_code]
    if status_code is not None and 400 <= status_code < 500:
        return "invalid_request_error", "Invalid request"
    return "api_error", "Internal server error"


class Provider(Protocol):
    name: str

    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]: ...
    async def count_tokens(self, request: CompletionRequest) -> int: ...
