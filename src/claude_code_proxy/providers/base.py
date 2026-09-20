"""Shared provider contract and errors."""

from collections.abc import AsyncIterator
import math
from typing import Protocol

from ..domain.models import CompletionRequest, CompletionResponse, StreamError, StreamEvent
from ..failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
    retryable_status,
    unexpected_failure_diagnostic,
)


class ProviderError(Exception):
    """A provider failure whose exception message is safe to return to clients."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int = 500,
        diagnostic: FailureDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.diagnostic = diagnostic


class UnsupportedOperationError(ProviderError):
    pass


def stream_error_from_exception(error: Exception, *, provider: str) -> StreamError:
    if isinstance(error, ProviderError):
        status_code = error.status_code
        error_type, _ = public_error(status_code)
        return StreamError(
            error_type=error_type,
            message=str(error),
            status_code=status_code,
            retryable=retryable_status(status_code),
            provider=error.provider,
            diagnostic=error.diagnostic or _provider_error_fallback(status_code),
        )

    status_code = _status_code(error)
    error_type, message = public_error(status_code)
    return StreamError(
        error_type=error_type,
        message=message,
        status_code=status_code,
        retryable=retryable_status(status_code),
        provider=provider,
        diagnostic=unexpected_failure_diagnostic(
            error, stage=FailureStage.STREAM
        ),
    )


def protocol_error(
    code: str,
    *,
    provider: str | None = None,
    stage: FailureStage = FailureStage.STREAM,
) -> StreamError:
    return StreamError(
        provider=provider,
        diagnostic=FailureDiagnostic(
            FailureCategory.PROVIDER_PROTOCOL,
            stage,
            code,
        ),
    )


def public_error(status_code: int | None) -> tuple[str, str]:
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


def scalar_provider_code(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return None


def _provider_error_fallback(status_code: int | None) -> FailureDiagnostic:
    category = (
        FailureCategory.UPSTREAM_HTTP
        if status_code is not None
        else FailureCategory.INTERNAL
    )
    return FailureDiagnostic(category, FailureStage.STREAM, "provider_error")


def _status_code(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code
    return None


class Provider(Protocol):
    name: str

    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]: ...
    async def count_tokens(self, request: CompletionRequest) -> int: ...
