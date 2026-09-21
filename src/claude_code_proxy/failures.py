"""Structured diagnostics for provider failures."""

from dataclasses import dataclass
from enum import StrEnum


class FailureCategory(StrEnum):
    """Classify failures by their source."""

    AUTHENTICATION = "authentication"
    TRANSPORT = "transport"
    UPSTREAM_HTTP = "upstream_http"
    PROVIDER_PROTOCOL = "provider_protocol"
    TRANSLATION = "translation"
    INTERNAL = "internal"


class FailureStage(StrEnum):
    """Identify where a failure occurred in request processing."""

    CREDENTIALS = "credentials"
    REQUEST = "request"
    RESPONSE = "response"
    STREAM = "stream"
    PROVIDER_TRANSLATION = "provider_translation"
    CLIENT_TRANSLATION = "client_translation"
    ROUTE = "route"


_RECOGNIZED_PROVIDER_CODES = frozenset({
    "ECONNRESET",
    "api_error",
    "authentication_error",
    "billing_error",
    "conflict_error",
    "content_policy_violation",
    "context_length_exceeded",
    "insufficient_quota",
    "invalid_prompt",
    "invalid_request_error",
    "model_not_found",
    "not_found_error",
    "overloaded",
    "overloaded_error",
    "permission_error",
    "rate_limit_error",
    "rate_limit_exceeded",
    "request_too_large",
    "server_error",
    "service_unavailable",
    "timeout_error",
})


def recognized_provider_code(value: object) -> str | None:
    """Return an exact recognized provider category, never arbitrary detail."""
    if not isinstance(value, str) or value not in _RECOGNIZED_PROVIDER_CODES:
        return None
    return value


@dataclass(frozen=True, slots=True)
class FailureDiagnostic:
    """Describe a provider failure without carrying untrusted detail text."""

    category: FailureCategory
    stage: FailureStage
    code: str
    provider_code: str | None = None
    exception_type: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_code",
            recognized_provider_code(self.provider_code),
        )


def retryable_status(status_code: int | None) -> bool:
    """Return whether normalized provider status permits retry."""
    return status_code is None or status_code == 429 or status_code >= 500


def safe_exception_location(error: Exception) -> str:
    """Return the innermost application traceback frame without a file path."""
    traceback = error.__traceback__
    selected = None
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__")
        if isinstance(module, str) and (
            module == "claude_code_proxy"
            or module.startswith("claude_code_proxy.")
        ):
            selected = traceback
        traceback = traceback.tb_next
    if selected is None:
        return "unknown:unknown:0"
    frame = selected.tb_frame
    module = frame.f_globals["__name__"]
    return f"{module}:{frame.f_code.co_name}:{selected.tb_lineno}"


def unexpected_failure_diagnostic(
    error: Exception,
    *,
    stage: FailureStage,
    code: str = "unexpected_exception",
    category: FailureCategory = FailureCategory.INTERNAL,
) -> FailureDiagnostic:
    """Capture safe unexpected-failure evidence before traceback context is lost."""
    return FailureDiagnostic(
        category,
        stage,
        code,
        exception_type=type(error).__name__,
        location=safe_exception_location(error),
    )
