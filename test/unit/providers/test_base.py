import re

import pytest

from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.providers.base import (
    ProviderError,
    protocol_error,
    scalar_provider_code,
    stream_error_from_exception,
)


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (402, "billing_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (409, "conflict_error"),
        (413, "request_too_large"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (503, "api_error"),
        (504, "timeout_error"),
        (529, "overloaded_error"),
    ],
)
def test_stream_error_preserves_safe_provider_error(status_code, error_type):
    diagnostic = FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.RESPONSE,
        "http_error",
        "provider-code",
    )
    error = ProviderError(
        "Safe provider message",
        provider="fake",
        status_code=status_code,
        diagnostic=diagnostic,
    )

    stream_error = stream_error_from_exception(error, provider="fallback")

    assert stream_error.error_type == error_type
    assert stream_error.message == "Safe provider message"
    assert stream_error.status_code == status_code
    assert stream_error.provider == "fake"
    assert stream_error.diagnostic is diagnostic


def test_stream_error_replaces_unexpected_exception_detail():
    namespace = {"__name__": "claude_code_proxy.synthetic"}
    exec(
        "def raise_failure():\n"
        "    local_secret = 'LOCAL_SECRET_MUST_NOT_LEAK'\n"
        "    raise RuntimeError('secret exception detail')\n",
        namespace,
    )
    try:
        namespace["raise_failure"]()
    except RuntimeError as error:
        stream_error = stream_error_from_exception(error, provider="fake")

    assert stream_error.message == "Internal server error"
    assert stream_error.provider == "fake"
    diagnostic = stream_error.diagnostic
    assert diagnostic is not None
    assert diagnostic.category == FailureCategory.INTERNAL
    assert diagnostic.stage == FailureStage.STREAM
    assert diagnostic.code == "unexpected_exception"
    assert diagnostic.exception_type == "RuntimeError"
    assert re.fullmatch(
        r"claude_code_proxy\.synthetic:raise_failure:\d+",
        diagnostic.location or "",
    )
    assert "secret" not in repr(stream_error)
    assert "LOCAL_SECRET_MUST_NOT_LEAK" not in repr(stream_error)


def test_provider_error_without_diagnostic_gets_structured_fallback():
    stream_error = stream_error_from_exception(
        ProviderError("Safe fake failure", provider="fake", status_code=503),
        provider="fallback",
    )

    assert stream_error.message == "Safe fake failure"
    assert stream_error.diagnostic == FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.STREAM,
        "provider_error",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("provider-code", "provider-code"),
        ("", ""),
        (True, "true"),
        (False, "false"),
        (42, "42"),
        (-3, "-3"),
        (1.25, "1.25"),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (None, None),
        ({"code": "nested"}, None),
        (["nested"], None),
    ],
)
def test_scalar_provider_code_canonicalizes_only_finite_scalars(value, expected):
    assert scalar_provider_code(value) == expected


def test_protocol_error_uses_stable_code_and_requested_stage():
    error = protocol_error(
        "malformed_event",
        provider="fake",
        stage=FailureStage.CLIENT_TRANSLATION,
    )

    assert error.message == "Internal server error"
    assert error.diagnostic == FailureDiagnostic(
        FailureCategory.PROVIDER_PROTOCOL,
        FailureStage.CLIENT_TRANSLATION,
        "malformed_event",
    )
