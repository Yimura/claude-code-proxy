from dataclasses import FrozenInstanceError
import logging
from types import SimpleNamespace

import pytest

import claude_code_proxy.api.routes as routes_module
from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.logging import RequestLogContext, SessionIdentity
from claude_code_proxy.observability import SessionMetadata, SessionRegistry


def metadata() -> SessionMetadata:
    return SessionMetadata(
        client_identity=ClientIdentity("session"),
        client_model="claude-opus",
        upstream_model="openai/gpt-5.6-sol",
        provider="openai",
        transport="codex",
        effort="high",
        context_window=1_000_000,
    )


def diagnostic() -> FailureDiagnostic:
    return FailureDiagnostic(
        FailureCategory.INTERNAL,
        FailureStage.ROUTE,
        "test_failure",
    )


def registry(mode: str) -> SessionRegistry:
    return SessionRegistry(
        10,
        secret=b"finalization-status-secret",
        performance_enabled=mode != "off",
        performance_logging_enabled=mode == "logging",
    )


def log_context() -> RequestLogContext:
    return RequestLogContext(
        session=SessionIdentity("session", "session", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude-opus",
        upstream_model="openai/gpt-5.6-sol",
        provider="codex",
        effort="high",
    )


def test_finish_with_status_distinguishes_off_success_from_duplicate() -> None:
    sessions = registry("off")
    handle = sessions.begin(metadata())

    finalized = sessions.finish_with_status(handle, "completed")
    duplicate = sessions.finish_with_status(handle, "failed")

    assert finalized.finalized is True
    assert finalized.performance is None
    assert duplicate.finalized is False
    assert duplicate.performance is None
    with pytest.raises(FrozenInstanceError):
        finalized.finalized = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("outcome", "expected_result"),
    [("completed", "completed"), ("client_disconnected", "failed")],
)
def test_off_stream_finalization_is_idempotent(
    outcome: str,
    expected_result: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sessions = registry("off")
    handle = sessions.begin(metadata())
    request = SimpleNamespace(state=SimpleNamespace())
    if outcome == "client_disconnected":
        routes_module._set_stream_response_outcome(request, outcome)

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        first = routes_module._finalize_stream_request(
            request, sessions, handle, log_context(), outcome
        )
        second = routes_module._finalize_stream_request(
            request, sessions, handle, log_context(), "failed"
        )

    state = routes_module._stream_terminal_state(request)
    assert first is True
    assert second is False
    assert state.finalized is True
    assert state.response_outcome == (
        "client_disconnected" if outcome == "client_disconnected" else None
    )
    assert sessions.snapshots()[0].last_result == expected_result
    assert "performance outcome=" not in caplog.text


@pytest.mark.parametrize("mode", ["off", "collector", "logging"])
@pytest.mark.parametrize(
    "outcome", ["completed", "cancelled", "client_disconnected"]
)
def test_nonfailed_outcome_rejects_explicit_diagnostic_atomically(
    mode: str,
    outcome: str,
) -> None:
    sessions = registry(mode)
    handle = sessions.begin(metadata())
    cursor = sessions.events.current_sequence

    with pytest.raises(
        ValueError,
        match="non-failed outcome cannot retain a failure diagnostic",
    ):
        sessions.finish(handle, outcome, diagnostic())

    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 1
    assert snapshot.last_result is None
    assert sessions.events.current_sequence == cursor


@pytest.mark.parametrize("mode", ["off", "collector", "logging"])
def test_failed_outcome_accepts_explicit_diagnostic(mode: str) -> None:
    sessions = registry(mode)
    handle = sessions.begin(metadata())

    result = sessions.finish_with_status(handle, "failed", diagnostic())

    assert result.finalized is True
    assert sessions.snapshots()[0].last_result == "failed"
    if mode == "off":
        assert result.performance is None
    else:
        assert result.performance is not None
        assert result.performance.failure == diagnostic()
