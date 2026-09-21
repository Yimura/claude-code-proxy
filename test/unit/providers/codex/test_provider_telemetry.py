import asyncio
from dataclasses import replace
import logging

import httpx
import pytest

from claude_code_proxy.domain.models import (
    Message,
    RedactedThinkingBlock,
    StreamComplete,
    StreamError,
    StreamStart,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)
from claude_code_proxy.failures import FailureCategory, FailureDiagnostic, FailureStage
from claude_code_proxy.providers.codex.provider import CodexProvider
from claude_code_proxy.providers.codex.reasoning import encode_reasoning
from claude_code_proxy.reasoning import ReasoningPolicy
from test.unit.providers.codex.provider_test_support import (
    Auth,
    BlockingEnterContext,
    Client,
    EnterFailureContext,
    ExitClient,
    RecordingTelemetry,
    Response,
    collect,
    completed_response,
    request,
    reset_client as reset_client,
)


async def test_codex_telemetry_callback_failure_is_isolated(caplog):
    marker = "sensitive telemetry failure"

    class FailingTelemetry:
        def mark_retries_supported(self):
            raise RuntimeError(marker)

        def set_reasoning_continuation(self, value):
            raise RuntimeError(marker)

    Client.responses = [completed_response()]

    with caplog.at_level(
        logging.WARNING, logger="claude_code_proxy.performance"
    ):
        events = await collect(
            CodexProvider(Auth(), Client), telemetry=FailingTelemetry()
        )

    assert events == [
        StreamStart(),
        StreamComplete("end_turn", TokenUsage(0, 0)),
    ]
    assert marker not in caplog.text


async def test_non_401_response_observes_zero_retries():
    telemetry = RecordingTelemetry()
    Client.responses = [completed_response()]

    await collect(CodexProvider(Auth(), Client), telemetry=telemetry)

    assert ("record_retry",) not in telemetry.calls


async def test_early_close_does_not_record_retry():
    telemetry = RecordingTelemetry()
    response = Response(block_lines=True)
    ExitClient.responses = [response]
    stream = CodexProvider(Auth(), ExitClient).stream(
        request(), telemetry=telemetry
    )

    assert await anext(stream) == StreamStart()
    await stream.aclose()

    assert ("record_retry",) not in telemetry.calls


async def test_stream_without_telemetry_does_not_classify_reasoning(
    monkeypatch,
):
    def fail_classifier(*args, **kwargs):
        raise RuntimeError("telemetry-only failure")

    monkeypatch.setattr(
        "claude_code_proxy.providers.codex.provider.reasoning_continuation_state",
        fail_classifier,
    )
    Client.responses = [completed_response()]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamStart(),
        StreamComplete("end_turn", TokenUsage(0, 0)),
    ]


async def test_stream_reports_original_request_before_reconciliation(
    monkeypatch,
):
    telemetry = RecordingTelemetry()

    def disable_reasoning(completion_request):
        return replace(
            completion_request,
            reasoning=ReasoningPolicy(False, None),
        )

    monkeypatch.setattr(
        "claude_code_proxy.providers.codex.provider.reconcile_codex_request",
        disable_reasoning,
    )
    Client.responses = [completed_response()]

    await collect(
        CodexProvider(Auth(), Client),
        request(reasoning=ReasoningPolicy(True, "high")),
        telemetry=telemetry,
    )

    assert telemetry.calls == [
        ("mark_retries_supported",),
        ("set_reasoning_continuation", "expected"),
    ]


async def test_stream_reports_capability_then_reasoning_once():
    telemetry = RecordingTelemetry()
    Client.responses = [completed_response()]

    await collect(
        CodexProvider(Auth(), Client),
        request(reasoning=ReasoningPolicy(True, "high")),
        telemetry=telemetry,
    )

    assert telemetry.calls == [
        ("mark_retries_supported",),
        ("set_reasoning_continuation", "expected"),
    ]


async def test_complete_reports_adapter_facts_only_once():
    telemetry = RecordingTelemetry()
    Client.responses = [completed_response()]

    await CodexProvider(Auth(), Client).complete(
        request(reasoning=ReasoningPolicy(True, "high")),
        telemetry=telemetry,
    )

    assert telemetry.calls == [
        ("mark_retries_supported",),
        ("set_reasoning_continuation", "expected"),
    ]


async def test_stream_reports_restored_reasoning_without_carrier_content():
    marker = "sensitive-encrypted-state"
    carrier = encode_reasoning(marker, [])
    completion_request = request(
        reasoning=ReasoningPolicy(True, "high"),
        messages=(
            Message(
                "assistant",
                (
                    RedactedThinkingBlock(carrier),
                    ToolUseBlock("call-1", "lookup", {}),
                ),
            ),
            Message("user", (ToolResultBlock("call-1", "done"),)),
        ),
    )
    telemetry = RecordingTelemetry()
    Client.responses = [completed_response()]

    await collect(
        CodexProvider(Auth(), Client),
        completion_request,
        telemetry=telemetry,
    )

    assert telemetry.calls[-1] == (
        "set_reasoning_continuation",
        "restored",
    )
    assert marker not in repr(telemetry.calls)
    assert carrier not in repr(telemetry.calls)


async def test_401_recovers_credentials_after_closing_response_and_retries_once():
    order = []
    rejected = Response(status=401)

    def recovered_after_close():
        assert rejected.exited is True
        order.append("credentials_recovered")

    class EnteredResponse(Response):
        async def __aenter__(self):
            response = await super().__aenter__()
            order.append("response_entered")
            return response

    class OrderedClient(Client):
        def stream(self, method, url, **kwargs):
            order.append("request_started")
            return super().stream(method, url, **kwargs)

    class OrderedTelemetry(RecordingTelemetry):
        def record_retry(self):
            order.append("retry_recorded")
            super().record_retry()

    auth = Auth(
        recovered=("new-access", "account"),
        on_recover=recovered_after_close,
    )
    OrderedClient.responses = [
        rejected,
        EnteredResponse(lines=completed_response().lines),
    ]

    telemetry = OrderedTelemetry()
    events = await collect(
        CodexProvider(auth, OrderedClient), telemetry=telemetry
    )

    assert telemetry.calls == [
        ("mark_retries_supported",),
        ("set_reasoning_continuation", "not_applicable"),
        ("record_retry",),
    ]
    assert order == [
        "request_started",
        "credentials_recovered",
        "retry_recorded",
        "request_started",
        "response_entered",
    ]
    assert events == [
        StreamStart(),
        StreamComplete("end_turn", TokenUsage(0, 0)),
    ]
    assert auth.rejected == ["secret-access"]
    assert len(OrderedClient.requests) == 2
    assert (
        OrderedClient.requests[0][2]["headers"]["Authorization"]
        == "Bearer secret-access"
    )
    assert (
        OrderedClient.requests[1][2]["headers"]["Authorization"]
        == "Bearer new-access"
    )


async def test_cancel_during_second_response_entry_records_retry():
    blocked = BlockingEnterContext()
    Client.responses = [Response(status=401), blocked]
    telemetry = RecordingTelemetry()
    stream = CodexProvider(Auth(), Client).stream(
        request(), telemetry=telemetry
    )
    pending = asyncio.create_task(anext(stream))
    await blocked.waiting.wait()

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert telemetry.calls.count(("record_retry",)) == 1


async def test_second_response_entry_failure_records_retry():
    error = httpx.ConnectError("SECOND_ATTEMPT_SECRET")
    Client.responses = [Response(status=401), EnterFailureContext(error)]
    telemetry = RecordingTelemetry()

    events = await collect(
        CodexProvider(Auth(), Client), telemetry=telemetry
    )

    assert telemetry.calls.count(("record_retry",)) == 1
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].diagnostic is not None
    assert events[0].diagnostic.stage == FailureStage.REQUEST


async def test_second_401_returns_authentication_error_without_stream_start():
    auth = Auth(recovered=("new-access", "account"))
    Client.responses = [Response(status=401), Response(status=401)]

    telemetry = RecordingTelemetry()
    events = await collect(
        CodexProvider(auth, Client), telemetry=telemetry
    )

    assert telemetry.calls.count(("record_retry",)) == 1
    assert len(Client.requests) == 2
    assert auth.rejected == ["secret-access"]
    assert events == [
        StreamError(
            error_type="authentication_error",
            message="Authentication failed",
            status_code=401,
            retryable=False,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.AUTHENTICATION,
                FailureStage.CREDENTIALS,
                "credentials_rejected",
            ),
        )
    ]


async def test_initial_credential_failure_returns_safe_structured_error():
    auth = Auth(
        get_failure=RuntimeError(
            "access sample-access-token refresh sample-refresh-token"
        )
    )

    telemetry = RecordingTelemetry()
    events = await collect(
        CodexProvider(auth, Client), telemetry=telemetry
    )

    assert telemetry.calls == [
        ("mark_retries_supported",),
        ("set_reasoning_continuation", "not_applicable"),
    ]
    assert Client.requests == []
    assert events == [
        StreamError(
            error_type="authentication_error",
            message="Authentication failed",
            status_code=401,
            retryable=False,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.AUTHENTICATION,
                FailureStage.CREDENTIALS,
                "credential_load_failed",
            ),
        )
    ]
    assert "sample-access-token" not in repr(events)
    assert "sample-refresh-token" not in repr(events)


async def test_recovery_failure_returns_safe_error_without_second_request():
    rejected = Response(status=401)
    auth = Auth(
        recovery_failure=RuntimeError("sample-refresh-token was rejected"),
        on_recover=lambda: rejected.exited
        or pytest.fail("response must close before credential recovery"),
    )
    Client.responses = [rejected]

    telemetry = RecordingTelemetry()
    events = await collect(
        CodexProvider(auth, Client), telemetry=telemetry
    )

    assert ("record_retry",) not in telemetry.calls
    assert len(Client.requests) == 1
    assert events == [
        StreamError(
            error_type="authentication_error",
            message="Authentication failed",
            status_code=401,
            retryable=False,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.AUTHENTICATION,
                FailureStage.CREDENTIALS,
                "credential_recovery_failed",
            ),
        )
    ]
    assert "sample-refresh-token" not in repr(events)


@pytest.mark.parametrize(
    "target",
    ["reconcile_codex_request", "build_request"],
)
async def test_preparation_failure_still_reports_adapter_facts(
    monkeypatch, target
):
    telemetry = RecordingTelemetry()

    def fail_preparation(*args, **kwargs):
        raise RuntimeError("sensitive carrier must not leak")

    monkeypatch.setattr(
        f"claude_code_proxy.providers.codex.provider.{target}",
        fail_preparation,
    )

    events = await collect(
        CodexProvider(Auth(), Client), telemetry=telemetry
    )

    assert telemetry.calls == [
        ("mark_retries_supported",),
        ("set_reasoning_continuation", "not_applicable"),
    ]
    assert "sensitive carrier" not in repr(events)


async def test_complete_without_telemetry_uses_legacy_stream_arity(monkeypatch):
    async def stream(self, completion_request):
        yield TextDelta("ok")
        yield StreamComplete("end_turn", TokenUsage(1, 1))

    monkeypatch.setattr(CodexProvider, "stream", stream)

    result = await CodexProvider(Auth(), Client).complete(request())

    assert result.content == (TextBlock("ok"),)


async def test_count_tokens_without_telemetry_uses_legacy_counter_arity():
    captured = []

    async def local_counter(completion_request):
        captured.append(completion_request)
        return 8

    provider = CodexProvider(Auth(), Client, local_counter)

    assert await provider.count_tokens(request()) == 8
    assert len(captured) == 1


async def test_complete_forwards_telemetry_to_internal_stream(monkeypatch):
    telemetry = object()
    captured = []
    async def stream(self, completion_request, telemetry=None):
        captured.append(telemetry)
        yield TextDelta("ok")
        yield StreamComplete("end_turn", TokenUsage(1, 1))

    monkeypatch.setattr(CodexProvider, "stream", stream)
    result = await CodexProvider(Auth(), Client).complete(
        request(), telemetry=telemetry
    )

    assert result.content == (TextBlock("ok"),)
    assert captured == [telemetry]


async def test_count_tokens_forwards_telemetry_to_local_counter():
    telemetry = object()
    captured = []

    async def local_counter(completion_request, telemetry=None):
        captured.append((completion_request, telemetry))
        return 8

    provider = CodexProvider(Auth(), Client, local_counter)

    assert await provider.count_tokens(request(), telemetry=telemetry) == 8
    assert captured[0][1] is telemetry
