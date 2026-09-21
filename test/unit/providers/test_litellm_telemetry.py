import logging

import pytest

from claude_code_proxy.domain.models import (
    CompletionResponse,
    StreamComplete,
    StreamStart,
    TextBlock,
    TokenUsage,
)
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.providers.litellm import LiteLLMProvider
from claude_code_proxy.reasoning import ReasoningPolicy
from test.unit.providers.litellm_test_support import (
    FakeClient,
    request,
    settings as settings,
)


class RecordingTelemetry:
    def __init__(self):
        self.calls = []

    def mark_retries_supported(self):
        self.calls.append(("mark_retries_supported",))

    def record_retry(self):
        self.calls.append(("record_retry",))

    def set_reasoning_continuation(self, value):
        self.calls.append(("set_reasoning_continuation", value))


class FailingTelemetry(RecordingTelemetry):
    def set_reasoning_continuation(self, value):
        raise RuntimeError("sensitive callback failure")


async def invoke_provider_operation(
    provider, operation, completion_request, telemetry
):
    if operation == "complete":
        return await provider.complete(
            completion_request, telemetry=telemetry
        )
    if operation == "stream":
        return [
            event
            async for event in provider.stream(
                completion_request, telemetry=telemetry
            )
        ]
    return await provider.count_tokens(
        completion_request, telemetry=telemetry
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["complete", "stream"])
@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(True, "unavailable"), (False, "not_applicable"), (None, "not_applicable")],
)
async def test_operations_report_reasoning_state_once(
    settings, operation, enabled, expected
):
    telemetry = RecordingTelemetry()
    client = FakeClient(
        response={
            "id": "response-1",
            "choices": [{
                "message": {"content": "ok", "tool_calls": None},
                "finish_reason": "stop",
            }],
            "usage": {},
        },
        chunks=[{"choices": [{"delta": {}, "finish_reason": "stop"}]}],
    )

    await invoke_provider_operation(
        LiteLLMProvider(settings, client),
        operation,
        request(reasoning=ReasoningPolicy(enabled, None)),
        telemetry,
    )

    assert telemetry.calls == [("set_reasoning_continuation", expected)]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["complete", "stream"])
async def test_operations_report_reasoning_before_translation_failure(
    settings, monkeypatch, operation
):
    telemetry = RecordingTelemetry()
    provider = LiteLLMProvider(settings, FakeClient())

    def fail_build(*args, **kwargs):
        raise RuntimeError("sensitive build failure")

    monkeypatch.setattr(provider, "build_request", fail_build)

    if operation == "stream":
        await invoke_provider_operation(
            provider, operation, request(), telemetry
        )
    else:
        with pytest.raises(ProviderError):
            await invoke_provider_operation(
                provider, operation, request(), telemetry
            )

    assert telemetry.calls == [("set_reasoning_continuation", "unavailable")]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["complete", "stream", "count_tokens"])
async def test_telemetry_callback_failure_does_not_change_provider_behavior(
    settings, operation, caplog
):
    client = FakeClient(
        response={
            "id": "response-1",
            "choices": [{
                "message": {"content": "ok", "tool_calls": None},
                "finish_reason": "stop",
            }],
            "usage": {},
        },
        chunks=[{"choices": [{"delta": {}, "finish_reason": "stop"}]}],
    )

    with caplog.at_level(
        logging.WARNING, logger="claude_code_proxy.performance"
    ):
        result = await invoke_provider_operation(
            LiteLLMProvider(settings, client),
            operation,
            request(),
            FailingTelemetry(),
        )

    if operation == "complete":
        assert result == CompletionResponse(
            "response-1",
            "openai/gpt-5.6-sol",
            (TextBlock("ok"),),
            "end_turn",
            TokenUsage.unavailable(),
        )
    elif operation == "stream":
        assert result == [
            StreamStart(),
            StreamComplete("end_turn", TokenUsage.unavailable()),
        ]
    else:
        assert result == 9

    if operation == "count_tokens":
        assert caplog.records == []
    else:
        assert caplog.records[-1].getMessage() == "telemetry callback failed"
    assert "sensitive callback failure" not in caplog.text


@pytest.mark.asyncio
async def test_count_tokens_does_not_report_reasoning_continuation(settings):
    telemetry = RecordingTelemetry()

    result = await LiteLLMProvider(
        settings, FakeClient(token_count=17)
    ).count_tokens(request(), telemetry=telemetry)

    assert result == 17
    assert telemetry.calls == []


@pytest.mark.asyncio
async def test_stream_without_usage_marks_fields_unavailable(settings):
    client = FakeClient(chunks=[
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])

    events = [event async for event in LiteLLMProvider(settings, client).stream(request())]

    assert events[-1].usage.observed_fields == frozenset()
