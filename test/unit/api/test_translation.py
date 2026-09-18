import pytest
from pydantic import ValidationError
from claude_code_proxy.api.schemas import MessagesRequest
from claude_code_proxy.api.translation import normalize_request, to_api_response
from claude_code_proxy.domain.models import (
    CompletionResponse,
    RedactedThinkingBlock,
    TextBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)
from claude_code_proxy.logging import (
    RequestLogContext,
    SessionIdentity,
    observe_stream,
)
from claude_code_proxy.reasoning import ReasoningPolicy


def test_normalize_request_preserves_ordered_content_and_options():
    request = MessagesRequest(
        model="claude-sonnet-test", max_tokens=200,
        system=[{"type": "text", "text": "system"}],
        messages=[
            {"role": "assistant", "content": [{"type": "text", "text": "before"}, {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"q": "x"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "done"}]},
        ],
        tools=[{"name": "lookup", "description": "Lookup", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "lookup"}, thinking={"type": "adaptive"},
    )
    normalized = normalize_request(request)
    assert normalized.original_model == "claude-sonnet-test"
    assert normalized.model == "claude-sonnet-test"
    assert normalized.response_model == "claude-sonnet-test"
    assert normalized.system == (TextBlock("system"),)
    assert normalized.messages[0].content == (TextBlock("before"), ToolUseBlock("call-1", "lookup", {"q": "x"}))
    assert normalized.messages[1].content == (ToolResultBlock("call-1", "done"),)
    assert normalized.reasoning == ReasoningPolicy(None, None)
    assert normalized.tool_choice.name == "lookup"


def test_normalize_request_preserves_redacted_thinking():
    request = MessagesRequest(
        model="model",
        max_tokens=100,
        messages=[{
            "role": "assistant",
            "content": [{
                "type": "redacted_thinking",
                "data": "codex-reasoning-v1:data",
            }],
        }],
    )

    normalized = normalize_request(request)

    assert normalized.messages[0].content == (
        RedactedThinkingBlock("codex-reasoning-v1:data"),
    )


def test_to_api_response_serializes_redacted_thinking():
    response = CompletionResponse(
        "msg-1",
        "model",
        (RedactedThinkingBlock("codex-reasoning-v1:data"),),
        "end_turn",
        TokenUsage(1, 1),
    )

    assert to_api_response(response).content[0].model_dump() == {
        "type": "redacted_thinking",
        "data": "codex-reasoning-v1:data",
    }


def test_schema_keeps_submitted_model_until_service_mapping():
    request = MessagesRequest(model="claude-sonnet", max_tokens=100, messages=[{"role": "user", "content": "hello"}])
    assert request.model == "claude-sonnet"
    assert normalize_request(request).model == "claude-sonnet"


@pytest.mark.parametrize("thinking", [{"type": "adaptive"}, {"type": "disabled"}, {"enabled": True}, {"enabled": False}])
def test_schema_accepts_supported_thinking_forms(thinking):
    MessagesRequest(model="model", max_tokens=100, messages=[], thinking=thinking)


def test_schema_rejects_invalid_effort():
    with pytest.raises(ValidationError):
        MessagesRequest(model="model", max_tokens=100, messages=[], output_config={"effort": "extreme"})


import asyncio
import json
from dataclasses import replace

from claude_code_proxy.api.translation import serialize_stream, to_api_response
from claude_code_proxy.domain.models import (
    CompletionResponse,
    RedactedThinking,
    StreamComplete,
    StreamError,
    StreamStart,
    TextDelta,
    TokenUsage,
    ToolInputDelta,
    ToolUseEnd,
    ToolUseStart,
)


async def event_source(*events):
    for event in events:
        yield event


def event_names(frames):
    return [frame.split("\n", 1)[0].removeprefix("event: ") for frame in frames if frame.startswith("event: ")]


def test_to_api_response_serializes_text_tools_and_usage():
    response = CompletionResponse(
        "msg-1",
        "openai/gpt-5",
        (TextBlock("hello"), ToolUseBlock("call-1", "lookup", {"q": "x"})),
        "tool_use",
        TokenUsage(4, 2, 1, 3),
    )
    api = to_api_response(response)
    assert api.model_dump() == {
        "id": "msg-1",
        "model": "openai/gpt-5",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"q": "x"}},
        ],
        "type": "message",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 2, "cache_creation_input_tokens": 1, "cache_read_input_tokens": 3},
    }


@pytest.mark.asyncio
async def test_text_stream_has_anthropic_lifecycle_order():
    normalized = normalize_request(MessagesRequest(model="model", max_tokens=10, messages=[]))
    frames = [frame async for frame in serialize_stream(normalized, event_source(StreamStart(3), TextDelta("hello"), StreamComplete("end_turn", TokenUsage(3, 1))))]
    assert event_names(frames) == ["message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_start_uses_client_response_model():
    normalized = normalize_request(
        MessagesRequest(model="claude-opus-5", max_tokens=10, messages=[])
    )
    prepared = replace(normalized, response_model="claude-opus-5[1m]")

    frames = [frame async for frame in serialize_stream(
        prepared,
        event_source(StreamComplete("end_turn", TokenUsage(1, 1))),
    )]

    start = json.loads(frames[0].split("data: ", 1)[1])
    assert start["message"]["model"] == "claude-opus-5[1m]"


@pytest.mark.asyncio
async def test_tool_stream_maps_slots_to_stable_indices():
    normalized = normalize_request(MessagesRequest(model="model", max_tokens=10, messages=[]))
    frames = [frame async for frame in serialize_stream(normalized, event_source(
        ToolUseStart("slot-b", "call-b", "second"),
        ToolUseStart("slot-a", "call-a", "first"),
        ToolInputDelta("slot-a", "{}"),
        ToolUseEnd("slot-b"),
        ToolUseEnd("slot-a"),
        StreamComplete("tool_use", TokenUsage(0, 0)),
    ))]
    payloads = [json.loads(frame.split("data: ", 1)[1]) for frame in frames if frame.startswith("event: content_block_start")]
    assert [(item["index"], item["content_block"]["type"]) for item in payloads] == [(0, "tool_use"), (1, "tool_use")]


def content_starts(frames):
    return [
        json.loads(frame.split("data: ", 1)[1])
        for frame in frames
        if frame.startswith("event: content_block_start")
    ]


@pytest.mark.asyncio
async def test_reasoning_then_tool_uses_exact_content_order():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    frames = [frame async for frame in serialize_stream(
        normalized,
        event_source(
            RedactedThinking("codex-reasoning-v1:data"),
            ToolUseStart("0", "call-1", "lookup"),
            ToolUseEnd("0"),
            StreamComplete("tool_use", TokenUsage(1, 1)),
        ),
    )]

    assert [
        (item["index"], item["content_block"]["type"])
        for item in content_starts(frames)
    ] == [(0, "redacted_thinking"), (1, "tool_use")]


@pytest.mark.asyncio
async def test_reasoning_then_text_preserves_content_order():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    frames = [frame async for frame in serialize_stream(
        normalized,
        event_source(
            RedactedThinking("codex-reasoning-v1:data"),
            TextDelta("answer"),
            StreamComplete("end_turn", TokenUsage(1, 1)),
        ),
    )]

    assert [
        (item["index"], item["content_block"]["type"])
        for item in content_starts(frames)
    ] == [(0, "redacted_thinking"), (1, "text")]
    assert event_names(frames)[-2:] == ["message_delta", "message_stop"]


@pytest.mark.asyncio
async def test_text_reasoning_tool_preserves_content_order():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    frames = [frame async for frame in serialize_stream(
        normalized,
        event_source(
            TextDelta("before"),
            RedactedThinking("codex-reasoning-v1:data"),
            ToolUseStart("0", "call-1", "lookup"),
            ToolUseEnd("0"),
            StreamComplete("tool_use", TokenUsage(1, 1)),
        ),
    )]

    assert [
        (item["index"], item["content_block"]["type"])
        for item in content_starts(frames)
    ] == [(0, "text"), (1, "redacted_thinking"), (2, "tool_use")]


@pytest.mark.asyncio
async def test_empty_success_emits_compatible_empty_text_block():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )

    frames = [frame async for frame in serialize_stream(
        normalized,
        event_source(StreamComplete("end_turn", TokenUsage(0, 0))),
    )]

    assert [
        (item["index"], item["content_block"]["type"])
        for item in content_starts(frames)
    ] == [(0, "text")]
    assert event_names(frames) == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


@pytest.mark.asyncio
async def test_stream_error_emits_terminal_anthropic_error_only():
    normalized = normalize_request(MessagesRequest(model="model", max_tokens=10, messages=[]))
    error = StreamError(
        error_type="api_error",
        message="Internal server error",
        diagnostic="upstream failed",
    )

    frames = [frame async for frame in serialize_stream(normalized, event_source(error))]

    assert event_names(frames)[-1] == "error"
    assert json.loads(frames[-1].split("data: ", 1)[1]) == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }
    assert "upstream failed" not in "".join(frames)
    assert "message_stop" not in event_names(frames)
    assert "data: [DONE]\n\n" not in frames


@pytest.mark.asyncio
async def test_stream_error_does_not_close_partial_tool_block_as_success():
    normalized = normalize_request(MessagesRequest(model="model", max_tokens=10, messages=[]))
    events = event_source(
        ToolUseStart("0", "call-1", "lookup"),
        ToolInputDelta("0", '{"q":'),
        StreamError(error_type="api_error", message="Internal server error"),
    )

    frames = [frame async for frame in serialize_stream(normalized, events)]

    assert event_names(frames)[-1] == "error"
    assert "content_block_stop" not in event_names(frames)
    assert "message_delta" not in event_names(frames)
    assert "message_stop" not in event_names(frames)


@pytest.mark.asyncio
async def test_unexpected_eof_emits_protocol_error():
    normalized = normalize_request(MessagesRequest(model="model", max_tokens=10, messages=[]))

    frames = [frame async for frame in serialize_stream(normalized, event_source())]

    assert event_names(frames)[-1] == "error"
    assert '"type": "api_error"' in frames[-1]
    assert "message_stop" not in event_names(frames)
    assert "data: [DONE]\n\n" not in frames


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        (ToolInputDelta("missing", "{}"),),
        (ToolUseEnd("missing"),),
        (
            ToolUseStart("0", "call-1", "lookup"),
            ToolUseStart("0", "call-2", "lookup"),
        ),
        (
            ToolUseStart("0", "call-1", "lookup"),
            ToolUseEnd("0"),
            ToolUseEnd("0"),
        ),
        (
            ToolUseStart("0", "call-1", "lookup"),
            TextDelta("late text"),
        ),
        (
            ToolUseStart("0", "call-1", "lookup"),
            StreamComplete("tool_use", TokenUsage(1, 1)),
        ),
    ],
)
async def test_invalid_content_transition_emits_protocol_error(events):
    normalized = normalize_request(MessagesRequest(model="model", max_tokens=10, messages=[]))

    frames = [frame async for frame in serialize_stream(normalized, event_source(*events))]

    assert event_names(frames)[-1] == "error"
    assert "message_stop" not in event_names(frames)


@pytest.mark.asyncio
async def test_stream_emits_pings_while_waiting_for_upstream_event():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    release = asyncio.Event()

    async def delayed_events():
        yield StreamStart()
        await release.wait()
        yield TextDelta("hello")
        yield StreamComplete("end_turn", TokenUsage(1, 1))

    stream = serialize_stream(
        normalized,
        delayed_events(),
        heartbeat_interval=0.001,
    )
    idle_frames = [await anext(stream) for _ in range(6)]
    release.set()
    remaining_frames = [frame async for frame in stream]

    assert event_names(idle_frames) == [
        "message_start",
        "ping",
        "ping",
        "ping",
        "ping",
        "ping",
    ]
    assert event_names(remaining_frames) == [
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


class BlockingEvents:
    def __init__(self):
        self.started = False
        self.cancelled = asyncio.Event()
        self.closed = False
        self._never = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.started:
            self.started = True
            return StreamStart()
        try:
            await self._never.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_closing_stream_cancels_and_closes_upstream_iterator():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    events = BlockingEvents()
    stream = serialize_stream(
        normalized,
        events,
        heartbeat_interval=0.005,
    )

    frames = [await anext(stream) for _ in range(4)]
    assert event_names(frames)[-1] == "ping"

    await stream.aclose()

    assert events.cancelled.is_set()
    assert events.closed is True


@pytest.mark.asyncio
async def test_closing_stream_closes_production_logging_wrapper():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    events = BlockingEvents()
    context = RequestLogContext(
        session=SessionIdentity("session", "[session session]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="model",
        upstream_model="model",
        provider="fake",
        effort="default",
    )
    stream = serialize_stream(
        normalized,
        observe_stream(events, context),
        heartbeat_interval=0.005,
    )

    frames = [await anext(stream) for _ in range(4)]
    assert event_names(frames)[-1] == "ping"

    await stream.aclose()

    assert events.cancelled.is_set()
    assert events.closed is True


class FailingEvents:
    def __init__(self):
        self.started = False
        self.release = asyncio.Event()
        self.failed = asyncio.Event()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.started:
            self.started = True
            return StreamStart()
        await self.release.wait()
        self.failed.set()
        raise RuntimeError("upstream failed")

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_closing_stream_propagates_completed_upstream_failure():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    events = FailingEvents()
    stream = serialize_stream(
        normalized,
        events,
        heartbeat_interval=0.001,
    )

    frames = [await anext(stream) for _ in range(4)]
    assert event_names(frames)[-1] == "ping"
    events.release.set()
    await events.failed.wait()

    with pytest.raises(RuntimeError, match="upstream failed"):
        await stream.aclose()

    assert events.closed is True


class FiniteEvents:
    def __init__(self):
        self.started = False
        self.release = asyncio.Event()
        self.exhausted = asyncio.Event()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.started:
            self.started = True
            return StreamStart()
        await self.release.wait()
        self.exhausted.set()
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_closing_stream_ignores_completed_upstream_exhaustion():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    events = FiniteEvents()
    stream = serialize_stream(
        normalized,
        events,
        heartbeat_interval=0.001,
    )

    frames = [await anext(stream) for _ in range(4)]
    assert event_names(frames)[-1] == "ping"
    events.release.set()
    await events.exhausted.wait()

    await stream.aclose()

    assert events.closed is True


class FutureEvents:
    def __init__(self):
        self.events = iter(
            [
                StreamStart(),
                StreamComplete("end_turn", TokenUsage(1, 1)),
            ]
        )
        self.closed = False

    def __aiter__(self):
        return self

    def __anext__(self):
        future = asyncio.get_running_loop().create_future()
        try:
            future.set_result(next(self.events))
        except StopIteration:
            future.set_exception(StopAsyncIteration())
        return future

    async def aclose(self):
        self.closed = True


class MalformedClosableEvents:
    def __init__(self):
        self.events = iter(
            [
                ToolUseStart("slot", "tool-1", "lookup"),
                StreamComplete("tool_use", TokenUsage(1, 1)),
            ]
        )
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_serialize_stream_ignores_ordinary_error_callback_failure():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    events = MalformedClosableEvents()

    def fail_callback(_error):
        raise OSError("logging sink unavailable")

    frames = [
        frame
        async for frame in serialize_stream(
            normalized,
            events,
            on_error=fail_callback,
        )
    ]

    assert event_names(frames)[-1] == "error"
    assert "data: [DONE]\n\n" not in frames
    assert events.closed is True


@pytest.mark.asyncio
async def test_serialize_stream_does_not_suppress_callback_cancellation():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    events = MalformedClosableEvents()

    def cancel_callback(_error):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        _ = [
            frame
            async for frame in serialize_stream(
                normalized,
                events,
                on_error=cancel_callback,
            )
        ]

    assert events.closed is True


@pytest.mark.asyncio
async def test_serialize_stream_reports_synthesized_state_error():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    errors = []
    events = event_source(
        ToolUseStart("slot", "tool-1", "lookup"),
        StreamComplete("tool_use", TokenUsage(1, 1)),
    )

    frames = [
        frame
        async for frame in serialize_stream(
            normalized,
            events,
            on_error=errors.append,
        )
    ]

    assert event_names(frames)[-1] == "error"
    assert len(errors) == 1
    assert errors[0].diagnostic == "stream completed with open tool blocks"


@pytest.mark.asyncio
async def test_serialize_stream_reports_synthesized_eof_error():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    errors = []

    frames = [
        frame
        async for frame in serialize_stream(
            normalized,
            event_source(TextDelta("partial")),
            on_error=errors.append,
        )
    ]

    assert event_names(frames)[-1] == "error"
    assert len(errors) == 1
    assert errors[0].diagnostic == "stream ended without terminal outcome"


@pytest.mark.asyncio
async def test_serialize_stream_does_not_report_upstream_error_as_synthesized():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    errors = []

    frames = [
        frame
        async for frame in serialize_stream(
            normalized,
            event_source(StreamError(diagnostic="upstream failed")),
            on_error=errors.append,
        )
    ]

    assert event_names(frames)[-1] == "error"
    assert errors == []


@pytest.mark.asyncio
async def test_stream_accepts_future_backed_async_iterator():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    events = FutureEvents()

    frames = [frame async for frame in serialize_stream(normalized, events)]

    assert event_names(frames) == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events.closed is True


def test_to_api_response_serializes_thinking_token_details():
    response = CompletionResponse(
        "msg-thinking",
        "model",
        (TextBlock("ok"),),
        "end_turn",
        TokenUsage(4, 10, 1, 3, thinking_tokens=6),
    )

    assert to_api_response(response).usage.model_dump() == {
        "input_tokens": 4,
        "output_tokens": 10,
        "cache_creation_input_tokens": 1,
        "cache_read_input_tokens": 3,
        "output_tokens_details": {"thinking_tokens": 6},
    }


def test_to_api_response_omits_unreported_thinking_details():
    response = CompletionResponse(
        "msg-no-thinking",
        "model",
        (TextBlock("ok"),),
        "end_turn",
        TokenUsage(4, 10),
    )

    assert to_api_response(response).usage.model_dump(exclude_none=True) == {
        "input_tokens": 4,
        "output_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


@pytest.mark.asyncio
async def test_final_delta_emits_authoritative_cumulative_usage():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    frames = [frame async for frame in serialize_stream(
        normalized,
        event_source(
            StreamComplete("end_turn", TokenUsage(30, 20, 10, 60, 7))
        ),
    )]
    start = next(
        json.loads(frame.split("data: ", 1)[1])
        for frame in frames
        if frame.startswith("event: message_start")
    )
    delta = next(
        json.loads(frame.split("data: ", 1)[1])
        for frame in frames
        if frame.startswith("event: message_delta")
    )

    assert start["message"]["usage"] == {
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
    }
    assert delta["usage"] == {
        "input_tokens": 30,
        "output_tokens": 20,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 60,
        "output_tokens_details": {"thinking_tokens": 7},
    }


@pytest.mark.asyncio
async def test_final_delta_omits_unreported_thinking_details():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    frames = [frame async for frame in serialize_stream(
        normalized,
        event_source(StreamComplete("end_turn", TokenUsage(4, 2))),
    )]
    delta = next(
        json.loads(frame.split("data: ", 1)[1])
        for frame in frames
        if frame.startswith("event: message_delta")
    )

    assert "output_tokens_details" not in delta["usage"]


@pytest.mark.asyncio
async def test_final_delta_preserves_explicit_zero_thinking_tokens():
    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )
    frames = [frame async for frame in serialize_stream(
        normalized,
        event_source(
            StreamComplete("end_turn", TokenUsage(4, 2, thinking_tokens=0))
        ),
    )]
    delta = next(
        json.loads(frame.split("data: ", 1)[1])
        for frame in frames
        if frame.startswith("event: message_delta")
    )

    assert delta["usage"]["output_tokens_details"] == {"thinking_tokens": 0}
