import pytest
from pydantic import ValidationError
from claude_code_proxy.api.schemas import MessagesRequest
from claude_code_proxy.api.translation import normalize_request
from claude_code_proxy.domain.models import TextBlock, ToolResultBlock, ToolUseBlock
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
    assert normalized.system == (TextBlock("system"),)
    assert normalized.messages[0].content == (TextBlock("before"), ToolUseBlock("call-1", "lookup", {"q": "x"}))
    assert normalized.messages[1].content == (ToolResultBlock("call-1", "done"),)
    assert normalized.reasoning == ReasoningPolicy(None, None)
    assert normalized.tool_choice.name == "lookup"


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


import json

from claude_code_proxy.api.translation import serialize_stream, to_api_response
from claude_code_proxy.domain.models import (
    CompletionResponse,
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
    assert event_names(frames) == ["message_start", "content_block_start", "ping", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert frames[-1] == "data: [DONE]\n\n"


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
    assert [(item["index"], item["content_block"]["type"]) for item in payloads] == [(0, "text"), (1, "tool_use"), (2, "tool_use")]


@pytest.mark.asyncio
async def test_stream_error_closes_message_and_does_not_leak_structure():
    normalized = normalize_request(MessagesRequest(model="model", max_tokens=10, messages=[]))
    frames = [frame async for frame in serialize_stream(normalized, event_source(StreamError("upstream failed")))]
    assert "upstream failed" in "".join(frames)
    assert frames[-1] == "data: [DONE]\n\n"
    assert event_names(frames)[-1] == "message_stop"
