from dataclasses import replace
import json

from claude_code_proxy.api.schemas import MessagesRequest
from claude_code_proxy.api.translation import normalize_request, to_api_response

from claude_code_proxy.domain.models import (
    ClientIdentity, CompletionRequest, Message, RedactedThinking, RedactedThinkingBlock, StreamComplete, TextBlock, TextDelta, TokenUsage,
    ToolChoice, ToolDefinition, ToolInputDelta, ToolResultBlock, ToolUseBlock,
    ToolUseEnd, ToolUseStart,
)
from claude_code_proxy.providers.codex.identity import CodexIdentity
from claude_code_proxy.providers.codex.reasoning import decode_reasoning, encode_reasoning
from claude_code_proxy.providers.codex.translation import CodexEventTranslator, build_request, response_from_events
from claude_code_proxy.reasoning import ReasoningPolicy


def request(**changes):
    base = CompletionRequest(
        original_model="claude-sonnet",
        model="openai/gpt-5.6-sol",
        response_model="claude-sonnet[1m]",
        max_tokens=100,
        messages=(Message("user", (TextBlock("hello"),)),),
        reasoning=ReasoningPolicy(True, "high"),
    )
    return replace(base, **changes)



def test_build_request_adds_shared_cache_and_turn_metadata():
    identity = CodexIdentity.from_client(
        ClientIdentity("session", "agent", "parent")
    )

    payload = build_request(request(), identity)

    assert payload["prompt_cache_key"] == "session"
    assert payload["client_metadata"]["session_id"] == "session"
    assert payload["client_metadata"]["thread_id"] == identity.thread_id
    assert json.loads(
        payload["client_metadata"]["x-codex-turn-metadata"]
    ) == {
        "parent_thread_id": identity.parent_thread_id,
        "session_id": "session",
        "thread_id": identity.thread_id,
    }

def test_build_request_maps_tools_messages_and_reasoning():
    body = build_request(request(
        system=(TextBlock("system"),),
        messages=(Message("assistant", (TextBlock("before"), ToolUseBlock("call-1", "lookup", {"q": "x"}))), Message("user", (ToolResultBlock("call-1", "done"),))),
        tools=(ToolDefinition("lookup", "Lookup", {"type": "object"}), ToolDefinition("builtin")),
        tool_choice=ToolChoice(type="tool", name="lookup"),
    ))
    assert body["model"] == "gpt-5.6-sol"
    assert body["reasoning"] == {"effort": "high"}
    assert body["instructions"] == "system"
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert [tool["name"] for tool in body["tools"]] == ["lookup"]
    assert all(tool["strict"] is False for tool in body["tools"])
    assert body["input"][1]["type"] == "function_call"
    assert body["input"][2] == {"type": "function_call_output", "call_id": "call-1", "output": "done"}


def test_build_request_preserves_optional_tool_schema():
    monitor_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "ws": {"type": "object"},
        },
    }

    body = build_request(request(tools=(
        ToolDefinition("Monitor", input_schema=monitor_schema),
    )))

    assert body["tools"][0]["strict"] is False
    assert body["tools"][0]["parameters"] is monitor_schema


def test_reasoning_enabled_request_asks_for_encrypted_content():
    body = build_request(request())

    assert body["include"] == ["reasoning.encrypted_content"]


def test_reasoning_disabled_request_omits_encrypted_include():
    body = build_request(request(reasoning=ReasoningPolicy(False, None)))

    assert "include" not in body


def test_replays_valid_reasoning_before_tool_call_and_output():
    carrier = encode_reasoning("encrypted-state", [])
    prepared = request(messages=(
        Message("assistant", (
            RedactedThinkingBlock(carrier),
            ToolUseBlock("call-1", "lookup", {"q": "x"}),
        )),
        Message("user", (ToolResultBlock("call-1", "done"),)),
    ))

    assert build_request(prepared)["input"] == [
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "encrypted-state",
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"q": "x"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "done",
        },
    ]


def test_omits_foreign_and_malformed_reasoning_carriers():
    prepared = request(messages=(Message("assistant", (
        RedactedThinkingBlock("anthropic-ciphertext"),
        RedactedThinkingBlock("codex-reasoning-v1:not-base64!"),
        ToolUseBlock("call-1", "lookup", {}),
    )),))

    assert [item["type"] for item in build_request(prepared)["input"]] == [
        "function_call"
    ]


def test_replays_one_reasoning_item_before_parallel_tool_calls():
    carrier = encode_reasoning("encrypted-state", [])
    prepared = request(messages=(Message("assistant", (
        RedactedThinkingBlock(carrier),
        ToolUseBlock("call-1", "first", {}),
        ToolUseBlock("call-2", "second", {}),
    )),))

    assert [item["type"] for item in build_request(prepared)["input"]] == [
        "reasoning",
        "function_call",
        "function_call",
    ]


def test_preserves_multiple_reasoning_tool_groups():
    first = encode_reasoning("encrypted-1", [])
    second = encode_reasoning("encrypted-2", [])
    prepared = request(messages=(Message("assistant", (
        RedactedThinkingBlock(first),
        ToolUseBlock("call-1", "first", {}),
        RedactedThinkingBlock(second),
        ToolUseBlock("call-2", "second", {}),
    )),))

    items = build_request(prepared)["input"]

    assert [item["type"] for item in items] == [
        "reasoning",
        "function_call",
        "reasoning",
        "function_call",
    ]
    assert [
        item["encrypted_content"] for item in items if item["type"] == "reasoning"
    ] == ["encrypted-1", "encrypted-2"]


def test_missing_selected_tool_falls_back_to_auto():
    body = build_request(request(tools=(ToolDefinition("builtin"),), tool_choice=ToolChoice(type="tool", name="builtin")))
    assert body["tool_choice"] == "auto"


def test_event_translator_maps_text_tools_usage_and_stop():
    translator = CodexEventTranslator()
    events = []
    events += translator.feed("response.output_text.delta", {"delta": "hello"})
    events += translator.feed("response.output_item.added", {"output_index": 2, "item": {"type": "function_call", "call_id": "call-1", "name": "lookup"}})
    events += translator.feed("response.function_call_arguments.delta", {"output_index": 2, "delta": '{"q":"x"}'})
    events += translator.feed("response.function_call_arguments.done", {"output_index": 2})
    translator.feed("response.completed", {"usage": {"input_tokens": 4, "output_tokens": 2}, "status": "completed"})
    assert events == [TextDelta("hello"), ToolUseStart("2", "call-1", "lookup"), ToolInputDelta("2", '{"q":"x"}'), ToolUseEnd("2")]
    assert translator.finish() == StreamComplete("tool_use", TokenUsage(4, 2))



def test_completion_maps_nested_cache_and_reasoning_usage():
    translator = CodexEventTranslator()

    translator.feed("response.completed", {"response": {
        "status": "completed",
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 60,
                "cache_write_tokens": 10,
            },
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 7},
        },
    }})

    assert translator.finish().usage == TokenUsage(30, 20, 10, 60, 7)


def test_incomplete_response_preserves_detailed_usage():
    translator = CodexEventTranslator()

    translator.feed("response.incomplete", {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    })

    assert translator.finish() == StreamComplete(
        "max_tokens", TokenUsage(6, 5, 0, 4, 3)
    )


def test_completed_reasoning_item_emits_opaque_carrier():
    translator = CodexEventTranslator()

    added = translator.feed(
        "response.output_item.added",
        {"output_index": 0, "item": {
            "type": "reasoning",
            "id": "rs-1",
            "encrypted_content": None,
        }},
    )
    done = translator.feed(
        "response.output_item.done",
        {"output_index": 0, "item": {
            "type": "reasoning",
            "id": "rs-1",
            "summary": [],
            "encrypted_content": "encrypted-state",
        }},
    )

    assert added == ()
    assert len(done) == 1
    assert isinstance(done[0], RedactedThinking)
    assert decode_reasoning(done[0].data) == {
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "encrypted-state",
    }


def test_completed_reasoning_without_encrypted_content_emits_nothing():
    translator = CodexEventTranslator()

    assert translator.feed(
        "response.output_item.done",
        {"item": {"type": "reasoning", "summary": []}},
    ) == ()


def test_response_from_events_preserves_reasoning_before_tool():
    carrier = encode_reasoning("encrypted-state", [])

    response = response_from_events(request(), [
        RedactedThinking(carrier),
        ToolUseStart("0", "call", "lookup"),
        ToolInputDelta("0", '{}'),
        ToolUseEnd("0"),
        StreamComplete("tool_use", TokenUsage(2, 3)),
    ])

    assert response.content == (
        RedactedThinkingBlock(carrier),
        ToolUseBlock("call", "lookup", {}),
    )


def test_response_from_events_preserves_text_segments_around_reasoning():
    carrier = encode_reasoning("encrypted-state", [])

    response = response_from_events(request(), [
        TextDelta("before"),
        RedactedThinking(carrier),
        TextDelta("after"),
        StreamComplete("end_turn", TokenUsage(2, 3)),
    ])

    assert response.content == (
        TextBlock("before"),
        RedactedThinkingBlock(carrier),
        TextBlock("after"),
    )


def test_response_from_events_keeps_text_segment_before_reasoning():
    carrier = encode_reasoning("encrypted-state", [])

    response = response_from_events(request(), [
        TextDelta("before"),
        RedactedThinking(carrier),
        ToolUseStart("0", "call", "lookup"),
        ToolInputDelta("0", '{}'),
        ToolUseEnd("0"),
        StreamComplete("tool_use", TokenUsage(2, 3)),
    ])

    assert response.content == (
        TextBlock("before"),
        RedactedThinkingBlock(carrier),
        ToolUseBlock("call", "lookup", {}),
    )


def test_reasoning_carrier_round_trips_through_anthropic_history():
    translator = CodexEventTranslator()
    reasoning = translator.feed(
        "response.output_item.done",
        {"item": {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "encrypted-state",
        }},
    )[0]
    events = [
        reasoning,
        ToolUseStart("0", "call-1", "lookup"),
        ToolInputDelta("0", '{"q":"x"}'),
        ToolUseEnd("0"),
        StreamComplete("tool_use", TokenUsage(2, 3)),
    ]
    response = response_from_events(request(), events)
    api_content = [
        block.model_dump() for block in to_api_response(response).content
    ]
    follow_up = MessagesRequest(
        model="claude-sonnet",
        max_tokens=100,
        messages=[
            {"role": "assistant", "content": api_content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "done",
                }],
            },
        ],
    )

    codex = build_request(normalize_request(follow_up))

    assert codex["input"] == [
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "encrypted-state",
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"q": "x"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "done",
        },
    ]


def test_arguments_done_supplies_json_when_no_deltas_arrive():
    translator = CodexEventTranslator()
    translator.feed("response.output_item.added", {"output_index": 0, "item": {"type": "function_call", "call_id": "call-1", "name": "lookup"}})
    assert translator.feed("response.function_call_arguments.done", {"output_index": 0, "arguments": '{"q":"x"}'}) == (
        ToolInputDelta("0", '{"q":"x"}'),
        ToolUseEnd("0"),
    )

def test_incomplete_response_maps_max_tokens():
    translator = CodexEventTranslator()
    translator.feed("response.incomplete", {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "usage": {}})

    assert translator.completed is True
    assert translator.finish().stop_reason == "max_tokens"


def test_response_from_events_uses_client_response_model():
    response = response_from_events(
        request(),
        [StreamComplete("end_turn", TokenUsage(2, 3))],
    )

    assert response.model == "claude-sonnet[1m]"


def test_response_from_events_buffers_text_and_tool_json():
    response = response_from_events(request(), [TextDelta("hi"), ToolUseStart("0", "call", "lookup"), ToolInputDelta("0", '{"q":"x"}'), ToolUseEnd("0"), StreamComplete("tool_use", TokenUsage(2, 3))])
    assert response.content == (TextBlock("hi"), ToolUseBlock("call", "lookup", {"q": "x"}))
    assert response.usage == TokenUsage(2, 3)


def _monitor_response(arguments):
    return response_from_events(
        request(),
        [
            ToolUseStart("0", "call-monitor", "Monitor"),
            ToolInputDelta("0", arguments),
            ToolUseEnd("0"),
            StreamComplete("tool_use", TokenUsage(2, 3)),
        ],
    )


def test_monitor_command_arguments_preserve_omitted_websocket():
    response = _monitor_response(
        '{"description":"job","timeout_ms":1000,"command":"tail -F run.log"}'
    )

    tool = response.content[0]
    assert tool.input == {
        "description": "job",
        "timeout_ms": 1000,
        "command": "tail -F run.log",
    }
    assert "ws" not in tool.input


def test_monitor_websocket_arguments_preserve_omitted_command():
    response = _monitor_response(
        '{"description":"events","timeout_ms":1000,'
        '"ws":{"url":"wss://events.example.com","protocols":[]}}'
    )

    tool = response.content[0]
    assert tool.input == {
        "description": "events",
        "timeout_ms": 1000,
        "ws": {
            "url": "wss://events.example.com",
            "protocols": [],
        },
    }
    assert "command" not in tool.input
