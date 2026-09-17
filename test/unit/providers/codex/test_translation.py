from dataclasses import replace

from claude_code_proxy.domain.models import (
    CompletionRequest, Message, RedactedThinking, RedactedThinkingBlock, StreamComplete, TextBlock, TextDelta, TokenUsage,
    ToolChoice, ToolDefinition, ToolInputDelta, ToolResultBlock, ToolUseBlock,
    ToolUseEnd, ToolUseStart,
)
from claude_code_proxy.providers.codex.reasoning import decode_reasoning, encode_reasoning
from claude_code_proxy.providers.codex.translation import CodexEventTranslator, build_request, response_from_events
from claude_code_proxy.reasoning import ReasoningPolicy


def request(**changes):
    base = CompletionRequest("claude-sonnet", "openai/gpt-5.6-sol", 100, (Message("user", (TextBlock("hello"),)),), ReasoningPolicy(True, "high"))
    return replace(base, **changes)


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
    assert body["input"][1]["type"] == "function_call"
    assert body["input"][2] == {"type": "function_call_output", "call_id": "call-1", "output": "done"}


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


def test_response_from_events_buffers_text_and_tool_json():
    response = response_from_events(request(), [TextDelta("hi"), ToolUseStart("0", "call", "lookup"), ToolInputDelta("0", '{"q":"x"}'), ToolUseEnd("0"), StreamComplete("tool_use", TokenUsage(2, 3))])
    assert response.content == (TextBlock("hi"), ToolUseBlock("call", "lookup", {"q": "x"}))
    assert response.usage == TokenUsage(2, 3)
