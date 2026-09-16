from dataclasses import replace
from pathlib import Path

import pytest

from claude_code_proxy.config import Settings
from claude_code_proxy.domain.models import (
    CompletionRequest,
    ImageBlock,
    Message,
    StreamComplete,
    StreamStart,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
    ToolInputDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseEnd,
    ToolUseStart,
)
from claude_code_proxy.providers.litellm import LiteLLMProvider, clean_gemini_schema
from claude_code_proxy.reasoning import OutputConfig, ReasoningPolicy, ThinkingConfig


@pytest.fixture
def settings():
    return Settings("anthropic-key", "openai-key", "gemini-key", "project", "region", False, None, "openai", "big", "small", Path("/auth"), Path("mapping.json"))


def request(model="openai/gpt-5.6-sol", **changes):
    base = CompletionRequest(
        original_model=model,
        model=model,
        max_tokens=20000,
        messages=(Message("user", (TextBlock("hello"),)),),
        reasoning=ReasoningPolicy(True, "high"),
    )
    return replace(base, **changes)


def test_build_request_preserves_tools_reasoning_and_auth(settings):
    provider = LiteLLMProvider(settings, object())
    payload = provider.build_request(request(
        tools=(ToolDefinition("lookup", "Lookup", {"type": "object"}),),
        tool_choice=ToolChoice(type="tool", name="lookup"),
    ), stream=False)
    assert payload["max_completion_tokens"] == 16384
    assert payload["reasoning_effort"] == "high"
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert payload["api_key"] == "openai-key"


def test_missing_selected_tool_falls_back_to_auto(settings):
    payload = LiteLLMProvider(settings, object()).build_request(request(
        tools=(ToolDefinition("lookup", input_schema={"type": "object"}), ToolDefinition("builtin")),
        tool_choice=ToolChoice(type="tool", name="builtin"),
    ), stream=False)
    assert [tool["function"]["name"] for tool in payload["tools"]] == ["lookup"]
    assert payload["tool_choice"] == "auto"


def test_tool_result_is_flattened_for_openai(settings):
    payload = LiteLLMProvider(settings, object()).build_request(request(messages=(
        Message("assistant", (ToolUseBlock("call-1", "lookup", {"q": "x"}),)),
        Message("user", (ToolResultBlock("call-1", [{"type": "text", "text": "done"}]),)),
    )), stream=False)
    assert payload["messages"][0]["content"].startswith("[Tool: lookup")
    assert payload["messages"][1]["content"] == "Tool result for call-1:\ndone"


def test_gemini_schema_cleaning_and_vertex_auth(settings):
    vertex = replace(settings, use_vertex_auth=True)
    payload = LiteLLMProvider(vertex, object()).build_request(request(
        model="gemini/gemini-2.5-pro",
        tools=(ToolDefinition("lookup", input_schema={"type": "object", "additionalProperties": False, "properties": {"date": {"type": "string", "format": "date"}}}),),
    ), stream=False)
    parameters = payload["tools"][0]["function"]["parameters"]
    assert "additionalProperties" not in parameters
    assert "format" not in parameters["properties"]["date"]
    assert payload["custom_llm_provider"] == "vertex_ai"


def test_anthropic_preserves_thinking_and_output_config(settings):
    payload = LiteLLMProvider(settings, object()).build_request(request(
        model="anthropic/claude-opus-5",
        thinking=ThinkingConfig(type="adaptive"),
        output_config=OutputConfig(effort="high", format={"type": "json_schema"}),
    ), stream=False)
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high", "format": {"type": "json_schema"}}
    assert "reasoning_effort" not in payload


class FakeClient:
    def __init__(self, response=None, chunks=(), token_count=9):
        self.response = response
        self.chunks = chunks
        self.token_count = token_count
        self.counter_args = None

    def completion(self, **kwargs):
        return self.response

    async def acompletion(self, **kwargs):
        async def generate():
            for chunk in self.chunks:
                yield chunk
        return generate()

    def token_counter(self, **kwargs):
        self.counter_args = kwargs
        return self.token_count


@pytest.mark.asyncio
async def test_complete_returns_normalized_text_and_usage(settings):
    client = FakeClient({"id": "response-1", "choices": [{"message": {"content": "hello", "tool_calls": None}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}})
    response = await LiteLLMProvider(settings, client).complete(request())
    assert response.content == (TextBlock("hello"),)
    assert response.stop_reason == "end_turn"
    assert response.usage == TokenUsage(4, 2)


@pytest.mark.asyncio
async def test_complete_normalizes_tool_call_and_invalid_arguments(settings):
    client = FakeClient({"id": "response-1", "choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": "not-json"}}]}, "finish_reason": "tool_calls"}], "usage": {}})
    response = await LiteLLMProvider(settings, client).complete(request())
    assert response.content == (ToolUseBlock("call-1", "lookup", {"raw": "not-json"}),)
    assert response.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_stream_returns_semantic_text_events(settings):
    client = FakeClient(chunks=[
        {"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    ])
    events = [event async for event in LiteLLMProvider(settings, client).stream(request())]
    assert events == [StreamStart(), TextDelta("hel"), TextDelta("lo"), StreamComplete("end_turn", TokenUsage(3, 2))]


@pytest.mark.asyncio
async def test_stream_returns_semantic_tool_events(settings):
    client = FakeClient(chunks=[
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "lookup", "arguments": "{\"q\":"}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"x\"}"}}]}, "finish_reason": "tool_calls"}]},
    ])
    events = [event async for event in LiteLLMProvider(settings, client).stream(request())]
    assert events == [StreamStart(), ToolUseStart("0", "call-1", "lookup"), ToolInputDelta("0", '{"q":'), ToolInputDelta("0", '"x"}'), ToolUseEnd("0"), StreamComplete("tool_use", TokenUsage(0, 0))]


@pytest.mark.asyncio
async def test_count_tokens_uses_local_counter(settings):
    client = FakeClient(token_count=17)
    assert await LiteLLMProvider(settings, client).count_tokens(request()) == 17
    assert client.counter_args["model"] == "openai/gpt-5.6-sol"


@pytest.mark.asyncio
async def test_count_tokens_preserves_import_fallback(settings):
    assert await LiteLLMProvider(settings, object()).count_tokens(request()) == 1000
