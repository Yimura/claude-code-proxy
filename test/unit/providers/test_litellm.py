import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from litellm.types.utils import Usage as LiteLLMUsage

from claude_code_proxy.config import Settings
from claude_code_proxy.domain.models import (
    CompletionRequest,
    ImageBlock,
    Message,
    StreamComplete,
    StreamError,
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
    return Settings(
        anthropic_api_key="anthropic-key",
        openai_api_key="openai-key",
        gemini_api_key="gemini-key",
        vertex_project="project",
        vertex_location="region",
        use_vertex_auth=False,
        openai_base_url=None,
        openai_transport="litellm",
        opencode_data_dir=Path("/auth"),
        model_mapping_path=Path("mapping.json"),
    )


def request(model="openai/gpt-5.6-sol", **changes):
    base = CompletionRequest(
        original_model=model,
        model=model,
        response_model=model,
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



def test_openai_token_cap_does_not_depend_on_transport(settings):
    codex_settings = replace(settings, openai_transport="codex")

    payload = LiteLLMProvider(codex_settings, object()).build_request(
        request(max_tokens=128_000), stream=False
    )

    assert payload["max_completion_tokens"] == 16_384

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
async def test_complete_uses_client_response_model(settings):
    client = FakeClient({
        "id": "response-1",
        "choices": [{
            "message": {"content": "hello", "tool_calls": None},
            "finish_reason": "stop",
        }],
        "usage": {},
    })

    response = await LiteLLMProvider(settings, client).complete(
        request(response_model="claude-opus-5[1m]")
    )

    assert response.model == "claude-opus-5[1m]"


@pytest.mark.asyncio
async def test_complete_preserves_nested_cache_and_reasoning_usage(settings):
    client = FakeClient({
        "id": "response-usage",
        "choices": [{
            "message": {"content": "hello", "tool_calls": None},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {
                "cached_tokens": 60,
                "cache_write_tokens": 10,
            },
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    })

    response = await LiteLLMProvider(settings, client).complete(request())

    assert response.usage == TokenUsage(30, 20, 10, 60, 7)


@pytest.mark.asyncio
async def test_complete_accepts_litellm_usage_model(settings):
    usage = LiteLLMUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details={
            "cached_tokens": 60,
            "cache_write_tokens": 10,
        },
        completion_tokens_details={"reasoning_tokens": 7},
    )
    client = FakeClient({
        "id": "response-model-usage",
        "choices": [{
            "message": {"content": "hello", "tool_calls": None},
            "finish_reason": "stop",
        }],
        "usage": usage,
    })

    response = await LiteLLMProvider(settings, client).complete(request())

    assert response.usage == TokenUsage(30, 20, 10, 60, 7)


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
async def test_stream_replaces_cumulative_detailed_usage(settings):
    client = FakeClient(chunks=[
        {
            "choices": [{"delta": {"content": "hel"}, "finish_reason": None}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
        },
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {
                    "cached_tokens": 60,
                    "cache_creation_tokens": 10,
                },
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        },
    ])

    events = [
        event async for event in LiteLLMProvider(settings, client).stream(request())
    ]

    assert events[-1] == StreamComplete(
        "end_turn", TokenUsage(30, 20, 10, 60, 7)
    )


@pytest.mark.asyncio
async def test_stream_returns_semantic_tool_events(settings):
    client = FakeClient(chunks=[
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "lookup", "arguments": "{\"q\":"}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"x\"}"}}]}, "finish_reason": "tool_calls"}]},
    ])
    events = [event async for event in LiteLLMProvider(settings, client).stream(request())]
    assert events == [StreamStart(), ToolUseStart("0", "call-1", "lookup"), ToolInputDelta("0", '{"q":'), ToolInputDelta("0", '"x"}'), ToolUseEnd("0"), StreamComplete("tool_use", TokenUsage(0, 0))]


@pytest.mark.asyncio
async def test_stream_requires_upstream_finish_reason(settings):
    client = FakeClient(chunks=[
        {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
    ])

    events = [event async for event in LiteLLMProvider(settings, client).stream(request())]

    assert events[:2] == [StreamStart(), TextDelta("partial")]
    assert isinstance(events[-1], StreamError)
    assert events[-1].message == "Internal server error"


class PlainUpstream:
    def __init__(self):
        self.events = iter([
            {
                "choices": [
                    {"delta": {"content": "done"}, "finish_reason": "stop"}
                ]
            }
        ])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration as error:
            raise StopAsyncIteration from error


@pytest.mark.asyncio
async def test_stream_accepts_upstream_without_aclose(settings):
    events = [
        event
        async for event in LiteLLMProvider(
            settings, ClosableClient(PlainUpstream())
        ).stream(request())
    ]

    assert events == [
        StreamStart(),
        TextDelta("done"),
        StreamComplete("end_turn", TokenUsage(0, 0)),
    ]


class ClosableUpstream:
    def __init__(self):
        self.sent = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.sent:
            self.sent = True
            return {"choices": [{"delta": {"content": "partial"}}]}
        await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class ClosableClient(FakeClient):
    def __init__(self, upstream):
        super().__init__()
        self.upstream = upstream

    async def acompletion(self, **kwargs):
        return self.upstream


@pytest.mark.asyncio
async def test_closing_provider_stream_closes_upstream_iterator(settings):
    upstream = ClosableUpstream()
    stream = LiteLLMProvider(settings, ClosableClient(upstream)).stream(request())

    assert await anext(stream) == StreamStart()
    assert await anext(stream) == TextDelta("partial")

    await stream.aclose()

    assert upstream.closed is True


class FailingClient(FakeClient):
    async def acompletion(self, **kwargs):
        raise RuntimeError("requested model is unavailable")


@pytest.mark.asyncio
async def test_stream_exception_preserves_useful_provider_message(settings):
    events = [event async for event in LiteLLMProvider(settings, FailingClient()).stream(request())]

    assert events == [
        StreamError(
            error_type="api_error",
            message="requested model is unavailable",
            status_code=500,
            retryable=True,
            provider="litellm",
            diagnostic="requested model is unavailable",
        )
    ]


@pytest.mark.asyncio
async def test_count_tokens_uses_local_counter(settings):
    client = FakeClient(token_count=17)
    assert await LiteLLMProvider(settings, client).count_tokens(request()) == 17
    assert client.counter_args["model"] == "openai/gpt-5.6-sol"


@pytest.mark.asyncio
async def test_count_tokens_preserves_import_fallback(settings):
    assert await LiteLLMProvider(settings, object()).count_tokens(request()) == 1000
