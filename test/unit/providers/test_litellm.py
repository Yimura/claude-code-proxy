import asyncio
from dataclasses import replace
import logging
from pathlib import Path
import re

import httpx
import litellm
import pytest
from litellm.types.utils import Usage as LiteLLMUsage

from claude_code_proxy.api.schemas import MessagesRequest
from claude_code_proxy.api.translation import normalize_request, serialize_stream
from claude_code_proxy.config import Settings
from claude_code_proxy.domain.models import (
    ClientIdentity, CompletionRequest,
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
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.logging import (
    RequestLogContext,
    SessionIdentity,
    log_stream_failure,
)
from claude_code_proxy.providers.base import ProviderError
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


def assert_failure_evidence(
    diagnostic,
    category,
    stage,
    code,
    exception_type,
):
    assert diagnostic.category == category
    assert diagnostic.stage == stage
    assert diagnostic.code == code
    assert diagnostic.exception_type == exception_type
    assert re.fullmatch(
        r"claude_code_proxy\.providers\.litellm:[^:]+:\d+",
        diagnostic.location,
    )


def log_context():
    return RequestLogContext(
        session=SessionIdentity("session", "[session session]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude",
        upstream_model="openai/gpt-5.6-sol",
        provider="litellm",
        effort="default",
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



def test_build_request_does_not_forward_client_identity(settings):
    provider = LiteLLMProvider(settings, object())
    payload = provider.build_request(
        request(
            client_identity=ClientIdentity("session", "agent", "parent")
        ),
        stream=False,
    )

    assert "session_id" not in payload
    assert "agent_id" not in payload
    assert "parent_agent_id" not in payload
    assert "client_metadata" not in payload
    assert "prompt_cache_key" not in payload

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

    assert events == [
        StreamStart(),
        TextDelta("partial"),
        StreamError(
            provider="litellm",
            diagnostic=FailureDiagnostic(
                FailureCategory.PROVIDER_PROTOCOL,
                FailureStage.STREAM,
                "missing_finish_reason",
            ),
        ),
    ]


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


class RaisingUpstream:
    def __init__(
        self,
        error=StopAsyncIteration(),
        chunks=(),
        close_error=None,
        block=False,
    ):
        self.error = error
        self.chunks = iter(chunks)
        self.close_error = close_error
        self.block = block
        self.waiting = asyncio.Event()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            pass
        if self.block:
            self.waiting.set()
            await asyncio.Event().wait()
        raise self.error

    async def aclose(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


@pytest.mark.asyncio
async def test_closing_provider_stream_closes_upstream_iterator(settings):
    upstream = ClosableUpstream()
    stream = LiteLLMProvider(settings, ClosableClient(upstream)).stream(request())

    assert await anext(stream) == StreamStart()
    assert await anext(stream) == TextDelta("partial")

    await stream.aclose()

    assert upstream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            litellm.Timeout("secret timeout", "model", "openai"),
            StreamError(
                error_type="timeout_error",
                message="Request timed out",
                status_code=504,
                provider="litellm",
                diagnostic=FailureDiagnostic(
                    FailureCategory.TRANSPORT,
                    FailureStage.STREAM,
                    "timeout",
                ),
            ),
        ),
        (
            RuntimeError("secret iterator failure"),
            StreamError(
                status_code=500,
                provider="litellm",
                diagnostic=FailureDiagnostic(
                    FailureCategory.INTERNAL,
                    FailureStage.STREAM,
                    "stream_failed",
                ),
            ),
        ),
    ],
)
async def test_iteration_failure_closes_before_emitting_terminal_error(
    settings, error, expected
):
    upstream = RaisingUpstream(error=error)
    stream = LiteLLMProvider(settings, ClosableClient(upstream)).stream(request())

    assert await anext(stream) == StreamStart()
    terminal = await anext(stream)

    if isinstance(error, RuntimeError):
        assert terminal.status_code == expected.status_code
        assert terminal.provider == expected.provider
        assert_failure_evidence(
            terminal.diagnostic,
            FailureCategory.INTERNAL,
            FailureStage.STREAM,
            "stream_failed",
            "RuntimeError",
        )
    else:
        assert terminal == expected
    assert upstream.closed is True
    assert "secret" not in repr(terminal)


@pytest.mark.asyncio
async def test_iteration_provider_error_is_preserved_after_cleanup(settings):
    diagnostic = FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.STREAM,
        "upstream_stream_failed",
    )
    upstream = RaisingUpstream(
        error=ProviderError(
            "Safe provider failure",
            provider="upstream",
            status_code=429,
            diagnostic=diagnostic,
        )
    )
    stream = LiteLLMProvider(settings, ClosableClient(upstream)).stream(request())

    assert await anext(stream) == StreamStart()
    terminal = await anext(stream)

    assert terminal == StreamError(
        error_type="rate_limit_error",
        message="Safe provider failure",
        status_code=429,
        provider="upstream",
        diagnostic=diagnostic,
    )
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_natural_eof_cleanup_failure_becomes_structured_error(settings):
    upstream = RaisingUpstream(close_error=RuntimeError("secret close failure"))

    events = [
        event
        async for event in LiteLLMProvider(
            settings, ClosableClient(upstream)
        ).stream(request())
    ]

    assert events[0] == StreamStart()
    assert len(events) == 2
    assert events[1].status_code == 500
    assert events[1].provider == "litellm"
    assert_failure_evidence(
        events[1].diagnostic,
        FailureCategory.INTERNAL,
        FailureStage.STREAM,
        "stream_cleanup_failed",
        "RuntimeError",
    )
    assert upstream.closed is True
    assert "secret close failure" not in repr(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_error", "expected"),
    [
        (
            litellm.Timeout("secret cleanup timeout", "model", "openai"),
            StreamError(
                error_type="timeout_error",
                message="Request timed out",
                status_code=504,
                provider="litellm",
                diagnostic=FailureDiagnostic(
                    FailureCategory.TRANSPORT,
                    FailureStage.STREAM,
                    "timeout",
                ),
            ),
        ),
        (
            litellm.APIConnectionError(
                "secret cleanup connection", "openai", "model"
            ),
            StreamError(
                status_code=503,
                provider="litellm",
                diagnostic=FailureDiagnostic(
                    FailureCategory.TRANSPORT,
                    FailureStage.STREAM,
                    "transport_error",
                ),
            ),
        ),
    ],
)
async def test_typed_cleanup_failure_preserves_typed_semantics(
    settings, cleanup_error, expected
):
    upstream = RaisingUpstream(close_error=cleanup_error)

    events = [
        event
        async for event in LiteLLMProvider(
            settings, ClosableClient(upstream)
        ).stream(request())
    ]

    assert events == [StreamStart(), expected]
    assert upstream.closed is True
    assert "secret" not in repr(events)


@pytest.mark.asyncio
async def test_primary_provider_error_wins_over_cleanup_failure(settings):
    primary = provider_failure()
    upstream = RaisingUpstream(
        error=primary,
        close_error=RuntimeError("secret cleanup failure"),
    )

    events = [
        event
        async for event in LiteLLMProvider(
            settings, ClosableClient(upstream)
        ).stream(request())
    ]

    assert events == [
        StreamStart(),
        StreamError(
            error_type="rate_limit_error",
            message="Safe provider failure",
            status_code=429,
            provider="upstream",
            diagnostic=primary.diagnostic,
        ),
    ]
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_primary_timeout_wins_over_cleanup_failure(settings):
    upstream = RaisingUpstream(
        error=litellm.Timeout("secret timeout", "model", "openai"),
        close_error=RuntimeError("secret cleanup failure"),
    )

    events = [
        event
        async for event in LiteLLMProvider(
            settings, ClosableClient(upstream)
        ).stream(request())
    ]

    assert events[-1].diagnostic == FailureDiagnostic(
        FailureCategory.TRANSPORT,
        FailureStage.STREAM,
        "timeout",
    )
    assert events[-1].status_code == 504
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_primary_translation_failure_wins_over_cleanup_failure(settings):
    upstream = RaisingUpstream(
        chunks=[object()],
        close_error=RuntimeError("secret cleanup failure"),
    )

    events = [
        event
        async for event in LiteLLMProvider(
            settings, ClosableClient(upstream)
        ).stream(request())
    ]

    assert_failure_evidence(
        events[-1].diagnostic,
        FailureCategory.TRANSLATION,
        FailureStage.PROVIDER_TRANSLATION,
        "stream_chunk_translation_failed",
        "AttributeError",
    )
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_cleanup_cancellation_propagates(settings):
    upstream = RaisingUpstream(close_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        async for _ in LiteLLMProvider(
            settings, ClosableClient(upstream)
        ).stream(request()):
            pass

    assert upstream.closed is True


@pytest.mark.asyncio
async def test_successful_terminal_event_is_emitted_after_cleanup(settings):
    upstream = RaisingUpstream(
        chunks=[{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
    )
    stream = LiteLLMProvider(settings, ClosableClient(upstream)).stream(request())

    assert await anext(stream) == StreamStart()
    terminal = await anext(stream)

    assert terminal == StreamComplete("end_turn", TokenUsage(0, 0))
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_external_close_is_not_replaced_by_cleanup_failure(settings):
    upstream = RaisingUpstream(
        chunks=[
            {
                "choices": [
                    {"delta": {"content": "partial"}, "finish_reason": None}
                ]
            }
        ],
        close_error=RuntimeError("secret close failure"),
        block=True,
    )
    stream = LiteLLMProvider(settings, ClosableClient(upstream)).stream(request())

    assert await anext(stream) == StreamStart()
    assert await anext(stream) == TextDelta("partial")

    await stream.aclose()

    assert upstream.closed is True


@pytest.mark.asyncio
async def test_cancellation_is_not_replaced_by_cleanup_failure(settings):
    upstream = RaisingUpstream(
        close_error=RuntimeError("secret close failure"),
        block=True,
    )
    stream = LiteLLMProvider(settings, ClosableClient(upstream)).stream(request())
    assert await anext(stream) == StreamStart()
    pending = asyncio.create_task(anext(stream))
    await upstream.waiting.wait()

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert upstream.closed is True


class FailingClient(FakeClient):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def completion(self, **kwargs):
        raise self.error

    async def acompletion(self, **kwargs):
        raise self.error

    def token_counter(self, **kwargs):
        raise self.error


def provider_failure():
    return ProviderError(
        "Safe provider failure",
        provider="upstream",
        status_code=429,
        diagnostic=FailureDiagnostic(
            FailureCategory.UPSTREAM_HTTP,
            FailureStage.REQUEST,
            "upstream_failure",
        ),
    )


def _http_response(status_code):
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://provider.example/v1/messages"),
    )


def _typed_litellm_errors():
    authentication = litellm.AuthenticationError(
        "secret auth detail", "openai", "model"
    )
    permission = litellm.PermissionDeniedError(
        "secret permission detail", "openai", "model", _http_response(403)
    )
    timeout = litellm.Timeout("secret timeout detail", "model", "openai")
    connection = litellm.APIConnectionError(
        "secret connection detail", "openai", "model"
    )
    status = litellm.APIError(529, "secret upstream detail", "openai", "model")
    status.code = "provider-overloaded"
    return [
        (
            authentication,
            401,
            "Authentication failed",
            FailureDiagnostic(
                FailureCategory.AUTHENTICATION,
                FailureStage.REQUEST,
                "authentication_error",
            ),
        ),
        (
            permission,
            403,
            "Permission denied",
            FailureDiagnostic(
                FailureCategory.AUTHENTICATION,
                FailureStage.REQUEST,
                "permission_denied",
            ),
        ),
        (
            timeout,
            504,
            "Request timed out",
            FailureDiagnostic(
                FailureCategory.TRANSPORT,
                FailureStage.REQUEST,
                "timeout",
            ),
        ),
        (
            connection,
            503,
            "Internal server error",
            FailureDiagnostic(
                FailureCategory.TRANSPORT,
                FailureStage.REQUEST,
                "transport_error",
            ),
        ),
        (
            status,
            529,
            "Overloaded",
            FailureDiagnostic(
                FailureCategory.UPSTREAM_HTTP,
                FailureStage.REQUEST,
                "upstream_http_error",
                "provider-overloaded",
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_complete_preserves_provider_error(settings):
    error = provider_failure()

    with pytest.raises(ProviderError) as caught:
        await LiteLLMProvider(settings, FailingClient(error)).complete(request())

    assert caught.value is error


@pytest.mark.asyncio
async def test_stream_invocation_preserves_provider_error(settings):
    error = provider_failure()

    events = [
        event
        async for event in LiteLLMProvider(
            settings, FailingClient(error)
        ).stream(request())
    ]

    assert events == [
        StreamError(
            error_type="rate_limit_error",
            message="Safe provider failure",
            status_code=429,
            provider="upstream",
            diagnostic=error.diagnostic,
        )
    ]


@pytest.mark.asyncio
async def test_count_tokens_preserves_provider_error(settings):
    error = provider_failure()

    with pytest.raises(ProviderError) as caught:
        await LiteLLMProvider(settings, FailingClient(error)).count_tokens(
            request()
        )

    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code", "message", "diagnostic"),
    _typed_litellm_errors(),
)
async def test_stream_classifies_typed_invocation_errors(
    settings, error, status_code, message, diagnostic
):
    events = [
        event
        async for event in LiteLLMProvider(
            settings, FailingClient(error)
        ).stream(request())
    ]

    assert events == [
        StreamError(
            error_type={
                401: "authentication_error",
                403: "permission_error",
                504: "timeout_error",
                529: "overloaded_error",
            }.get(status_code, "api_error"),
            message=message,
            status_code=status_code,
            retryable=status_code >= 500,
            provider="litellm",
            diagnostic=diagnostic,
        )
    ]
    assert events[0].diagnostic.exception_type is None
    assert events[0].diagnostic.location is None
    assert "secret" not in repr(events)


@pytest.mark.asyncio
async def test_stream_unknown_invocation_error_is_internal_and_safe(
    settings, caplog
):
    events = [
        event
        async for event in LiteLLMProvider(
            settings, FailingClient(RuntimeError("secret model detail"))
        ).stream(request())
    ]

    assert len(events) == 1
    event = events[0]
    assert event.status_code == 500
    assert event.provider == "litellm"
    assert event.diagnostic.category == FailureCategory.INTERNAL
    assert event.diagnostic.stage == FailureStage.REQUEST
    assert event.diagnostic.code == "provider_invocation_failed"
    assert event.diagnostic.exception_type == "RuntimeError"
    assert re.fullmatch(
        r"claude_code_proxy\.providers\.litellm:stream:\d+",
        event.diagnostic.location,
    )
    assert "secret model detail" not in repr(events)

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_stream_failure(log_context(), event)

    rendered = caplog.records[-1].getMessage()
    assert "exception=RuntimeError" in rendered
    assert re.search(
        r"location=claude_code_proxy\.providers\.litellm:stream:\d+",
        rendered,
    )
    assert "secret model detail" not in rendered


@pytest.mark.asyncio
async def test_unexpected_error_code_is_excluded_from_logs_and_client(
    settings, caplog
):
    marker = "ACCESS_TOKEN_MUST_NOT_LEAK"
    error = RuntimeError("secret model detail")
    error.code = marker
    events = [
        event
        async for event in LiteLLMProvider(
            settings, FailingClient(error)
        ).stream(request())
    ]
    event = events[0]

    assert event.diagnostic.category == FailureCategory.INTERNAL
    assert event.diagnostic.code == "provider_invocation_failed"
    assert event.diagnostic.provider_code is None
    assert marker not in repr(event)

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_stream_failure(log_context(), event)
    assert marker not in caplog.records[-1].getMessage()

    normalized = normalize_request(
        MessagesRequest(model="model", max_tokens=10, messages=[])
    )

    async def event_source():
        yield event

    frames = [
        frame async for frame in serialize_stream(normalized, event_source())
    ]
    assert marker not in "".join(frames)
    assert "secret model detail" not in "".join(frames)


@pytest.mark.asyncio
async def test_complete_request_build_failure_is_translation_error(
    settings, monkeypatch
):
    provider = LiteLLMProvider(settings, FakeClient())

    def fail_build(*args, **kwargs):
        raise RuntimeError("secret build detail")

    monkeypatch.setattr(provider, "build_request", fail_build)

    with pytest.raises(ProviderError) as caught:
        await provider.complete(request())

    assert_failure_evidence(
        caught.value.diagnostic,
        FailureCategory.TRANSLATION,
        FailureStage.PROVIDER_TRANSLATION,
        "request_translation_failed",
        "RuntimeError",
    )
    assert "secret build detail" not in repr(caught.value)


@pytest.mark.asyncio
async def test_stream_request_build_failure_is_translation_error(
    settings, monkeypatch
):
    provider = LiteLLMProvider(settings, FakeClient())

    def fail_build(*args, **kwargs):
        raise RuntimeError("secret build detail")

    monkeypatch.setattr(provider, "build_request", fail_build)

    events = [event async for event in provider.stream(request())]

    assert len(events) == 1
    assert events[0].status_code == 500
    assert events[0].provider == "litellm"
    assert_failure_evidence(
        events[0].diagnostic,
        FailureCategory.TRANSLATION,
        FailureStage.PROVIDER_TRANSLATION,
        "request_translation_failed",
        "RuntimeError",
    )


@pytest.mark.asyncio
async def test_count_tokens_request_build_failure_is_translation_error(
    settings, monkeypatch
):
    provider = LiteLLMProvider(settings, FakeClient())

    def fail_build(*args, **kwargs):
        raise RuntimeError("secret build detail")

    monkeypatch.setattr(provider, "build_request", fail_build)

    with pytest.raises(ProviderError) as caught:
        await provider.count_tokens(request())

    assert_failure_evidence(
        caught.value.diagnostic,
        FailureCategory.TRANSLATION,
        FailureStage.PROVIDER_TRANSLATION,
        "token_count_request_translation_failed",
        "RuntimeError",
    )


@pytest.mark.asyncio
async def test_complete_response_normalization_failure_is_translation_error(settings):
    with pytest.raises(ProviderError) as caught:
        await LiteLLMProvider(settings, FakeClient(response=object())).complete(
            request()
        )

    assert str(caught.value) == "Internal server error"
    assert_failure_evidence(
        caught.value.diagnostic,
        FailureCategory.TRANSLATION,
        FailureStage.PROVIDER_TRANSLATION,
        "response_translation_failed",
        "AttributeError",
    )


@pytest.mark.asyncio
async def test_stream_response_initialization_failure_is_translation_error(settings):
    class NonIterableClient(FakeClient):
        async def acompletion(self, **kwargs):
            return object()

    events = [
        event
        async for event in LiteLLMProvider(
            settings, NonIterableClient()
        ).stream(request())
    ]

    assert len(events) == 1
    assert events[0].status_code == 500
    assert events[0].provider == "litellm"
    assert_failure_evidence(
        events[0].diagnostic,
        FailureCategory.TRANSLATION,
        FailureStage.PROVIDER_TRANSLATION,
        "stream_initialization_failed",
        "TypeError",
    )


@pytest.mark.asyncio
async def test_stream_chunk_translation_failure_is_structured(settings):
    events = [
        event
        async for event in LiteLLMProvider(
            settings, FakeClient(chunks=[object()])
        ).stream(request())
    ]

    assert events[0] == StreamStart()
    assert len(events) == 2
    assert events[1].status_code == 500
    assert events[1].provider == "litellm"
    assert_failure_evidence(
        events[1].diagnostic,
        FailureCategory.TRANSLATION,
        FailureStage.PROVIDER_TRANSLATION,
        "stream_chunk_translation_failed",
        "AttributeError",
    )


@pytest.mark.asyncio
async def test_count_tokens_failure_has_distinct_stage_and_safe_message(settings):
    with pytest.raises(ProviderError) as caught:
        await LiteLLMProvider(
            settings, FailingClient(RuntimeError("secret tokenizer detail"))
        ).count_tokens(request())

    assert str(caught.value) == "Internal server error"
    assert_failure_evidence(
        caught.value.diagnostic,
        FailureCategory.INTERNAL,
        FailureStage.REQUEST,
        "token_count_failed",
        "RuntimeError",
    )
    assert "secret tokenizer detail" not in repr(caught.value)


@pytest.mark.asyncio
async def test_count_tokens_uses_local_counter(settings):
    client = FakeClient(token_count=17)
    assert await LiteLLMProvider(settings, client).count_tokens(request()) == 17
    assert client.counter_args["model"] == "openai/gpt-5.6-sol"


@pytest.mark.asyncio
async def test_count_tokens_preserves_import_fallback(settings):
    assert await LiteLLMProvider(settings, object()).count_tokens(request()) == 1000
