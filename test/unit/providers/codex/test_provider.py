import json
import pytest
from claude_code_proxy.domain.models import CompletionRequest, Message, StreamComplete, StreamError, StreamStart, TextBlock, TextDelta, TokenUsage
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.providers.codex.provider import CODEX_RESPONSES_URL, CodexProvider
from claude_code_proxy.reasoning import ReasoningPolicy


class Auth:
    async def get_auth(self):
        return "secret-access", "account"


class Response:
    def __init__(self, status=200, lines=(), text=()): self.status_code, self.lines, self.text = status, lines, text
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def aiter_lines(self):
        for line in self.lines: yield line
    async def aiter_text(self):
        for part in self.text: yield part


class Client:
    response = None
    request = None
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    def stream(self, method, url, **kwargs):
        Client.request = (method, url, kwargs)
        return Client.response


def request():
    return CompletionRequest("claude", "openai/gpt-5", 100, (Message("user", (TextBlock("hi"),)),), ReasoningPolicy(None, None))


@pytest.mark.asyncio
async def test_stream_posts_headers_and_returns_semantic_events():
    Client.response = Response(lines=["event: response.output_text.delta", 'data: {"delta":"hello"}', "event: response.completed", 'data: {"usage":{"input_tokens":2,"output_tokens":1},"status":"completed"}', "data: [DONE]"])
    events = [event async for event in CodexProvider(Auth(), Client).stream(request())]
    assert events == [StreamStart(), TextDelta("hello"), StreamComplete("end_turn", TokenUsage(2, 1))]
    assert Client.request[1] == CODEX_RESPONSES_URL
    assert Client.request[2]["headers"]["Authorization"] == "Bearer secret-access"


@pytest.mark.asyncio
async def test_non_200_stream_returns_safe_error_event():
    Client.response = Response(status=429, text=["quota resets in 30 seconds"])
    events = [event async for event in CodexProvider(Auth(), Client).stream(request())]

    assert events == [
        StreamStart(),
        StreamError(
            error_type="rate_limit_error",
            message="Codex API error 429: quota resets in 30 seconds",
            status_code=429,
            retryable=True,
            provider="codex",
            diagnostic="Codex API error 429: quota resets in 30 seconds",
        ),
    ]


@pytest.mark.asyncio
async def test_stream_forwards_explicit_provider_failure_message():
    Client.response = Response(lines=[
        "event: response.failed",
        'data: {"response":{"error":{"code":"server_error","message":"Model backend unavailable"}}}',
        "data: [DONE]",
    ])

    events = [event async for event in CodexProvider(Auth(), Client).stream(request())]

    assert events == [
        StreamStart(),
        StreamError(
            error_type="api_error",
            message="Model backend unavailable",
            retryable=True,
            provider="codex",
            diagnostic="server_error: Model backend unavailable",
        ),
    ]


@pytest.mark.asyncio
async def test_stream_requires_explicit_completed_event():
    Client.response = Response(lines=[
        "event: response.output_text.delta",
        'data: {"delta":"partial"}',
        "data: [DONE]",
    ])

    events = [event async for event in CodexProvider(Auth(), Client).stream(request())]

    assert events[:2] == [StreamStart(), TextDelta("partial")]
    assert isinstance(events[-1], StreamError)
    assert events[-1].message == "Internal server error"


@pytest.mark.asyncio
async def test_complete_preserves_stream_error_status():
    Client.response = Response(status=429, text=["quota exceeded"])

    with pytest.raises(ProviderError) as caught:
        await CodexProvider(Auth(), Client).complete(request())

    assert caught.value.status_code == 429
    assert str(caught.value) == "Codex API error 429: quota exceeded"


@pytest.mark.asyncio
async def test_complete_buffers_stream():
    Client.response = Response(lines=["event: response.output_text.delta", 'data: {"delta":"hello"}', "event: response.completed", 'data: {"usage":{},"status":"completed"}'])
    response = await CodexProvider(Auth(), Client).complete(request())
    assert response.content == (TextBlock("hello"),)


@pytest.mark.asyncio
async def test_count_tokens_uses_local_counter_only():
    calls = []
    async def local_counter(completion_request):
        calls.append(completion_request)
        return 8
    provider = CodexProvider(Auth(), Client, local_counter)
    assert await provider.count_tokens(request()) == 8
    assert calls[0].model == "openai/gpt-5"
