import json
import pytest
from claude_code_proxy.domain.models import CompletionRequest, Message, StreamComplete, StreamError, StreamStart, TextBlock, TextDelta, TokenUsage
from claude_code_proxy.providers.codex.provider import CODEX_RESPONSES_URL, CodexProvider
from claude_code_proxy.reasoning import ReasoningPolicy


class Auth:
    def get_auth(self): return "secret-access", "account"


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
    Client.response = Response(status=429, text=["busy"])
    events = [event async for event in CodexProvider(Auth(), Client).stream(request())]
    assert isinstance(events[-1], StreamError)
    assert "429" in events[-1].message
    assert "secret-access" not in events[-1].message


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
