"""Tests for the service-level WebSearch intercept loop."""

import pytest

from claude_code_proxy.domain.models import (
    CompletionRequest,
    CompletionResponse,
    Message,
    StreamComplete,
    StreamError,
    TextBlock,
    TokenUsage,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseEnd,
    ToolUseStart,
)
from claude_code_proxy.reasoning import ReasoningPolicy
from claude_code_proxy.service import ProxyService


class ScriptedProvider:
    """Yields scripted responses in order, recording every request."""

    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request, telemetry=None):
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request, telemetry=None):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        for event in response:
            yield event

    async def count_tokens(self, request, telemetry=None):
        self.requests.append(request)
        return 7


class FakeSearcher:
    def __init__(self, text="search hit https://example.com", pages=None):
        self.queries = []
        self.fetched = []
        self.text = text
        self.pages = pages or {}

    async def search(self, query):
        self.queries.append(query)
        return self.text

    async def fetch_page(self, url):
        self.fetched.append(url)
        return self.pages.get(url, "page content")


def request_with_search(backend="vllm", **changes):
    base = CompletionRequest(
        original_model="openai/qwen",
        model="openai/qwen",
        response_model="openai/qwen",
        max_tokens=2000,
        messages=(Message("user", (TextBlock("price?"),)),),
        reasoning=ReasoningPolicy(True, "medium"),
        backend=backend,
        tools=(ToolDefinition("WebSearch", "search the web", {"type": "object"}),),
    )
    from dataclasses import replace

    return replace(base, **changes)


def provider_and_service(responses, searcher=None):
    provider = ScriptedProvider(responses)
    service = ProxyService.__new__(ProxyService)
    service._resolver = None
    service._openai_transport = "litellm"
    service._litellm_provider = provider
    service._codex_provider = provider
    service._web_search = searcher or FakeSearcher()
    return provider, service


@pytest.mark.asyncio
async def test_complete_intercepts_web_search_and_refeeds_upstream():
    first = CompletionResponse(
        "id-1",
        "openai/qwen",
        (ToolUseBlock("call_1", "WebSearch", {"query": "galaxy a57 price"}),),
        "tool_use",
        TokenUsage(10, 5),
    )
    second = CompletionResponse(
        "id-2", "openai/qwen", (TextBlock("It costs 319 euros."),), "end_turn", TokenUsage(20, 5)
    )
    provider, service = provider_and_service([first, second])
    searcher = service._web_search

    response = await service.complete_prepared(request_with_search())

    assert response is second
    assert searcher.queries == ["galaxy a57 price"]
    followup = provider.requests[1]
    assert followup.messages[-2].role == "assistant"
    assert followup.messages[-2].content[0] == ToolUseBlock(
        "call_1", "WebSearch", {"query": "galaxy a57 price"}
    )
    result = followup.messages[-1].content[-1]
    assert isinstance(result, ToolResultBlock)
    assert result.tool_use_id == "call_1"
    assert "search hit" in result.content


@pytest.mark.asyncio
async def test_complete_fetches_top_result_pages_when_search_lists_urls():
    first = CompletionResponse(
        "id-1",
        "openai/qwen",
        (ToolUseBlock("call_1", "WebSearch", {"query": "q"}),),
        "tool_use",
        TokenUsage(1, 1),
    )
    second = CompletionResponse(
        "id-2", "openai/qwen", (TextBlock("done"),), "end_turn", TokenUsage(1, 1)
    )
    provider, service = provider_and_service([first, second])
    searcher = FakeSearcher(
        text="- Hit one\n  https://example.com/1\n  snip one\n- Hit two\n  https://example.com/2\n  snip two\n- Hit three\n  https://example.com/3\n  snip three",
        pages={
            "https://example.com/1": "first page body",
            "https://example.com/2": "second page body",
        },
    )
    service._web_search = searcher

    await service.complete_prepared(request_with_search())

    assert searcher.fetched == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    result = provider.requests[1].messages[-1].content[-1]
    assert "first page body" in result.content
    assert "second page body" in result.content
    assert "https://example.com/3" in result.content
    assert "snip three" in result.content


@pytest.mark.asyncio
async def test_complete_passes_through_without_web_search_tool():
    direct = CompletionResponse(
        "id-1", "openai/qwen", (TextBlock("hello"),), "end_turn", TokenUsage(1, 1)
    )
    provider, service = provider_and_service([direct])
    service._web_search = None

    response = await service.complete_prepared(request_with_search(tools=()))

    assert response is direct
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_complete_passes_through_on_zai_backend():
    direct = CompletionResponse(
        "id-1", "openai/glm", (TextBlock("hello"),), "end_turn", TokenUsage(1, 1)
    )
    provider, service = provider_and_service([direct])

    response = await service.complete_prepared(request_with_search(backend="zai"))

    assert response is direct
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_complete_passes_through_when_mixed_tools_called():
    first = CompletionResponse(
        "id-1",
        "openai/qwen",
        (
            ToolUseBlock("call_1", "WebSearch", {"query": "q"}),
            ToolUseBlock("call_2", "Bash", {"command": "ls"}),
        ),
        "tool_use",
        TokenUsage(1, 1),
    )
    provider, service = provider_and_service([first])

    response = await service.complete_prepared(request_with_search())

    assert response is first
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_complete_stops_after_max_rounds():
    loops = [
        CompletionResponse(
            f"id-{i}",
            "openai/qwen",
            (ToolUseBlock(f"call_{i}", "WebSearch", {"query": f"q{i}"}),),
            "tool_use",
            TokenUsage(1, 1),
        )
        for i in range(10)
    ]
    provider, service = provider_and_service(loops)

    await service.complete_prepared(request_with_search())

    assert len(provider.requests) == 5


@pytest.mark.asyncio
async def test_stream_intercepts_web_search():
    first = [
        ToolUseStart("0", "call_1", "WebSearch"),
        ToolUseEnd("0"),
        StreamComplete("tool_use", TokenUsage(10, 5)),
    ]
    second = [
        StreamComplete("end_turn", TokenUsage(20, 5)),
    ]
    provider, service = provider_and_service([first, second])

    events = [e async for e in service.stream_prepared(request_with_search())]

    assert len(provider.requests) == 2
    assert isinstance(events[-1], StreamComplete)
    assert events[-1].stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_stream_passes_through_without_search_tool():
    events_in = [StreamComplete("end_turn", TokenUsage(1, 1))]
    provider, service = provider_and_service([events_in])
    service._web_search = None

    events = [e async for e in service.stream_prepared(request_with_search(tools=()))]

    assert events == events_in
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_stream_buffers_intermediate_round_and_yields_final_events_only():
    first = [
        ToolUseStart("0", "call_1", "WebSearch"),
        ToolUseEnd("0"),
        StreamComplete("tool_use", TokenUsage(10, 5)),
    ]
    second = [
        TextBlock_final_marker := TextBlock("answer"),
        StreamComplete("end_turn", TokenUsage(20, 5)),
    ]
    provider, service = provider_and_service([first, second])

    events = [e async for e in service.stream_prepared(request_with_search())]

    assert TextBlock_final_marker not in events or events[-2] == TextBlock_final_marker
    assert not any(isinstance(e, ToolUseStart) and e.id == "call_1" for e in events)


@pytest.mark.asyncio
async def test_search_failure_feeds_error_text_upstream():
    class FailingSearcher:
        async def search(self, query):
            raise RuntimeError("no network")

        async def fetch_page(self, url):
            return ""

    first = CompletionResponse(
        "id-1",
        "openai/qwen",
        (ToolUseBlock("call_1", "WebSearch", {"query": "q"}),),
        "tool_use",
        TokenUsage(1, 1),
    )
    second = CompletionResponse(
        "id-2", "openai/qwen", (TextBlock("cannot verify"),), "end_turn", TokenUsage(1, 1)
    )
    provider, service = provider_and_service([first, second])
    service._web_search = FailingSearcher()

    response = await service.complete_prepared(request_with_search())

    assert response is second
    result_block = provider.requests[1].messages[-1].content[-1]
    assert isinstance(result_block, ToolResultBlock)
    assert "no network" in result_block.content
