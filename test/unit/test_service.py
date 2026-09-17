import asyncio
from dataclasses import replace

import pytest
from claude_code_proxy.config import ModelConfig
from claude_code_proxy.domain.models import CompletionRequest, CompletionResponse, Message, StreamComplete, StreamError, TextBlock, TextDelta, TokenUsage
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.reasoning import MappingEntry, ReasoningPolicy
from claude_code_proxy.service import ProxyService


class FakeProvider:
    name = "fake"

    def __init__(self): self.last_request = None
    async def complete(self, request):
        self.last_request = request
        return CompletionResponse("id", request.model, (TextBlock("ok"),), "end_turn", TokenUsage(1, 1))
    async def stream(self, request):
        self.last_request = request
        yield StreamComplete("end_turn", TokenUsage(1, 1))
    async def count_tokens(self, request):
        self.last_request = request
        return 7


def make_request(model="claude-sonnet", **changes):
    request = CompletionRequest(model, model, 100, (Message("user", (TextBlock("hi"),)),), ReasoningPolicy(None, None))
    return replace(request, **changes)


@pytest.mark.asyncio
async def test_codex_transport_selects_codex_for_resolved_openai_model():
    lite, codex = FakeProvider(), FakeProvider()
    service = ProxyService(ModelResolver(ModelConfig({"big": "openai/gpt-5.6-sol"}, {"sonnet": MappingEntry(tier="big", effort="high")})), "codex", lite, codex)
    await service.complete(make_request())
    assert codex.last_request.model == "openai/gpt-5.6-sol"
    assert codex.last_request.reasoning == ReasoningPolicy(True, "high")
    assert lite.last_request is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini/gemini-2.5-pro", "anthropic/claude-opus-5"])
async def test_non_openai_models_use_litellm(model):
    lite, codex = FakeProvider(), FakeProvider()
    service = ProxyService(ModelResolver(ModelConfig({}, {})), "codex", lite, codex)
    await service.complete(make_request(model))
    assert lite.last_request.model == model
    assert codex.last_request is None


@pytest.mark.asyncio
async def test_count_tokens_uses_same_selection():
    lite, codex = FakeProvider(), FakeProvider()
    service = ProxyService(ModelResolver(ModelConfig({}, {})), "codex", lite, codex)
    assert await service.count_tokens(make_request("openai/gpt-5.6-sol")) == 7
    assert codex.last_request is not None


@pytest.mark.asyncio
async def test_stream_uses_same_selection():
    lite, codex = FakeProvider(), FakeProvider()
    service = ProxyService(ModelResolver(ModelConfig({}, {})), "litellm", lite, codex)
    assert [event async for event in service.stream(make_request())]
    assert lite.last_request is not None


class EventProvider(FakeProvider):
    def __init__(self, events=(), error=None):
        super().__init__()
        self.events = events
        self.error = error

    async def stream(self, request):
        self.last_request = request
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event


async def service_events(provider):
    service = ProxyService(
        ModelResolver(ModelConfig({}, {})), "litellm", provider, provider
    )
    return [event async for event in service.stream(make_request())]


@pytest.mark.asyncio
async def test_stream_converts_eof_without_terminal_to_protocol_error():
    events = await service_events(EventProvider([TextDelta("partial")]))

    assert events[0] == TextDelta("partial")
    assert isinstance(events[-1], StreamError)
    assert events[-1].error_type == "api_error"


@pytest.mark.asyncio
async def test_stream_converts_raised_exception_to_safe_error():
    events = await service_events(EventProvider(error=RuntimeError("secret body")))

    assert events == [
        StreamError(
            error_type="api_error",
            message="Internal server error",
            retryable=True,
            provider="fake",
            diagnostic="secret body",
        )
    ]


class TerminalThenBlocksProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.closed = False
        self.waiting = asyncio.Event()

    async def stream(self, request):
        try:
            yield StreamComplete("end_turn", TokenUsage(1, 1))
            self.waiting.set()
            await asyncio.Event().wait()
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_stream_forwards_terminal_without_waiting_for_provider_eof():
    provider = TerminalThenBlocksProvider()
    service = ProxyService(
        ModelResolver(ModelConfig({}, {})), "litellm", provider, provider
    )
    stream = service.stream(make_request())

    event = await asyncio.wait_for(anext(stream), timeout=0.1)

    assert event == StreamComplete("end_turn", TokenUsage(1, 1))
    assert provider.waiting.is_set() is False

    await stream.aclose()

    assert provider.closed is True


IDENTITY_SYSTEM = (
    TextBlock("You are Claude Code, Anthropic's official CLI for Claude."),
    TextBlock(
        "You are powered by the model named Opus 5. "
        "The exact model ID is claude-opus-5."
    ),
)
EXPECTED_MAPPED_IDENTITY = (
    TextBlock(
        "You are running inside Claude Code, Anthropic's coding-agent CLI harness."
    ),
    TextBlock(
        "The model generating this response is openai/gpt-5.6-sol, "
        "not an Anthropic Claude model."
    ),
)


def mapped_service(transport="codex"):
    lite, codex = FakeProvider(), FakeProvider()
    resolver = ModelResolver(
        ModelConfig(
            {},
            {"opus": MappingEntry(model="openai/gpt-5.6-sol", effort="high")},
        )
    )
    return ProxyService(resolver, transport, lite, codex), lite, codex


def identity_request():
    return make_request("claude-opus-5", system=IDENTITY_SYSTEM)


def test_prepare_reconciles_mapped_model_identity():
    service, _, _ = mapped_service()

    prepared = service.prepare(identity_request())

    assert prepared.model == "openai/gpt-5.6-sol"
    assert prepared.system == EXPECTED_MAPPED_IDENTITY


@pytest.mark.asyncio
async def test_complete_dispatches_reconciled_identity_to_codex():
    service, _, codex = mapped_service()

    await service.complete(identity_request())

    assert codex.last_request.system == EXPECTED_MAPPED_IDENTITY


@pytest.mark.asyncio
async def test_stream_dispatches_reconciled_identity_to_litellm():
    service, lite, _ = mapped_service("litellm")

    assert [event async for event in service.stream(identity_request())]

    assert lite.last_request.system == EXPECTED_MAPPED_IDENTITY


@pytest.mark.asyncio
async def test_count_tokens_dispatches_reconciled_identity():
    service, _, codex = mapped_service()

    assert await service.count_tokens(identity_request()) == 7

    assert codex.last_request.system == EXPECTED_MAPPED_IDENTITY
