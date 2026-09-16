from dataclasses import replace
import pytest
from claude_code_proxy.config import ModelConfig
from claude_code_proxy.domain.models import CompletionRequest, CompletionResponse, Message, StreamComplete, TextBlock, TokenUsage
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.reasoning import MappingEntry, ReasoningPolicy
from claude_code_proxy.service import ProxyService


class FakeProvider:
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
