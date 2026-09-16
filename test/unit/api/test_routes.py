from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from claude_code_proxy.api.routes import build_router
from claude_code_proxy.config import Settings
from claude_code_proxy.domain.models import CompletionResponse, StreamComplete, TextBlock, TextDelta, TokenUsage
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.service import ProxyService
from fastapi import FastAPI


class Provider:
    def __init__(self, error=None): self.error = error
    async def complete(self, request):
        if self.error: raise self.error
        return CompletionResponse("msg-1", request.model, (TextBlock("hello"),), "end_turn", TokenUsage(2, 1))
    async def stream(self, request):
        yield TextDelta("hello")
        yield StreamComplete("end_turn", TokenUsage(2, 1))
    async def count_tokens(self, request): return 7


def client(provider=None):
    provider = provider or Provider()
    service = ProxyService(ModelResolver({}, "openai", "big", "small"), "openai", provider, provider)
    app = FastAPI()
    app.include_router(build_router(service))
    return TestClient(app)


def test_root_response_is_preserved():
    assert client().get("/").json() == {"message": "Anthropic Proxy for LiteLLM"}


def test_non_streaming_messages_return_anthropic_json():
    response = client().post("/v1/messages", json={"model": "model", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]


def test_streaming_messages_return_event_stream():
    response = client().post("/v1/messages", json={"model": "model", "max_tokens": 10, "stream": True, "messages": []})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")


def test_count_tokens_returns_anthropic_shape():
    response = client().post("/v1/messages/count_tokens", json={"model": "model", "messages": []})
    assert response.json() == {"input_tokens": 7}


def test_provider_error_maps_to_http_status():
    response = client(Provider(ProviderError("busy", provider="fake", status_code=429))).post("/v1/messages", json={"model": "model", "max_tokens": 10, "messages": []})
    assert response.status_code == 429
    assert response.json() == {"detail": "busy"}


def test_invalid_request_remains_validation_error():
    assert client().post("/v1/messages", json={"model": "model"}).status_code == 422
