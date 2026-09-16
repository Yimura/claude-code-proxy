import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claude_code_proxy.api.routes import build_router
from claude_code_proxy.config import ModelConfig
from claude_code_proxy.domain.models import (
    CompletionResponse,
    StreamComplete,
    StreamError,
    TextBlock,
    TextDelta,
    TokenUsage,
)
from claude_code_proxy.logging import SessionTracker, request_logging_middleware
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.service import ProxyService


class PlainStream:
    def isatty(self):
        return False


class Provider:
    name = "fake"

    def __init__(
        self,
        error=None,
        *,
        count_error=None,
        stream_events=None,
        stream_error=None,
    ):
        self.error = error
        self.count_error = count_error
        self.stream_events = stream_events
        self.stream_error = stream_error

    async def complete(self, request):
        if self.error:
            raise self.error
        return CompletionResponse(
            "msg-1",
            request.model,
            (TextBlock("hello"),),
            "end_turn",
            TokenUsage(2, 1),
        )

    async def stream(self, request):
        if self.stream_error:
            raise self.stream_error
        if self.stream_events is not None:
            for event in self.stream_events:
                yield event
            return
        yield TextDelta("hello")
        yield StreamComplete("end_turn", TokenUsage(2, 1))

    async def count_tokens(self, request):
        if self.count_error:
            raise self.count_error
        return 7


def client(provider=None, *, tracker=None, with_middleware=False):
    provider = provider or Provider()
    service = ProxyService(ModelResolver(ModelConfig({}, {})), "litellm", provider, provider)
    app = FastAPI()
    tracker = tracker or SessionTracker(PlainStream(), environ={})
    if with_middleware:
        app.middleware("http")(request_logging_middleware(tracker))
    app.include_router(
        build_router(
            service,
            tracker,
        )
    )
    return TestClient(app)


def messages_payload(**changes):
    payload = {
        "model": "claude-sonnet",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(changes)
    return payload


def test_root_response_is_preserved():
    assert client().get("/").json() == {"message": "Anthropic Proxy for LiteLLM"}


def test_hello_probe_returns_empty_success():
    response = client().head("/api/hello")

    assert response.status_code == 200
    assert response.content == b""


def test_non_streaming_messages_return_anthropic_json():
    response = client().post("/v1/messages", json=messages_payload())
    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]


def test_streaming_messages_return_event_stream():
    response = client().post(
        "/v1/messages", json=messages_payload(stream=True, messages=[])
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")


def test_count_tokens_returns_anthropic_shape():
    response = client().post(
        "/v1/messages/count_tokens",
        json={"model": "claude-sonnet", "messages": []},
    )
    assert response.json() == {"input_tokens": 7}


def test_provider_error_maps_to_http_status():
    provider = Provider(ProviderError("busy", provider="fake", status_code=429))
    response = client(provider).post(
        "/v1/messages", json=messages_payload(messages=[])
    )
    assert response.status_code == 429
    assert response.json() == {"detail": "busy"}


def test_invalid_request_remains_validation_error():
    assert client().post("/v1/messages", json={"model": "model"}).status_code == 422


def test_new_session_logs_resolved_request_context_without_success_summary(caplog):
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging.session"):
        response = client().post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json=messages_payload(messages=[]),
        )
    assert response.status_code == 200
    assert "[NEW] [session abcdef12]" in caplog.text
    assert "POST /v1/messages" in caplog.text
    assert "claude-sonnet → claude-sonnet" in caplog.text
    assert "provider=fake" in caplog.text
    assert "effort=default" in caplog.text
    assert "200 OK" not in caplog.text


def test_repeated_session_logs_new_only_once(caplog):
    api = client()
    headers = {"x-claude-code-session-id": "abcdef123456"}
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging.session"):
        api.post("/v1/messages", headers=headers, json=messages_payload(messages=[]))
        api.post("/v1/messages", headers=headers, json=messages_payload(messages=[]))
    assert caplog.text.count("[NEW]") == 1


def test_provider_error_logs_once_without_success(caplog):
    provider = Provider(ProviderError("busy", provider="fake", status_code=429))
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json=messages_payload(messages=[]),
        )
    assert response.status_code == 429
    assert caplog.text.count("provider request failed") == 1
    assert "status=429" in caplog.text
    assert "busy" not in caplog.text
    assert "200 OK" not in caplog.text


def test_token_count_provider_error_logs_once(caplog):
    provider = Provider(
        count_error=ProviderError("busy", provider="fake", status_code=503)
    )
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider).post(
            "/v1/messages/count_tokens",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json={"model": "claude-sonnet", "messages": []},
        )
    assert response.status_code == 503
    assert caplog.text.count("provider request failed") == 1
    assert "/v1/messages/count_tokens" in caplog.text


def test_unexpected_provider_exception_logs_once_and_propagates(caplog):
    provider = Provider(RuntimeError("secret response body"))
    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        with pytest.raises(RuntimeError, match="secret response body"):
            client(provider).post(
                "/v1/messages",
                headers={"x-claude-code-session-id": "abcdef123456"},
                json=messages_payload(messages=[]),
            )
    assert caplog.text.count("unexpected request failure") == 1
    assert "error=RuntimeError" in caplog.text
    assert "secret response body" not in caplog.text


def test_stream_error_logs_once_and_preserves_sse(caplog):
    provider = Provider(stream_events=[TextDelta("hello"), StreamError("secret body")])
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json=messages_payload(stream=True, messages=[]),
        )
    assert response.status_code == 200
    assert caplog.text.count("provider stream failed") == 1
    assert "secret body" not in caplog.text
    assert '"type": "text_delta", "text": "secret body"' in response.text


def test_successful_stream_has_no_completion_log(caplog):
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = client().post(
            "/v1/messages",
            json=messages_payload(stream=True, messages=[]),
        )
    assert response.status_code == 200
    assert "completed" not in caplog.text
    assert "200 OK" not in caplog.text


def test_stream_iterator_exception_logs_once_and_propagates(caplog):
    provider = Provider(stream_error=RuntimeError("secret body"))
    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        with pytest.raises(RuntimeError, match="secret body"):
            client(provider, with_middleware=True).post(
                "/v1/messages",
                headers={"x-claude-code-session-id": "abcdef123456"},
                json=messages_payload(stream=True, messages=[]),
            )
    assert caplog.text.count("unexpected request failure") == 1
    assert "secret body" not in caplog.text


def test_hello_probe_does_not_log_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(with_middleware=True).head("/api/hello")

    assert response.status_code == 200
    assert "HTTP request failed" not in caplog.text


def test_unknown_route_logs_one_http_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(with_middleware=True).get("/unknown")

    assert response.status_code == 404
    assert caplog.text.count("HTTP request failed") == 1
    assert "status=404" in caplog.text


def test_validation_failure_logs_one_http_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(with_middleware=True).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json={"model": "model"},
        )
    assert response.status_code == 422
    assert caplog.text.count("HTTP request failed") == 1
    assert "status=422" in caplog.text
    assert "[session abcdef12]" in caplog.text


def test_middleware_does_not_duplicate_route_provider_warning(caplog):
    provider = Provider(ProviderError("busy", provider="fake", status_code=429))
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider, with_middleware=True).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json=messages_payload(messages=[]),
        )
    assert response.status_code == 429
    assert caplog.text.count("provider request failed") == 1
    assert "HTTP request failed" not in caplog.text
