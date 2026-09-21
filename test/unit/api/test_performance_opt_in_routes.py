import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claude_code_proxy.api.routes import build_router
from claude_code_proxy.config import ModelConfig, PerformanceMode
from claude_code_proxy.domain.models import (
    CompletionResponse,
    StreamComplete,
    TextBlock,
    TextDelta,
    TokenUsage,
)
from claude_code_proxy.logging import RequestLoggingMiddleware
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.observability import SessionRegistry
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.service import ProxyService


class Provider:
    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.telemetry: list[object | None] = []

    async def complete(self, request, telemetry=None):
        self.telemetry.append(telemetry)
        if self.fail:
            raise ProviderError("busy", provider="fake", status_code=503)
        return CompletionResponse(
            "message-id",
            request.response_model,
            (TextBlock("hello"),),
            "end_turn",
            TokenUsage(2, 1),
        )

    async def stream(self, request, telemetry=None):
        self.telemetry.append(telemetry)
        yield TextDelta("hello")
        yield StreamComplete("end_turn", TokenUsage(2, 1))

    async def count_tokens(self, request, telemetry=None):
        self.telemetry.append(telemetry)
        return 7


def client(
    provider: Provider,
    mode: PerformanceMode = PerformanceMode.OFF,
) -> tuple[TestClient, SessionRegistry]:
    sessions = SessionRegistry(
        10,
        secret=b"route-gate-secret",
        performance_enabled=mode is not PerformanceMode.OFF,
        performance_logging_enabled=mode is PerformanceMode.LOGGING,
    )
    service = ProxyService(
        ModelResolver(ModelConfig({}, {}, {})),
        "litellm",
        provider,
        provider,
    )
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, sessions=sessions)
    app.include_router(build_router(service, sessions))
    return TestClient(app), sessions


def messages_payload(*, stream: bool = False) -> dict[str, object]:
    return {
        "model": "claude-sonnet",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }


def assert_quiet_inventory(
    provider: Provider,
    sessions: SessionRegistry,
    caplog: pytest.LogCaptureFixture,
    *,
    result: str,
) -> None:
    assert provider.telemetry == [None]
    snapshot = sessions.snapshots()[0]
    assert snapshot.requests == 1
    assert snapshot.active_requests == 0
    assert snapshot.last_result == result
    assert sessions.events.current_sequence == 0
    assert "performance outcome=" not in caplog.text
    assert "request telemetry setup failed" not in caplog.text


def test_disabled_success_is_quiet_and_preserves_response(caplog) -> None:
    provider = Provider()
    api, sessions = client(provider)

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = api.post("/v1/messages", json=messages_payload())

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]
    assert_quiet_inventory(provider, sessions, caplog, result="completed")


def test_disabled_failure_is_quiet_and_preserves_base_state(caplog) -> None:
    provider = Provider(fail=True)
    api, sessions = client(provider)

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = api.post("/v1/messages", json=messages_payload())

    assert response.status_code == 503
    assert_quiet_inventory(provider, sessions, caplog, result="failed")


def test_disabled_stream_is_quiet_and_preserves_frames(caplog) -> None:
    provider = Provider()
    api, sessions = client(provider)

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = api.post(
            "/v1/messages", json=messages_payload(stream=True)
        )

    assert response.status_code == 200
    assert "hello" in response.text
    assert "[DONE]" in response.text
    assert_quiet_inventory(provider, sessions, caplog, result="completed")


def test_disabled_count_is_quiet_and_preserves_response(caplog) -> None:
    provider = Provider()
    api, sessions = client(provider)

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = api.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-sonnet", "messages": []},
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 7}
    assert_quiet_inventory(provider, sessions, caplog, result="completed")


def test_collector_records_performance_without_terminal_log(caplog) -> None:
    provider = Provider()
    api, sessions = client(provider, PerformanceMode.COLLECTOR)

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = api.post("/v1/messages", json=messages_payload())

    assert response.status_code == 200
    assert provider.telemetry[0] is not None
    assert sessions.events.current_sequence == 3
    performance = sessions.performance_snapshots().sessions[0].performance
    assert performance.recent_requests[0].outcome == "completed"
    assert "performance outcome=" not in caplog.text


def test_logging_mode_emits_one_terminal_performance_log(caplog) -> None:
    provider = Provider()
    api, sessions = client(provider, PerformanceMode.LOGGING)

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = api.post("/v1/messages", json=messages_payload())

    assert response.status_code == 200
    assert sessions.events.current_sequence == 3
    assert caplog.text.count("performance outcome=completed") == 1
