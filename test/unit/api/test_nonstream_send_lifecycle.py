import asyncio
import json
from collections.abc import Callable

import pytest
from fastapi import FastAPI
from starlette.requests import ClientDisconnect

import claude_code_proxy.api.routes as routes_module
from claude_code_proxy.api.routes import build_router
from claude_code_proxy.config import ModelConfig
from claude_code_proxy.domain.models import (
    CompletionResponse,
    TextBlock,
    TokenUsage,
)
from claude_code_proxy.failures import FailureStage
from claude_code_proxy.logging import RequestLoggingMiddleware
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.observability import SessionRegistry
from claude_code_proxy.performance import RequestOutcome
from claude_code_proxy.service import ProxyService


class Provider:
    name = "fake"

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    async def complete(self, request, telemetry=None):
        if self.error is not None:
            raise self.error
        return CompletionResponse(
            "msg-1",
            request.response_model,
            (TextBlock("hello"),),
            "end_turn",
            TokenUsage(2, 1),
        )

    async def count_tokens(self, request, telemetry=None):
        if self.error is not None:
            raise self.error
        return 7

    async def stream(self, request, telemetry=None):
        if False:
            yield


class RecordingRegistry(SessionRegistry):
    def __init__(
        self,
        events: list[str],
        *,
        performance_enabled: bool = True,
        performance_logging_enabled: bool = True,
    ) -> None:
        super().__init__(
            10,
            secret=b"x" * 32,
            performance_enabled=performance_enabled,
            performance_logging_enabled=performance_logging_enabled,
        )
        self.lifecycle_events = events
        self.finish_calls: list[tuple[RequestOutcome, object]] = []

    def finish_with_status(self, handle, result, failure=None):
        self.lifecycle_events.append(f"finish:{result}")
        self.finish_calls.append((result, failure))
        return super().finish_with_status(handle, result, failure)


def application(provider: Provider, sessions: SessionRegistry) -> FastAPI:
    service = ProxyService(
        ModelResolver(ModelConfig({}, {}, {})),
        "litellm",
        provider,
        provider,
    )
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, sessions=sessions)
    app.include_router(build_router(service, sessions))
    return app


def request_body(path: str) -> bytes:
    payload: dict[str, object] = {
        "model": "claude-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
    }
    if path == "/v1/messages":
        payload["max_tokens"] = 10
    return json.dumps(payload, separators=(",", ":")).encode()


def asgi_scope(path: str) -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"x-claude-code-session-id", b"nonstream-session"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def call_app(
    app: FastAPI,
    path: str,
    send: Callable,
    *,
    disconnected: bool = False,
) -> None:
    body = request_body(path)
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        if disconnected:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"", "more_body": False}

    await app(asgi_scope(path), receive, send)


@pytest.mark.parametrize(
    ("performance_enabled", "logging_enabled"),
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize(
    ("path", "expected_body"),
    [
        ("/v1/messages", b'{"id":"msg-1","model":"claude-sonnet","role":"assistant","content":[{"type":"text","text":"hello"}],"type":"message","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}'),
        ("/v1/messages/count_tokens", b'{"input_tokens":7}'),
    ],
)
async def test_success_finalizes_and_logs_only_after_body_send(
    monkeypatch,
    performance_enabled: bool,
    logging_enabled: bool,
    path: str,
    expected_body: bytes,
) -> None:
    events: list[str] = []
    sessions = RecordingRegistry(
        events,
        performance_enabled=performance_enabled,
        performance_logging_enabled=logging_enabled,
    )
    monkeypatch.setattr(
        routes_module,
        "log_performance",
        lambda performance, context: events.append(
            f"log:{performance.outcome}"
        ),
    )
    sent: list[dict[str, object]] = []

    async def send(message):
        sent.append(message)
        events.append(f"send:{message['type']}")

    await call_app(application(Provider(), sessions), path, send)

    assert sent[-1]["body"] == expected_body
    assert sessions.finish_calls == [("completed", None)]
    assert events[:2] == ["send:http.response.start", "send:http.response.body"]
    assert events[2] == "finish:completed"
    expected_logs = ["log:completed"] if logging_enabled else []
    assert events[3:] == expected_logs
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "completed"


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
@pytest.mark.parametrize("failed_message", ["http.response.start", "http.response.body"])
@pytest.mark.parametrize("error", [OSError("SEND_SECRET"), ClientDisconnect()])
async def test_send_disconnect_is_classified_and_preserves_original(
    path: str,
    failed_message: str,
    error: BaseException,
) -> None:
    events: list[str] = []
    sessions = RecordingRegistry(events)

    async def send(message):
        if message["type"] == failed_message:
            raise error

    with pytest.raises(type(error)) as raised:
        await call_app(application(Provider(), sessions), path, send)

    assert raised.value is error
    assert [result for result, _ in sessions.finish_calls] == [
        "client_disconnected"
    ]
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    performance = sessions.performance_snapshots().sessions[0].performance
    assert performance.current_concurrency == 0
    assert performance.recent_requests[0].outcome == "client_disconnected"


@pytest.mark.parametrize("failed_message", ["http.response.start", "http.response.body"])
async def test_unexpected_send_failure_records_client_translation(
    failed_message: str,
) -> None:
    events: list[str] = []
    sessions = RecordingRegistry(events)
    error = ValueError("SEND_SECRET")

    async def send(message):
        if message["type"] == failed_message:
            raise error

    with pytest.raises(ValueError) as raised:
        await call_app(application(Provider(), sessions), "/v1/messages", send)

    assert raised.value is error
    assert len(sessions.finish_calls) == 1
    outcome, failure = sessions.finish_calls[0]
    assert outcome == "failed"
    assert failure is not None
    assert failure.stage == FailureStage.CLIENT_TRANSLATION
    assert failure.exception_type == "ValueError"


@pytest.mark.parametrize(
    ("disconnected", "expected"),
    [(True, "client_disconnected"), (False, "cancelled")],
)
async def test_cancelled_provider_uses_shielded_disconnect_detection(
    disconnected: bool,
    expected: RequestOutcome,
) -> None:
    events: list[str] = []
    sessions = RecordingRegistry(events)

    async def send(_message):
        pytest.fail("cancelled request must not send a response")

    with pytest.raises(asyncio.CancelledError):
        await call_app(
            application(Provider(asyncio.CancelledError()), sessions),
            "/v1/messages",
            send,
            disconnected=disconnected,
        )

    assert [result for result, _ in sessions.finish_calls] == [expected]
    assert sessions.snapshots()[0].active_requests == 0
