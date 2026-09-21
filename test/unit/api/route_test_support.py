import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

import claude_code_proxy.api.routes as routes_module
from claude_code_proxy import cli as cli_module
from claude_code_proxy.api.routes import build_router
from claude_code_proxy.config import ModelConfig, ModelDefinition
from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.control.schemas import SessionListResponse
from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionResponse,
    StreamComplete,
    TextBlock,
    TextDelta,
    TokenUsage,
)
from claude_code_proxy.failures import FailureDiagnostic
from claude_code_proxy.logging import (
    RequestLogContext,
    RequestLoggingMiddleware,
    SessionIdentity,
    observe_stream,
)
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.observability import (
    ObservationHandle,
    SessionMetadata,
    SessionRegistry,
    SessionResult,
)
from claude_code_proxy.performance import RequestOutcome
from claude_code_proxy.reasoning import MappingEntry
from claude_code_proxy.service import ProxyService


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
        self.requests = []

    async def complete(self, request, telemetry=None):
        self.requests.append(request)
        if self.error:
            raise self.error
        return CompletionResponse(
            "msg-1",
            request.response_model,
            (TextBlock("hello"),),
            "end_turn",
            TokenUsage(2, 1),
        )

    async def stream(self, request, telemetry=None):
        if self.stream_error:
            raise self.stream_error
        if self.stream_events is not None:
            for event in self.stream_events:
                yield event
            return
        yield TextDelta("hello")
        yield StreamComplete("end_turn", TokenUsage(2, 1))

    async def count_tokens(self, request, telemetry=None):
        if self.count_error:
            raise self.count_error
        return 7


class RecordingSessionRegistry(SessionRegistry):
    def __init__(self, *, wall_clock=None, monotonic_clock=None) -> None:
        super().__init__(
            100,
            secret=b"x" * 32,
            wall_clock=wall_clock,
            monotonic_clock=monotonic_clock,
        )
        self.finish_calls: list[tuple[ObservationHandle, RequestOutcome]] = []
        self.finish_attempts: list[tuple[ObservationHandle, RequestOutcome]] = []
        self.finish_failures: list[FailureDiagnostic | None] = []

    def finish_with_status(
        self,
        handle: ObservationHandle,
        result: RequestOutcome,
        failure: FailureDiagnostic | None = None,
    ):
        self.finish_attempts.append((handle, result))
        finalized = super().finish_with_status(handle, result, failure)
        if finalized.finalized:
            self.finish_calls.append((handle, result))
            self.finish_failures.append(failure)
        return finalized


def registry() -> RecordingSessionRegistry:
    return RecordingSessionRegistry()


def assert_finished_once(
    sessions: RecordingSessionRegistry, expected: SessionResult
) -> None:
    assert len(sessions.finish_calls) == 1
    handle, result = sessions.finish_calls[0]
    assert handle.public_id == sessions.snapshots()[0].id
    assert result == expected


def application(
    provider=None, *, sessions=None, with_middleware=True, config=None
) -> FastAPI:
    provider = provider or Provider()
    config = config or ModelConfig({}, {}, {})
    service = ProxyService(ModelResolver(config), "litellm", provider, provider)
    app = FastAPI()
    sessions = sessions or registry()
    if with_middleware:
        app.add_middleware(RequestLoggingMiddleware, sessions=sessions)
    app.include_router(build_router(service, sessions))
    return app


def client(provider=None, *, sessions=None, with_middleware=True, config=None):
    app = application(
        provider,
        sessions=sessions,
        with_middleware=with_middleware,
        config=config,
    )
    return TestClient(app)


def mapped_config():
    return ModelConfig(
        models={
            "sol": ModelDefinition(
                target="openai/gpt-5.6-sol", context_window=1_000_000
            )
        },
        tiers={"big": "sol"},
        mappings={"sonnet": MappingEntry(tier="big", effort="high")},
    )


def messages_payload(**changes):
    payload = {
        "model": "claude-sonnet",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(changes)
    return payload


_SENSITIVE_MARKERS = {
    "raw_session": "raw-session-sensitive-marker-10",
    "raw_agent": "raw-agent-sensitive-marker-10",
    "raw_parent_agent": "raw-parent-agent-sensitive-marker-10",
    "credential": "credential-sensitive-marker-10",
    "system": "system-sensitive-marker-10",
    "user": "user-sensitive-marker-10",
    "tool_name": "tool_name_sensitive_marker_10",
    "tool_description": "tool-description-sensitive-marker-10",
    "tool_schema": "tool-schema-sensitive-marker-10",
    "tool_input": "tool-input-sensitive-marker-10",
    "tool_result": "tool-result-sensitive-marker-10",
    "thinking": "thinking-sensitive-marker-10",
}


_SESSION_RESPONSE_FIELDS = {
    "id",
    "state",
    "active_requests",
    "requests",
    "client_model",
    "model",
    "provider",
    "transport",
    "effort",
    "context_window",
    "first_seen",
    "last_seen",
    "elapsed_seconds",
    "last_result",
    "agents",
}


def _sensitive_messages_payload():
    markers = _SENSITIVE_MARKERS
    return messages_payload(
        system=markers["system"],
        messages=[
            {"role": "user", "content": markers["user"]},
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": markers["thinking"]},
                    {
                        "type": "tool_use",
                        "id": "toolu_10",
                        "name": markers["tool_name"],
                        "input": {"value": markers["tool_input"]},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_10",
                        "content": markers["tool_result"],
                    }
                ],
            },
        ],
        tools=[
            {
                "name": markers["tool_name"],
                "description": markers["tool_description"],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"description": markers["tool_schema"]}
                    },
                },
            }
        ],
    )


def _session_exposure_surfaces(sessions, logs):
    control_app = create_control_app(
        sessions,
        application_version="1.0",
        pid=123,
    )
    with TestClient(control_app) as control_client:
        control = control_client.get("/v1/sessions")
    assert control.status_code == 200
    parsed = SessionListResponse.model_validate(control.json())
    async def journal_surface():
        journal = sessions.events.subscribe(after=0)
        try:
            return repr(journal.replay)
        finally:
            journal.close()

    journal_text = asyncio.run(journal_surface())
    return control, {
        "registry": repr(sessions.snapshots()[0]),
        "performance": repr(sessions.performance_snapshots()),
        "journal": journal_text,
        "control": control.text,
        "table": cli_module._render_sessions(
            parsed, cli_module.OutputFormat.TABLE, False
        ),
        "full_table": cli_module._render_sessions(
            parsed, cli_module.OutputFormat.TABLE, True
        ),
        "json": cli_module._render_sessions(
            parsed, cli_module.OutputFormat.JSON, True
        ),
        "logs": logs,
    }


def stream_metadata(
    session_id: str = "direct-stream",
) -> SessionMetadata:
    return SessionMetadata(
        client_identity=ClientIdentity(session_id=session_id),
        client_model="claude-sonnet",
        upstream_model="openai/gpt-test",
        provider="openai",
        transport="fake",
        effort="default",
        context_window=None,
    )


def stream_context() -> RequestLogContext:
    return RequestLogContext(
        session=SessionIdentity(
            "safe-public", "[session safe-public]", False
        ),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude-sonnet",
        upstream_model="openai/gpt-test",
        provider="fake",
        effort="default",
    )


class DisconnectRequest:
    def __init__(self, disconnected: bool) -> None:
        self.disconnected = disconnected
        self.state = SimpleNamespace()
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.disconnected


def serialized_lifecycle_stream(
    events,
    sessions: RecordingSessionRegistry,
    observation: ObservationHandle,
):
    prepared = routes_module.normalize_request(
        routes_module.MessagesRequest(
            model="claude-sonnet", max_tokens=10, messages=[]
        )
    )
    context = stream_context()
    observed = observe_stream(events, context)
    serialized = routes_module.serialize_stream(
        prepared,
        observed,
        heartbeat_interval=0.005,
        on_error=lambda error: routes_module.log_stream_failure(
            context, error
        ),
    )
    return routes_module._record_stream_lifecycle(
        serialized,
        DisconnectRequest(False),
        context,
        sessions,
        observation,
    )


class ClosingEvents:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return TextDelta("pending")

    async def aclose(self):
        self.closed = True


class CancellingEvents:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise asyncio.CancelledError

    async def aclose(self):
        self.closed = True


async def frame_source(*frames):
    for frame in frames:
        yield frame


def latest_performance(sessions: SessionRegistry):
    return sessions.performance_snapshots().sessions[0].performance.recent_requests[0]


class FailingFinalizationRegistry(RecordingSessionRegistry):
    def finish_with_status(self, handle, result, failure=None):
        raise RuntimeError("FINALIZATION_SECRET")


class FailingCountObserver:
    def upstream_started(self):
        return None

    def upstream_finished(self):
        return None

    def count_tokens(self, _value):
        raise RuntimeError("COUNT_CALLBACK_SECRET")


class FailingCountTelemetryRegistry(RecordingSessionRegistry):
    def observer(self, handle):
        super().observer(handle)
        return FailingCountObserver()


class LifecycleFrames:
    def __init__(self, error: BaseException, *, close_error=None):
        self.error = error
        self.close_error = close_error
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self.error

    async def aclose(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _assert_diagnostic_rendered(logs: str, diagnostic: FailureDiagnostic) -> None:
    assert f"category={diagnostic.category}" in logs
    assert f"stage={diagnostic.stage}" in logs
    assert f"code={diagnostic.code}" in logs


class CancellingCloseFailureFrames:
    def __init__(self) -> None:
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise asyncio.CancelledError

    async def aclose(self):
        self.close_calls += 1
        raise RuntimeError("CLOSE_SECRET")


class UnsupportedStreamEvent:
    pass


class UnsupportedJsonValue:
    __slots__ = ()


def _asgi_scope(path: str = "/v1/messages"):
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
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def _streaming_route_response(
    provider, sessions, *, disconnected: bool = True
):
    app = application(provider, sessions=sessions)
    included = next(
        route for route in app.routes if hasattr(route, "original_router")
    )
    endpoint = next(
        route.endpoint
        for route in included.original_router.routes
        if getattr(route, "path", None) == "/v1/messages"
    )
    scope = _asgi_scope()

    async def receive():
        if disconnected:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"", "more_body": False}

    raw_request = Request(scope, receive)
    request = routes_module.MessagesRequest(
        model="claude-sonnet",
        max_tokens=10,
        messages=[],
        stream=True,
    )
    response = await endpoint(request, raw_request)
    return response, scope, receive


class ClosingSendFailureProvider(Provider):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    async def stream(self, request, telemetry=None):
        try:
            yield TextDelta("hello")
            yield StreamComplete("end_turn", TokenUsage(2, 1))
        finally:
            self.close_calls += 1


class ProviderCloseErrorEvents:
    def __init__(self, error: ProviderError) -> None:
        self.error = error
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        self.close_calls += 1
        raise self.error


async def _call_full_stack_stream(provider, sessions, send):
    app = application(provider, sessions=sessions, with_middleware=True)
    scope = _asgi_scope()
    scope["headers"] = [
        (b"host", b"testserver"),
        (b"content-type", b"application/json"),
    ]
    body = json.dumps(messages_payload(stream=True, messages=[])).encode()
    requested = False

    async def receive():
        nonlocal requested
        if not requested:
            requested = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    await app(scope, receive, send)
