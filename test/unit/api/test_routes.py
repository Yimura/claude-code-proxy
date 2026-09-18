import asyncio
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import claude_code_proxy.api.routes as routes_module
from claude_code_proxy import cli as cli_module
from claude_code_proxy.api.routes import build_router
from claude_code_proxy.config import ModelConfig, ModelDefinition
from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.control.schemas import SessionListResponse
from claude_code_proxy.domain.models import (
    CompletionResponse,
    StreamComplete,
    StreamError,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolUseStart,
)
from claude_code_proxy.logging import (
    RequestLogContext,
    SessionIdentity,
    observe_stream,
    request_logging_middleware,
)
from claude_code_proxy.observability import (
    ObservationHandle,
    SessionMetadata,
    SessionRegistry,
    SessionResult,
)
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.providers.base import ProviderError
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

    async def complete(self, request):
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


class RecordingSessionRegistry(SessionRegistry):
    def __init__(self) -> None:
        super().__init__(100, secret=b"x" * 32)
        self.finish_calls: list[tuple[ObservationHandle, SessionResult]] = []

    def finish(
        self, handle: ObservationHandle, result: SessionResult
    ) -> None:
        self.finish_calls.append((handle, result))
        super().finish(handle, result)


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
    provider=None, *, sessions=None, with_middleware=False, config=None
) -> FastAPI:
    provider = provider or Provider()
    config = config or ModelConfig({}, {}, {})
    service = ProxyService(ModelResolver(config), "litellm", provider, provider)
    app = FastAPI()
    sessions = sessions or registry()
    if with_middleware:
        app.middleware("http")(request_logging_middleware(sessions))
    app.include_router(build_router(service, sessions))
    return app


def client(provider=None, *, sessions=None, with_middleware=False, config=None):
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
    assert "output_tokens_details" not in response.json()["usage"]


def test_mapped_non_streaming_response_uses_client_capability_identity():
    sessions = registry()
    response = client(sessions=sessions, config=mapped_config()).post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "mapped-session"},
        json=messages_payload(),
    )

    assert response.status_code == 200
    assert response.json()["model"] == "claude-sonnet[1m]"
    snapshot = sessions.snapshots()[0]
    assert snapshot.client_model == "claude-sonnet"
    assert snapshot.model == "gpt-5.6-sol"
    assert snapshot.provider == "openai"
    assert snapshot.transport == "fake"
    assert snapshot.effort == "high"
    assert snapshot.context_window == 1_000_000


def test_mapped_streaming_response_uses_client_capability_identity():
    response = client(config=mapped_config()).post(
        "/v1/messages", json=messages_payload(stream=True)
    )

    start_frame = response.text.split("\n\n", 1)[0]
    start = json.loads(start_frame.split("data: ", 1)[1])
    assert start["message"]["model"] == "claude-sonnet[1m]"


def test_session_header_reaches_provider_unchanged():
    provider = Provider()

    response = client(provider).post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "session-1"},
        json=messages_payload(),
    )

    assert response.status_code == 200
    assert provider.requests[0].session_id == "session-1"


def test_missing_and_blank_session_headers_become_none():
    provider = Provider()
    api = client(provider)

    api.post("/v1/messages", json=messages_payload())
    api.post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "   "},
        json=messages_payload(),
    )

    assert [request.session_id for request in provider.requests] == [None, None]


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
    sessions = registry()
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging.session"):
        response = client(sessions=sessions).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json=messages_payload(messages=[]),
        )
    assert response.status_code == 200
    assert f"[NEW] [session {sessions.public_id('abcdef123456')[:12]}]" in caplog.text
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


def test_stream_error_logs_once_and_returns_safe_sse(caplog):
    error = StreamError(
        error_type="api_error",
        message="Internal server error",
        retryable=True,
        provider="fake",
        diagnostic="secret body",
    )
    provider = Provider(stream_events=[TextDelta("hello"), error])
    sessions = registry()
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider, sessions=sessions).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json=messages_payload(stream=True, messages=[]),
        )
    assert response.status_code == 200
    assert caplog.text.count("provider stream failed") == 1
    assert "error=api_error" in caplog.text
    assert "retryable=True" in caplog.text
    assert "secret body" not in caplog.text
    assert 'event: error' in response.text
    assert '"message": "Internal server error"' in response.text
    assert "message_stop" not in response.text
    assert "[DONE]" not in response.text
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert_finished_once(sessions, "failed")


def test_serializer_protocol_error_logs_once_and_finishes_failed(caplog):
    sessions = registry()
    provider = Provider(
        stream_events=[
            ToolUseStart("slot", "tool-1", "lookup"),
            StreamComplete("tool_use", TokenUsage(2, 1)),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider, sessions=sessions).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "invalid-stream"},
            json=messages_payload(stream=True, messages=[]),
        )

    assert response.status_code == 200
    assert 'event: error' in response.text
    assert "[DONE]" not in response.text
    assert caplog.text.count("provider stream failed") == 1
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert_finished_once(sessions, "failed")


def test_synthesized_error_survives_logging_callback_failure(
    monkeypatch,
):
    sessions = registry()
    provider = Provider(
        stream_events=[
            ToolUseStart("slot", "tool-1", "lookup"),
            StreamComplete("tool_use", TokenUsage(2, 1)),
        ]
    )

    def fail_log(_context, _error):
        raise OSError("logging sink unavailable")

    monkeypatch.setattr(routes_module, "log_stream_failure", fail_log)

    response = client(provider, sessions=sessions).post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "logging-failure"},
        json=messages_payload(stream=True, messages=[]),
    )

    assert response.status_code == 200
    assert 'event: error' in response.text
    assert "[DONE]" not in response.text
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert_finished_once(sessions, "failed")


def test_premature_stream_eof_logs_once_and_finishes_failed(caplog):
    sessions = registry()
    provider = Provider(stream_events=[TextDelta("partial")])

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider, sessions=sessions).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "premature-eof"},
            json=messages_payload(stream=True, messages=[]),
        )

    assert response.status_code == 200
    assert 'event: error' in response.text
    assert "[DONE]" not in response.text
    assert caplog.text.count("provider stream failed") == 1
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert_finished_once(sessions, "failed")


def test_successful_stream_has_no_completion_log(caplog):
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = client().post(
            "/v1/messages",
            json=messages_payload(stream=True, messages=[]),
        )
    assert response.status_code == 200
    assert "completed" not in caplog.text
    assert "200 OK" not in caplog.text


def test_stream_iterator_exception_becomes_safe_terminal_error(caplog):
    provider = Provider(stream_error=RuntimeError("secret body"))
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider, with_middleware=True).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json=messages_payload(stream=True, messages=[]),
        )

    assert response.status_code == 200
    assert caplog.text.count("provider stream failed") == 1
    assert "secret body" not in caplog.text
    assert 'event: error' in response.text
    assert '"message": "Internal server error"' in response.text


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
    sessions = registry()
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(sessions=sessions, with_middleware=True).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json={"model": "model"},
        )
    assert response.status_code == 422
    assert caplog.text.count("HTTP request failed") == 1
    assert "status=422" in caplog.text
    assert f"[session {sessions.public_id('abcdef123456')[:12]}]" in caplog.text


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


def test_non_stream_success_finishes_retained_session_as_completed():
    sessions = registry()

    response = client(sessions=sessions).post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "session-success"},
        json=messages_payload(messages=[]),
    )

    assert response.status_code == 200
    snapshot = sessions.snapshots()[0]
    assert snapshot.state == "idle"
    assert snapshot.active_requests == 0
    assert snapshot.requests == 1
    assert snapshot.last_result == "completed"
    assert_finished_once(sessions, "completed")


def test_provider_failure_finishes_session_as_failed():
    sessions = registry()
    provider = Provider(ProviderError("busy", provider="fake", status_code=503))

    response = client(provider, sessions=sessions).post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "session-failure"},
        json=messages_payload(messages=[]),
    )

    assert response.status_code == 503
    snapshot = sessions.snapshots()[0]
    assert snapshot.state == "failed"
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert_finished_once(sessions, "failed")


def test_response_translation_failure_finishes_session_as_failed(monkeypatch):
    sessions = registry()

    def fail_translation(_response):
        raise RuntimeError("translation failed")

    monkeypatch.setattr(
        "claude_code_proxy.api.routes.to_api_response", fail_translation
    )
    with pytest.raises(RuntimeError, match="translation failed"):
        client(sessions=sessions).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "translation-session"},
            json=messages_payload(messages=[]),
        )

    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert_finished_once(sessions, "failed")


@pytest.mark.parametrize(
    ("provider", "expected_status", "expected_result"),
    [
        (Provider(), 200, "completed"),
        (
            Provider(
                count_error=ProviderError(
                    "busy", provider="fake", status_code=503
                )
            ),
            503,
            "failed",
        ),
    ],
)
def test_token_count_records_lifecycle(provider, expected_status, expected_result):
    sessions = registry()

    response = client(provider, sessions=sessions).post(
        "/v1/messages/count_tokens",
        headers={"x-claude-code-session-id": "count-session"},
        json={"model": "claude-sonnet", "messages": []},
    )

    assert response.status_code == expected_status
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == expected_result
    assert_finished_once(sessions, expected_result)


def test_successful_stream_finishes_session_on_terminal_event():
    sessions = registry()

    response = client(sessions=sessions).post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "stream-session"},
        json=messages_payload(stream=True, messages=[]),
    )

    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "completed"
    assert_finished_once(sessions, "completed")


def test_missing_and_blank_headers_create_unique_request_scoped_rows():
    sessions = registry()
    api = client(sessions=sessions)

    api.post("/v1/messages", json=messages_payload(messages=[]))
    api.post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "   "},
        json=messages_payload(messages=[]),
    )

    snapshots = sessions.snapshots()
    assert len(snapshots) == 2
    assert len({snapshot.id for snapshot in snapshots}) == 2
    assert all(snapshot.requests == 1 for snapshot in snapshots)
    assert all(snapshot.last_result == "completed" for snapshot in snapshots)


def test_later_same_session_request_updates_metadata_and_request_count():
    sessions = registry()
    api = client(sessions=sessions)
    headers = {"x-claude-code-session-id": "shared-session"}

    api.post(
        "/v1/messages",
        headers=headers,
        json=messages_payload(model="openai/gpt-first", messages=[]),
    )
    api.post(
        "/v1/messages",
        headers=headers,
        json=messages_payload(model="gemini/gemini-second", messages=[]),
    )

    snapshot = sessions.snapshots()[0]
    assert snapshot.requests == 2
    assert snapshot.client_model == "gemini/gemini-second"
    assert snapshot.model == "gemini-second"
    assert snapshot.provider == "gemini"
    assert snapshot.transport == "fake"


def test_raw_session_id_is_absent_from_logs_and_safe_prefix_is_present(caplog):
    sessions = registry()
    raw_id = "raw-secret-session-value"

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging.session"):
        response = client(sessions=sessions).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": raw_id},
            json=messages_payload(messages=[]),
        )

    assert response.status_code == 200
    assert raw_id not in caplog.text
    assert sessions.public_id(raw_id)[:12] in caplog.text


_SENSITIVE_MARKERS = {
    "raw_session": "raw-session-sensitive-marker-10",
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
    return control, {
        "registry": repr(sessions.snapshots()[0]),
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


def test_sensitive_request_data_never_crosses_the_session_metadata_boundary(caplog):
    provider = Provider()
    sessions = registry()
    headers = {
        "authorization": f"Bearer {_SENSITIVE_MARKERS['credential']}",
        "x-api-key": _SENSITIVE_MARKERS["credential"],
        "x-claude-code-session-id": _SENSITIVE_MARKERS["raw_session"],
    }

    with caplog.at_level(logging.INFO, logger="claude_code_proxy"):
        response = client(
            provider,
            sessions=sessions,
            with_middleware=True,
            config=mapped_config(),
        ).post(
            "/v1/messages",
            headers=headers,
            json=_sensitive_messages_payload(),
        )

    assert response.status_code == 200
    provider_payload = repr(provider.requests[0])
    for name, marker in _SENSITIVE_MARKERS.items():
        if name != "credential":
            assert marker in provider_payload

    snapshot = sessions.snapshots()[0]
    safe_id = sessions.public_id(_SENSITIVE_MARKERS["raw_session"])
    control, exposed_surfaces = _session_exposure_surfaces(sessions, caplog.text)
    for surface, content in exposed_surfaces.items():
        for marker in _SENSITIVE_MARKERS.values():
            assert marker not in content, f"{marker!r} leaked through {surface}"

    assert len(safe_id) == 64
    assert snapshot.id == safe_id
    assert safe_id in exposed_surfaces["registry"]
    assert safe_id in exposed_surfaces["control"]
    assert safe_id[:12] in exposed_surfaces["table"]
    assert safe_id in exposed_surfaces["full_table"]
    assert _SENSITIVE_MARKERS["raw_session"] not in exposed_surfaces["full_table"]
    assert safe_id in exposed_surfaces["json"]
    for value in ("claude-sonnet", "gpt-5.6-sol", "openai", "fake", "high"):
        assert value in exposed_surfaces["control"]
    assert set(control.json()["sessions"][0]) == _SESSION_RESPONSE_FIELDS


def test_fallback_validation_log_uses_safe_id_without_registry_row(caplog):
    sessions = registry()
    raw_id = "raw-validation-session"

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(sessions=sessions, with_middleware=True).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": f"  {raw_id}  "},
            json={"model": "model"},
        )

    assert response.status_code == 422
    assert sessions.snapshots() == []
    assert raw_id not in caplog.text
    assert sessions.public_id(raw_id)[:12] in caplog.text


def test_fallback_exception_log_uses_safe_id_without_registry_row(
    caplog, monkeypatch
):
    sessions = registry()
    raw_id = "raw-exception-session"

    def fail_normalization(*_args, **_kwargs):
        raise RuntimeError("pre-observation failure")

    monkeypatch.setattr(
        "claude_code_proxy.api.routes.normalize_request", fail_normalization
    )
    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        with pytest.raises(RuntimeError, match="pre-observation failure"):
            client(sessions=sessions, with_middleware=True).post(
                "/v1/messages",
                headers={"x-claude-code-session-id": raw_id},
                json=messages_payload(messages=[]),
            )

    assert sessions.snapshots() == []
    assert raw_id not in caplog.text
    assert sessions.public_id(raw_id)[:12] in caplog.text


class ConcurrentProvider(Provider):
    def __init__(self):
        super().__init__()
        self.started = 0
        self.loops = []
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request):
        self.started += 1
        self.loops.append(asyncio.get_running_loop())
        if self.started == 2:
            self.all_started.set()
        await self.release.wait()
        return await super().complete(request)


@pytest.mark.asyncio
async def test_overlapping_requests_share_row_until_both_finish():
    sessions = registry()
    provider = ConcurrentProvider()
    app = application(provider, sessions=sessions)
    headers = {"x-claude-code-session-id": "overlap-session"}
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as api:
        requests = [
            asyncio.create_task(
                api.post(
                    "/v1/messages",
                    headers=headers,
                    json=messages_payload(messages=[]),
                )
            )
            for _ in range(2)
        ]
        await asyncio.wait_for(provider.all_started.wait(), timeout=5)
        assert provider.started == 2
        assert provider.loops[0] is provider.loops[1]
        active = sessions.snapshots()[0]
        assert active.state == "active"
        assert active.active_requests == 2
        assert active.requests == 2
        provider.release.set()
        responses = await asyncio.wait_for(
            asyncio.gather(*requests), timeout=5
        )

    assert all(response.status_code == 200 for response in responses)
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "completed"
    assert len(sessions.finish_calls) == 2
    assert all(result == "completed" for _, result in sessions.finish_calls)


def test_token_count_unexpected_exception_finishes_once_and_propagates(caplog):
    sessions = registry()
    provider = Provider(count_error=RuntimeError("secret count failure"))

    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        with pytest.raises(RuntimeError, match="secret count failure"):
            client(provider, sessions=sessions).post(
                "/v1/messages/count_tokens",
                headers={"x-claude-code-session-id": "count-exception"},
                json={"model": "claude-sonnet", "messages": []},
            )

    assert caplog.text.count("unexpected request failure") == 1
    assert "error=RuntimeError" in caplog.text
    assert "secret count failure" not in caplog.text
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert_finished_once(sessions, "failed")


def test_token_count_response_failure_finishes_once_after_provider_success(
    monkeypatch,
):
    sessions = registry()

    def fail_response(*, input_tokens):
        assert input_tokens == 7
        raise RuntimeError("response construction failed")

    monkeypatch.setattr(routes_module, "TokenCountResponse", fail_response)

    with pytest.raises(RuntimeError, match="response construction failed"):
        client(sessions=sessions).post(
            "/v1/messages/count_tokens",
            headers={"x-claude-code-session-id": "count-response"},
            json={"model": "claude-sonnet", "messages": []},
        )

    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert_finished_once(sessions, "failed")


def stream_metadata(
    session_id: str = "direct-stream",
) -> SessionMetadata:
    return SessionMetadata(
        client_session_id=session_id,
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
        serialized, sessions, observation
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


@pytest.mark.asyncio
async def test_stream_consumer_close_finishes_failed_exactly_once():
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    events = ClosingEvents()
    stream = serialized_lifecycle_stream(events, sessions, observation)

    for _ in range(3):
        await anext(stream)
    assert sessions.snapshots()[0].active_requests == 1
    await stream.aclose()

    assert events.closed is True
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert sessions.finish_calls == [(observation, "failed")]


@pytest.mark.asyncio
async def test_overlapping_streams_remain_active_until_each_closes():
    sessions = registry()
    first_observation = sessions.begin(stream_metadata("shared-stream"))
    second_observation = sessions.begin(stream_metadata("shared-stream"))
    first = serialized_lifecycle_stream(
        ClosingEvents(), sessions, first_observation
    )
    second = serialized_lifecycle_stream(
        ClosingEvents(), sessions, second_observation
    )

    for _ in range(3):
        await anext(first)
        await anext(second)
    active = sessions.snapshots()[0]
    assert active.requests == 2
    assert active.active_requests == 2

    await first.aclose()
    assert sessions.snapshots()[0].active_requests == 1

    await second.aclose()
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert sessions.finish_calls == [
        (first_observation, "failed"),
        (second_observation, "failed"),
    ]


class CancellingEvents:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise asyncio.CancelledError

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_stream_cancellation_finishes_failed_once_without_swallowing():
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    events = CancellingEvents()
    stream = serialized_lifecycle_stream(events, sessions, observation)

    await anext(stream)
    await anext(stream)
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert events.closed is True
    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "failed"
    assert sessions.finish_calls == [(observation, "failed")]


async def frame_source(*frames):
    for frame in frames:
        yield frame


@pytest.mark.asyncio
async def test_stream_completes_only_after_done_frame_yield_resumes():
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    stream = routes_module._record_stream_lifecycle(
        frame_source(routes_module.DONE_FRAME), sessions, observation
    )

    assert await anext(stream) == routes_module.DONE_FRAME
    assert sessions.snapshots()[0].active_requests == 1
    assert sessions.finish_calls == []

    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    snapshot = sessions.snapshots()[0]
    assert snapshot.active_requests == 0
    assert snapshot.last_result == "completed"
    assert sessions.finish_calls == [(observation, "completed")]
