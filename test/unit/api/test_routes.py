import asyncio
import json
import logging
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import claude_code_proxy.api.routes as routes_module
from claude_code_proxy import cli as cli_module
from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.control.schemas import SessionListResponse
from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionResponse,
    StreamComplete,
    StreamError,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolUseStart,
)
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.providers.base import ProviderError

from test.unit.api.route_test_support import (
    ClosingEvents,
    DisconnectRequest,
    Provider,
    application,
    assert_finished_once,
    client,
    frame_source,
    latest_performance,
    mapped_config,
    messages_payload,
    registry,
    serialized_lifecycle_stream,
    stream_context,
    stream_metadata,
)


def test_root_response_is_preserved():
    assert client().get("/").json() == {"message": "Anthropic Proxy for LiteLLM"}


def test_hello_probe_returns_empty_success():
    response = client().head("/api/hello")

    assert response.status_code == 200
    assert response.content == b""


def test_openapi_operation_ids_remain_compatible():
    paths = client().get("/openapi.json").json()["paths"]

    assert paths["/api/hello"]["head"]["operationId"] == "hello_api_hello_head"
    assert paths["/"]["get"]["operationId"] == "root__get"


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


def test_surrogate_model_reaches_provider_unchanged_and_control_snapshot_is_safe():
    provider = Provider()

    async def complete_with_safe_response(request, telemetry=None):
        provider.requests.append(request)
        return CompletionResponse(
            "msg-1",
            "safe-response-model",
            (TextBlock("hello"),),
            "end_turn",
            TokenUsage(2, 1),
        )

    provider.complete = complete_with_safe_response
    sessions = registry()
    public = client(provider, sessions=sessions)
    body = (
        b'{"model":"poison\\ud800model","max_tokens":10,'
        b'"messages":[{"role":"user","content":"hi"}]}'
    )

    response = public.post(
        "/v1/messages",
        content=body,
        headers={"content-type": "application/json"},
    )
    normal_response = public.post(
        "/v1/messages", json=messages_payload(model="healthy-model")
    )
    control = TestClient(
        create_control_app(sessions, started_at=datetime.now(UTC))
    )
    snapshot_response = control.get("/v1/sessions")

    assert response.status_code == normal_response.status_code == 200
    assert provider.requests[0].original_model == "poison\ud800model"
    assert provider.requests[0].model == "poison\ud800model"
    assert snapshot_response.status_code == 200
    snapshot_response.content.decode("utf-8", errors="strict")
    payload = json.loads(snapshot_response.content)
    by_client_model = {
        item["client_model"]: item for item in payload["sessions"]
    }
    assert by_client_model["poison\\ud800model"]["model"] == "poison\\ud800model"
    assert by_client_model["healthy-model"]["model"] == "healthy-model"

    validated = SessionListResponse.model_validate(payload)
    table = cli_module._render_sessions(
        validated, cli_module.OutputFormat.TABLE, no_trunc=True
    )
    structured = cli_module._render_sessions(
        validated, cli_module.OutputFormat.JSON, no_trunc=True
    )
    table.encode("utf-8", errors="strict")
    assert "poison\\ud800model" in table
    assert "healthy-model" in table
    assert len(table.splitlines()) == 3
    assert json.loads(structured) == payload["sessions"]


def test_session_header_reaches_provider_unchanged():
    provider = Provider()

    response = client(provider).post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "session-1"},
        json=messages_payload(),
    )

    assert response.status_code == 200
    assert provider.requests[0].client_identity == ClientIdentity("session-1")


def test_route_preserves_shared_session_and_distinct_agent_identity():
    provider = Provider()
    api = client(provider)

    for agent_id in ("agent-one", "agent-two"):
        response = api.post(
            "/v1/messages",
            headers={
                "x-claude-code-session-id": "shared-session",
                "x-claude-code-agent-id": agent_id,
            },
            json=messages_payload(),
        )
        assert response.status_code == 200

    assert [request.client_identity for request in provider.requests] == [
        ClientIdentity("shared-session", "agent-one", None),
        ClientIdentity("shared-session", "agent-two", None),
    ]


def test_route_preserves_nested_agent_identity():
    provider = Provider()

    response = client(provider=provider).post(
        "/v1/messages",
        headers={
            "x-claude-code-session-id": "shared-session",
            "x-claude-code-agent-id": "nested-agent",
            "x-claude-code-parent-agent-id": "parent-agent",
        },
        json=messages_payload(),
    )

    assert response.status_code == 200
    assert provider.requests[0].client_identity == ClientIdentity(
        "shared-session",
        "nested-agent",
        "parent-agent",
    )


def test_missing_and_blank_session_headers_become_none():
    provider = Provider()
    api = client(provider)

    api.post("/v1/messages", json=messages_payload())
    api.post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "   "},
        json=messages_payload(),
    )

    assert [request.client_identity for request in provider.requests] == [
        ClientIdentity(),
        ClientIdentity(),
    ]


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


def test_route_log_encodes_hostile_model_without_changing_provider_request(
    caplog, monkeypatch
):
    monkeypatch.setenv("NO_COLOR", "1")
    provider = Provider()

    async def complete_with_safe_response(request, telemetry=None):
        provider.requests.append(request)
        return CompletionResponse(
            "msg-1",
            "safe-response-model",
            (TextBlock("hello"),),
            "end_turn",
            TokenUsage(2, 1),
        )

    provider.complete = complete_with_safe_response
    hostile = "model\nforged\r\x1b\x85\u2028\u202e\ud800"
    body = json.dumps(
        messages_payload(model=hostile, messages=[]), ensure_ascii=True
    ).encode("ascii")
    with caplog.at_level(
        logging.INFO, logger="claude_code_proxy.logging.session"
    ):
        response = client(provider).post(
            "/v1/messages",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    assert provider.requests[0].original_model == hostile
    assert "model\\x0aforged\\x0d\\x1b\\x85\\u2028\\u202e\\ud800" in caplog.text
    assert "model\nforged" not in caplog.text
    for control in ("\r", "\x1b", "\x85", "\u2028", "\u202e", "\ud800"):
        assert control not in caplog.text


def test_repeated_session_logs_new_only_once(caplog):
    api = client()
    headers = {"x-claude-code-session-id": "abcdef123456"}
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging.session"):
        api.post("/v1/messages", headers=headers, json=messages_payload(messages=[]))
        api.post("/v1/messages", headers=headers, json=messages_payload(messages=[]))
    assert caplog.text.count("[NEW]") == 1


def test_provider_error_logs_once_without_success(caplog):
    provider = Provider(
        ProviderError(
            "Rate limit exceeded",
            provider="fake",
            status_code=429,
            diagnostic=FailureDiagnostic(
                FailureCategory.UPSTREAM_HTTP,
                FailureStage.RESPONSE,
                "rate_limit",
                provider_code="rate_limit_exceeded",
            ),
        )
    )
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "abcdef123456"},
            json=messages_payload(messages=[]),
        )
    assert response.status_code == 429
    assert caplog.text.count("provider request failed") == 1
    assert "category=upstream_http" in caplog.text
    assert "stage=response" in caplog.text
    assert "code=rate_limit" in caplog.text
    assert "provider_code=rate_limit_exceeded" in caplog.text
    assert "status=429" in caplog.text
    assert "Rate limit exceeded" not in caplog.text
    assert response.json() == {"detail": "Rate limit exceeded"}
    assert "rate_limit_exceeded" not in response.text
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
    assert "category=upstream_http" in caplog.text
    assert "stage=request" in caplog.text
    assert "code=provider_error" in caplog.text
    assert "/v1/messages/count_tokens" in caplog.text


def test_unexpected_provider_exception_logs_once_and_propagates(caplog):
    provider = Provider(RuntimeError("secret response body"))
    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        with pytest.raises(RuntimeError, match="secret response body"):
            client(provider, with_middleware=True).post(
                "/v1/messages",
                headers={"x-claude-code-session-id": "abcdef123456"},
                json=messages_payload(messages=[]),
            )
    assert caplog.text.count("unexpected request failure") == 1
    assert "category=internal" in caplog.text
    assert "stage=route" in caplog.text
    assert "code=unexpected_exception" in caplog.text
    assert "exception=RuntimeError" in caplog.text
    assert "location=claude_code_proxy.service:complete_prepared:" in caplog.text
    assert "secret response body" not in caplog.text
    assert "/home/" not in caplog.text


def test_stream_error_logs_once_and_returns_safe_sse(caplog):
    error = StreamError(
        error_type="api_error",
        message="Internal server error",
        retryable=True,
        provider="fake",
        diagnostic=FailureDiagnostic(
            FailureCategory.UPSTREAM_HTTP,
            FailureStage.STREAM,
            "stream_http_error",
            provider_code="SECRET_PROVIDER_CODE",
        ),
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
    assert "category=upstream_http" in caplog.text
    assert "stage=stream" in caplog.text
    assert "code=stream_http_error" in caplog.text
    assert "provider_code=" not in caplog.text and "SECRET_PROVIDER_CODE" not in caplog.text
    assert "error=api_error" in caplog.text
    assert "retryable=True" in caplog.text
    assert 'event: error' in response.text
    assert '"message": "Internal server error"' in response.text
    assert "SECRET_PROVIDER_CODE" not in response.text
    assert "upstream_http" not in response.text
    assert "stream_http_error" not in response.text
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
    assert "category=translation" in caplog.text
    assert "stage=client_translation" in caplog.text
    assert "code=invalid_event_sequence" in caplog.text
    performance = latest_performance(sessions)
    assert performance.failure is not None
    assert performance.failure.category == FailureCategory.TRANSLATION
    assert performance.failure.stage == FailureStage.CLIENT_TRANSLATION
    assert performance.failure.code == "invalid_event_sequence"
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
    assert "exception=RuntimeError" in caplog.text
    assert "location=claude_code_proxy.service:_validated_stream:" in caplog.text
    assert "secret body" not in caplog.text
    assert 'event: error' in response.text
    assert '"message": "Internal server error"' in response.text
    assert "RuntimeError" not in response.text
    assert "claude_code_proxy.service" not in response.text


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
    provider = Provider(
        ProviderError("Rate limit exceeded", provider="fake", status_code=429)
    )
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


def test_new_agent_log_uses_safe_agent_and_parent_ids(caplog):
    sessions = registry()
    raw_session = "raw-session-marker"
    raw_agent = "raw-agent-marker"
    raw_parent = "raw-parent-marker"

    with caplog.at_level(
        logging.INFO,
        logger="claude_code_proxy.logging.session",
    ):
        response = client(sessions=sessions).post(
            "/v1/messages",
            headers={
                "x-claude-code-session-id": raw_session,
                "x-claude-code-agent-id": raw_agent,
                "x-claude-code-parent-agent-id": raw_parent,
            },
            json=messages_payload(),
        )

    assert response.status_code == 200
    assert "[NEW AGENT]" in caplog.text
    assert raw_session not in caplog.text
    assert raw_agent not in caplog.text
    assert raw_parent not in caplog.text
    assert sessions.public_agent_id(raw_session, raw_agent)[:12] in caplog.text
    assert sessions.public_agent_id(raw_session, raw_parent)[:12] in caplog.text


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


def test_fallback_validation_log_uses_safe_id_without_registry_row(caplog):
    sessions = registry()
    raw_id = "raw-validation-session"
    raw_agent = "raw-validation-agent"
    raw_parent = "raw-validation-parent"

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(sessions=sessions, with_middleware=True).post(
            "/v1/messages",
            headers={
                "x-claude-code-session-id": f"  {raw_id}  ",
                "x-claude-code-agent-id": raw_agent,
                "x-claude-code-parent-agent-id": raw_parent,
            },
            json={"model": "model"},
        )

    assert response.status_code == 422
    assert sessions.snapshots() == []
    assert raw_id not in caplog.text
    assert raw_agent not in caplog.text
    assert raw_parent not in caplog.text
    assert sessions.public_id(raw_id)[:12] in caplog.text
    assert sessions.public_agent_id(raw_id, raw_agent)[:12] in caplog.text
    assert sessions.public_agent_id(raw_id, raw_parent)[:12] in caplog.text


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
    assert caplog.text.count("unexpected HTTP failure") == 1
    assert "category=internal" in caplog.text
    assert "stage=route" in caplog.text
    assert "code=unexpected_exception" in caplog.text
    assert "exception=RuntimeError" in caplog.text
    assert "location=claude_code_proxy.api.routes:create_message:" in caplog.text
    assert "pre-observation failure" not in caplog.text
    assert "/home/" not in caplog.text


class ConcurrentProvider(Provider):
    def __init__(self):
        super().__init__()
        self.started = 0
        self.loops = []
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request, telemetry=None):
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
    assert "exception=RuntimeError" in caplog.text
    assert "location=claude_code_proxy.service:count_tokens_prepared:" in caplog.text
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
        (first_observation, "client_disconnected"),
        (second_observation, "client_disconnected"),
    ]


@pytest.mark.asyncio
async def test_stream_completes_only_after_done_frame_yield_resumes():
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    stream = routes_module._record_stream_lifecycle(
        frame_source(routes_module.DONE_FRAME),
        DisconnectRequest(False),
        stream_context(),
        sessions,
        observation,
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
