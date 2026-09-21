import asyncio
import gc
import logging
from datetime import UTC, datetime

import pytest
from starlette.requests import ClientDisconnect

import claude_code_proxy.api.routes as routes_module
import claude_code_proxy.logging as logging_module
from claude_code_proxy.domain.models import (
    CompletionResponse,
    StreamError,
    TokenUsage,
    ToolUseBlock,
)
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.logging import observe_stream
from claude_code_proxy.providers.base import ProviderError
from test.unit.api.route_test_support import (
    CancellingCloseFailureFrames,
    CancellingEvents,
    ClosingEvents,
    ClosingSendFailureProvider,
    DisconnectRequest,
    FailingCountTelemetryRegistry,
    FailingFinalizationRegistry,
    LifecycleFrames,
    Provider,
    ProviderCloseErrorEvents,
    RecordingSessionRegistry,
    UnsupportedJsonValue,
    UnsupportedStreamEvent,
    _SENSITIVE_MARKERS,
    _SESSION_RESPONSE_FIELDS,
    _assert_diagnostic_rendered,
    _call_full_stack_stream,
    _sensitive_messages_payload,
    _session_exposure_surfaces,
    _streaming_route_response,
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


def test_successful_stream_has_performance_completion_log(caplog):
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = client().post(
            "/v1/messages",
            json=messages_payload(stream=True, messages=[]),
        )
    assert response.status_code == 200
    assert "200 OK" not in caplog.text
    performance = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("performance ")
    ]
    assert len(performance) == 1
    assert "performance outcome=completed" in performance[0]


def test_sensitive_request_data_never_crosses_the_session_metadata_boundary(caplog):
    provider = Provider()
    sessions = registry()
    headers = {
        "authorization": f"Bearer {_SENSITIVE_MARKERS['credential']}",
        "x-api-key": _SENSITIVE_MARKERS["credential"],
        "x-claude-code-session-id": _SENSITIVE_MARKERS["raw_session"],
        "x-claude-code-agent-id": _SENSITIVE_MARKERS["raw_agent"],
        "x-claude-code-parent-agent-id": _SENSITIVE_MARKERS["raw_parent_agent"],
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
        if name not in {"credential", "raw_session", "raw_agent", "raw_parent_agent"}:
            assert marker in provider_payload
    assert _SENSITIVE_MARKERS["raw_session"] not in provider_payload
    assert _SENSITIVE_MARKERS["raw_agent"] not in provider_payload
    assert _SENSITIVE_MARKERS["raw_parent_agent"] not in provider_payload

    snapshot = sessions.snapshots()[0]
    safe_id = sessions.public_id(_SENSITIVE_MARKERS["raw_session"])
    safe_agent_id = sessions.public_agent_id(
        _SENSITIVE_MARKERS["raw_session"],
        _SENSITIVE_MARKERS["raw_agent"],
    )
    safe_parent_id = sessions.public_agent_id(
        _SENSITIVE_MARKERS["raw_session"],
        _SENSITIVE_MARKERS["raw_parent_agent"],
    )
    control, exposed_surfaces = _session_exposure_surfaces(sessions, caplog.text)
    for surface, content in exposed_surfaces.items():
        for marker in _SENSITIVE_MARKERS.values():
            assert marker not in content, f"{marker!r} leaked through {surface}"

    assert len(safe_id) == 64
    assert snapshot.id == safe_id
    assert safe_id in exposed_surfaces["registry"]
    assert safe_id in exposed_surfaces["performance"]
    assert safe_id in exposed_surfaces["journal"]
    assert "input_tokens" in exposed_surfaces["performance"]
    assert "input_tokens" in exposed_surfaces["journal"]
    assert safe_id in exposed_surfaces["control"]
    assert safe_id[:12] in exposed_surfaces["table"]
    assert safe_id in exposed_surfaces["full_table"]
    assert _SENSITIVE_MARKERS["raw_session"] not in exposed_surfaces["full_table"]
    assert safe_id in exposed_surfaces["json"]
    assert safe_agent_id in exposed_surfaces["json"]
    assert safe_parent_id in exposed_surfaces["json"]
    for value in ("claude-sonnet", "gpt-5.6-sol", "openai", "fake", "high"):
        assert value in exposed_surfaces["control"]
    assert set(control.json()["sessions"][0]) == _SESSION_RESPONSE_FIELDS


@pytest.mark.asyncio
async def test_stream_consumer_close_finishes_disconnected_exactly_once():
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
    assert sessions.finish_calls == [(observation, "client_disconnected")]
    performance = sessions.performance_snapshots().sessions[0].performance
    assert performance.recent_requests[0].outcome == "client_disconnected"


@pytest.mark.asyncio
async def test_stream_cancellation_finishes_cancelled_once_without_swallowing():
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
    assert sessions.finish_calls == [(observation, "cancelled")]
    performance = sessions.performance_snapshots().sessions[0].performance
    assert performance.recent_requests[0].outcome == "cancelled"


def test_nonstream_success_records_messages_performance_once(caplog):
    sessions = registry()

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = client(sessions=sessions).post(
            "/v1/messages",
            headers={"x-claude-code-session-id": "message-performance"},
            json=messages_payload(messages=[]),
        )

    assert response.status_code == 200
    snapshot = latest_performance(sessions)
    assert snapshot.operation == "messages"
    assert snapshot.outcome == "completed"
    assert snapshot.input_tokens.value == 2
    assert snapshot.output_tokens.value == 1
    assert caplog.text.count("performance ") == 1
    assert "operation=messages" in caplog.text
    assert "outcome=completed" in caplog.text


def test_count_success_records_distinct_metrics_and_preserves_response(caplog):
    sessions = registry()

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        response = client(sessions=sessions).post(
            "/v1/messages/count_tokens",
            headers={"x-claude-code-session-id": "count-performance"},
            json={"model": "claude-sonnet", "messages": []},
        )

    assert response.status_code == 200
    assert response.content == b'{"input_tokens":7}'
    snapshot = latest_performance(sessions)
    assert snapshot.operation == "count_tokens"
    assert snapshot.outcome == "completed"
    assert snapshot.input_tokens.value == 7
    assert snapshot.ttft.status == "not_applicable"
    assert snapshot.output_tokens.status == "not_applicable"
    assert snapshot.tool_calls.status == "not_applicable"
    assert caplog.text.count("performance ") == 1
    assert "operation=count_tokens" in caplog.text
    assert "input_tokens=7" in caplog.text


def test_route_uses_registry_clock_domain_before_normalization(monkeypatch):
    calls = []
    wall = datetime(2000, 1, 1, tzinfo=UTC)
    original_normalize = routes_module.normalize_request

    def wall_clock():
        return wall

    def monotonic_clock():
        return 10.0

    def tracked_normalize(*args, **kwargs):
        calls.append("normalize")
        return original_normalize(*args, **kwargs)

    sessions = RecordingSessionRegistry(
        wall_clock=wall_clock,
        monotonic_clock=monotonic_clock,
    )
    original_sample = sessions.sample_clocks

    def tracked_sample():
        calls.append("sample")
        return original_sample()

    monkeypatch.setattr(sessions, "sample_clocks", tracked_sample)
    monkeypatch.setattr(routes_module, "normalize_request", tracked_normalize)

    response = client(sessions=sessions).post(
        "/v1/messages", json=messages_payload(messages=[])
    )

    assert response.status_code == 200
    assert calls[:2] == ["sample", "normalize"]
    performance = latest_performance(sessions)
    assert performance.started_at == wall
    assert performance.finished_at == wall
    assert performance.duration.value == 0


def test_provider_failure_preserves_status_and_records_safe_diagnostic_once(caplog):
    diagnostic = FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.RESPONSE,
        "overloaded",
        provider_code="overloaded",
    )
    provider = Provider(
        ProviderError(
            "SECRET_PROVIDER_BODY",
            provider="fake",
            status_code=503,
            diagnostic=diagnostic,
        )
    )
    sessions = registry()

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(provider, sessions=sessions, with_middleware=True).post(
            "/v1/messages", json=messages_payload(messages=[])
        )

    assert response.status_code == 503
    assert latest_performance(sessions).failure == diagnostic
    assert sessions.finish_failures == [diagnostic]
    assert caplog.text.count("provider request failed") == 1
    assert caplog.text.count("performance ") == 1
    assert "provider_code=overloaded" in caplog.text and "SECRET_PROVIDER_BODY" not in caplog.text


def test_finalization_failure_does_not_change_successful_response(caplog):
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(sessions=FailingFinalizationRegistry()).post(
            "/v1/messages", json=messages_payload(messages=[])
        )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]
    assert caplog.text.count("request finalization failed") == 1
    assert "FINALIZATION_SECRET" not in caplog.text


def test_count_callback_failure_does_not_change_response(caplog):
    sessions = FailingCountTelemetryRegistry()

    with caplog.at_level(logging.WARNING):
        response = client(sessions=sessions).post(
            "/v1/messages/count_tokens",
            json={"model": "claude-sonnet", "messages": []},
        )

    assert response.content == b'{"input_tokens":7}'
    assert latest_performance(sessions).input_tokens.status == "unavailable"
    assert "COUNT_CALLBACK_SECRET" not in caplog.text


@pytest.mark.parametrize(
    ("disconnected", "expected"),
    [(False, "cancelled"), (True, "client_disconnected")],
)
@pytest.mark.asyncio
async def test_stream_cancellation_classifies_disconnect_state(disconnected, expected):
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    frames = LifecycleFrames(asyncio.CancelledError())
    request = DisconnectRequest(disconnected)
    stream = routes_module._record_stream_lifecycle(
        frames, request, stream_context(), sessions, observation
    )

    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert request.checks == 1
    assert frames.close_calls == 1
    assert sessions.finish_calls == [(observation, expected)]
    assert latest_performance(sessions).outcome == expected


@pytest.mark.asyncio
async def test_stream_client_disconnect_is_recorded_and_reraised():
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    frames = LifecycleFrames(ClientDisconnect())
    stream = routes_module._record_stream_lifecycle(
        frames, DisconnectRequest(False), stream_context(), sessions, observation
    )

    with pytest.raises(ClientDisconnect):
        await anext(stream)

    assert frames.close_calls == 1
    assert sessions.finish_calls == [(observation, "client_disconnected")]


@pytest.mark.asyncio
async def test_original_cancellation_wins_iterator_close_failure():
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    frames = LifecycleFrames(
        asyncio.CancelledError(), close_error=RuntimeError("close failed")
    )
    stream = routes_module._record_stream_lifecycle(
        frames, DisconnectRequest(False), stream_context(), sessions, observation
    )

    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert frames.close_calls == 1
    assert sessions.finish_calls == [(observation, "cancelled")]


def test_duplicate_finalization_emits_one_terminal_record(caplog):
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    context = stream_context()

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        routes_module._finalize_request(
            sessions, observation, context, "completed"
        )
        routes_module._finalize_request(
            sessions, observation, context, "completed"
        )

    assert sessions.finish_attempts == [
        (observation, "completed"),
        (observation, "completed"),
    ]
    assert sessions.finish_calls == [(observation, "completed")]
    assert caplog.text.count("performance ") == 1


@pytest.mark.asyncio
async def test_stream_error_then_consumer_close_remains_failed():
    diagnostic = FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.STREAM,
        "terminal_stream_error",
    )
    error = StreamError(
        error_type="api_error",
        message="Internal server error",
        provider="fake",
        diagnostic=diagnostic,
    )
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    sessions.observer(observation).stream_event(error)
    request = DisconnectRequest(False)
    context = stream_context()
    routes_module._record_stream_error(request, context, error)
    stream = routes_module._record_stream_lifecycle(
        frame_source("event: error\ndata: {}\n\n"),
        request,
        context,
        sessions,
        observation,
    )

    assert await anext(stream) == "event: error\ndata: {}\n\n"
    await stream.aclose()

    assert sessions.snapshots()[0].active_requests == 0
    assert sessions.finish_calls == [(observation, "failed")]
    performance = latest_performance(sessions)
    assert performance.outcome == "failed"
    assert performance.failure == diagnostic


def test_provider_stream_error_survives_logging_sink_failure(
    monkeypatch,
):
    diagnostic = FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.STREAM,
        "provider_stream_error",
    )
    provider = Provider(
        stream_events=[
            StreamError(
                error_type="api_error",
                message="Internal server error",
                provider="fake",
                diagnostic=diagnostic,
            )
        ]
    )
    sessions = registry()

    def fail_log(*_args, **_kwargs):
        raise OSError("LOG_SINK_SECRET")

    monkeypatch.setattr(
        "claude_code_proxy.logging.log_stream_failure", fail_log
    )

    response = client(provider, sessions=sessions).post(
        "/v1/messages",
        json=messages_payload(stream=True, messages=[]),
    )

    assert response.status_code == 200
    assert 'event: error' in response.text
    assert '"message": "Internal server error"' in response.text
    assert "LOG_SINK_SECRET" not in response.text
    performance = latest_performance(sessions)
    assert performance.outcome == "failed"
    assert performance.failure == diagnostic


def test_diagnosticless_complete_failure_retains_logged_fallback(caplog):
    error = ProviderError("busy", provider="fake", status_code=503)
    sessions = registry()

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(Provider(error), sessions=sessions).post(
            "/v1/messages",
            json=messages_payload(messages=[]),
        )

    expected = logging_module.provider_failure_diagnostic(error)
    assert response.status_code == 503
    assert latest_performance(sessions).failure == expected
    _assert_diagnostic_rendered(caplog.text, expected)
    assert caplog.text.count("provider request failed") == 1
    assert caplog.text.count("performance outcome=failed") == 1


def test_diagnosticless_count_failure_retains_logged_fallback(caplog):
    error = ProviderError("busy", provider="fake", status_code=503)
    sessions = registry()

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(
            Provider(count_error=error), sessions=sessions
        ).post(
            "/v1/messages/count_tokens",
            json={"model": "claude-sonnet", "messages": []},
        )

    expected = logging_module.provider_failure_diagnostic(error)
    assert response.status_code == 503
    assert latest_performance(sessions).failure == expected
    _assert_diagnostic_rendered(caplog.text, expected)
    assert caplog.text.count("provider request failed") == 1
    assert caplog.text.count("performance outcome=failed") == 1


def test_diagnosticless_semantic_stream_retains_logged_fallback(caplog):
    error = StreamError(error_type="api_error", provider="fake")
    sessions = registry()

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        response = client(
            Provider(stream_events=[error]), sessions=sessions
        ).post(
            "/v1/messages",
            json=messages_payload(stream=True, messages=[]),
        )

    expected = logging_module.stream_failure_diagnostic(error)
    assert response.status_code == 200
    assert 'event: error' in response.text
    assert latest_performance(sessions).failure == expected
    _assert_diagnostic_rendered(caplog.text, expected)
    assert caplog.text.count("provider stream failed") == 1
    assert caplog.text.count("performance outcome=failed") == 1


@pytest.mark.asyncio
async def test_full_stream_chain_preserves_cancellation_over_close_failure():
    events = CancellingCloseFailureFrames()
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    request = DisconnectRequest(False)
    context = stream_context()
    prepared = routes_module.normalize_request(
        routes_module.MessagesRequest(
            model="claude-sonnet", max_tokens=10, messages=[]
        )
    )
    observed = observe_stream(events, context)
    serialized = routes_module.serialize_stream(prepared, observed)
    stream = routes_module._record_stream_lifecycle(
        serialized, request, context, sessions, observation
    )

    await anext(stream)
    await anext(stream)
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert events.close_calls == 1
    assert latest_performance(sessions).outcome == "cancelled"


def test_unsupported_stream_event_records_client_translation_failure_once(caplog):
    sessions = registry()
    provider = Provider(stream_events=[UnsupportedStreamEvent()])

    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        with pytest.raises(TypeError, match="Unsupported stream event") as raised:
            client(provider, sessions=sessions, with_middleware=True).post(
                "/v1/messages",
                json=messages_payload(stream=True, messages=[]),
            )

    assert type(raised.value) is TypeError
    assert caplog.text.count("unexpected request failure") == 1
    performance = latest_performance(sessions)
    assert performance.outcome == "failed"
    assert performance.failure is not None
    assert performance.failure.category == FailureCategory.INTERNAL
    assert performance.failure.stage == FailureStage.CLIENT_TRANSLATION
    assert performance.failure.code == "unexpected_exception"
    assert performance.failure.exception_type == "TypeError"
    assert "Unsupported stream event" not in caplog.text


@pytest.mark.asyncio
async def test_observed_stream_exception_is_not_logged_twice_by_lifecycle(caplog):
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    request = DisconnectRequest(False)
    context = stream_context()
    prepared = routes_module.normalize_request(
        routes_module.MessagesRequest(
            model="claude-sonnet", max_tokens=10, messages=[]
        )
    )

    async def failing_events():
        raise RuntimeError("OBSERVED_CHAIN_SECRET")
        yield

    observed = observe_stream(
        failing_events(),
        context,
        on_exception=lambda diagnostic: (
            routes_module._mark_observed_stream_exception(
                request, diagnostic
            )
        ),
    )
    serialized = routes_module.serialize_stream(prepared, observed)
    stream = routes_module._record_stream_lifecycle(
        serialized, request, context, sessions, observation
    )

    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        await anext(stream)
        await anext(stream)
        with pytest.raises(RuntimeError, match="OBSERVED_CHAIN_SECRET"):
            await anext(stream)

    assert caplog.text.count("unexpected request failure") == 1
    performance = latest_performance(sessions)
    assert performance.failure is not None
    assert performance.failure.stage == FailureStage.STREAM
    assert performance.failure.exception_type == "RuntimeError"
    assert "OBSERVED_CHAIN_SECRET" not in caplog.text


def test_nonstream_json_render_failure_finalizes_failed_not_completed(caplog):
    provider = Provider()

    async def complete_with_unserializable_tool(request, telemetry=None):
        return CompletionResponse(
            "msg-render-failure",
            request.response_model,
            (
                ToolUseBlock(
                    "tool-1",
                    "lookup",
                    {"unsupported": UnsupportedJsonValue()},
                ),
            ),
            "tool_use",
            TokenUsage(2, 1),
        )

    provider.complete = complete_with_unserializable_tool
    sessions = registry()

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        with pytest.raises(Exception) as raised:
            client(provider, sessions=sessions, with_middleware=True).post(
                "/v1/messages",
                json=messages_payload(messages=[]),
            )

    assert type(raised.value).__name__ == "PydanticSerializationError"
    performance = latest_performance(sessions)
    assert performance.outcome == "failed"
    assert performance.failure is not None
    assert performance.failure.category == FailureCategory.INTERNAL
    assert performance.failure.stage == FailureStage.CLIENT_TRANSLATION
    assert performance.failure.code == "unexpected_exception"
    assert caplog.text.count("performance outcome=completed") == 0
    assert caplog.text.count("performance outcome=failed") == 1
    assert "UnsupportedJsonValue" not in caplog.text


def test_nonstream_json_bytes_and_headers_remain_exact():
    response = client().post("/v1/messages", json=messages_payload())

    assert response.headers["content-type"] == "application/json"
    assert response.content == (
        b'{"id":"msg-1","model":"claude-sonnet","role":"assistant",'
        b'"content":[{"type":"text","text":"hello"}],"type":"message",'
        b'"stop_reason":"end_turn","stop_sequence":null,"usage":'
        b'{"input_tokens":2,"output_tokens":1,'
        b'"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}'
    )


@pytest.mark.asyncio
async def test_send_failure_before_body_start_finalizes_disconnected(caplog):
    sessions = registry()
    response, scope, receive = await _streaming_route_response(
        Provider(), sessions
    )

    async def send(_message):
        raise OSError("SEND_SECRET")

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)

    gc.collect()
    performance = sessions.performance_snapshots().sessions[0].performance
    assert performance.current_concurrency == 0
    assert len(performance.recent_requests) == 1
    assert performance.recent_requests[0].outcome == "client_disconnected"
    assert caplog.text.count("performance outcome=client_disconnected") == 1
    assert "unexpected request failure" not in caplog.text
    assert "SEND_SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_body_send_failure_closes_provider_once_and_finalizes_disconnected(caplog):
    sessions = registry()
    provider = ClosingSendFailureProvider()
    response, scope, receive = await _streaming_route_response(
        provider, sessions
    )
    body_messages = 0

    async def send(message):
        nonlocal body_messages
        if message["type"] != "http.response.body":
            return
        body_messages += 1
        if body_messages == 3:
            raise OSError("BODY_SEND_SECRET")

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)

    gc.collect()
    performance = sessions.performance_snapshots().sessions[0].performance
    assert body_messages == 3
    assert provider.close_calls == 1
    assert performance.current_concurrency == 0
    assert len(performance.recent_requests) == 1
    assert performance.recent_requests[0].outcome == "client_disconnected"
    assert caplog.text.count("performance outcome=client_disconnected") == 1
    assert "unexpected request failure" not in caplog.text
    assert "BODY_SEND_SECRET" not in caplog.text


@pytest.mark.parametrize(
    ("disconnected", "expected"),
    [(False, "cancelled"), (True, "client_disconnected")],
)
@pytest.mark.asyncio
async def test_send_cancellation_uses_request_disconnect_state(
    disconnected, expected
):
    sessions = registry()
    response, scope, receive = await _streaming_route_response(
        Provider(), sessions, disconnected=disconnected
    )

    async def send(_message):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await response(scope, receive, send)

    assert latest_performance(sessions).outcome == expected


@pytest.mark.asyncio
async def test_unexpected_send_failure_records_client_translation_diagnostic(caplog):
    sessions = registry()
    response, scope, receive = await _streaming_route_response(
        Provider(), sessions
    )

    async def send(_message):
        raise ValueError("SEND_VALUE_SECRET")

    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        with pytest.raises(ValueError, match="SEND_VALUE_SECRET"):
            await response(scope, receive, send)

    performance = latest_performance(sessions)
    assert performance.outcome == "failed"
    assert performance.failure is not None
    assert performance.failure.stage == FailureStage.CLIENT_TRANSLATION
    assert performance.failure.exception_type == "ValueError"
    assert caplog.text.count("unexpected request failure") == 1
    assert "SEND_VALUE_SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_asgi_23_disconnect_before_body_pull_finalizes_disconnected(caplog):
    sessions = registry()
    response, scope, _ = await _streaming_route_response(
        Provider(), sessions
    )
    scope["asgi"]["spec_version"] = "2.3"
    sent = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        await asyncio.sleep(0)

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        await response(scope, receive, send)

    gc.collect()
    assert [message["type"] for message in sent] == ["http.response.start"]
    performance = sessions.performance_snapshots().sessions[0].performance
    assert performance.current_concurrency == 0
    assert len(performance.recent_requests) == 1
    assert performance.recent_requests[0].outcome == "client_disconnected"
    assert caplog.text.count("performance outcome=client_disconnected") == 1


@pytest.mark.asyncio
async def test_observer_close_provider_error_stays_provider_failure(caplog):
    error = ProviderError("PROVIDER_CLOSE_SECRET", provider="fake")
    events = ProviderCloseErrorEvents(error)
    sessions = registry()
    observation = sessions.begin(stream_metadata())
    request = DisconnectRequest(False)
    context = stream_context()
    prepared = routes_module.normalize_request(
        routes_module.MessagesRequest(
            model="claude-sonnet", max_tokens=10, messages=[]
        )
    )
    observed = observe_stream(
        events,
        context,
        on_provider_error=lambda provider_error: (
            routes_module._mark_observed_provider_error(
                request, provider_error
            )
        ),
    )
    serialized = routes_module.serialize_stream(prepared, observed)
    stream = routes_module._record_stream_lifecycle(
        serialized, request, context, sessions, observation
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        await anext(stream)
        await anext(stream)
        with pytest.raises(ProviderError) as raised:
            await anext(stream)

    assert raised.value is error
    assert events.close_calls == 1
    assert caplog.text.count("provider request failed") == 1
    assert "unexpected request failure" not in caplog.text
    performance = latest_performance(sessions)
    assert performance.outcome == "failed"
    assert performance.failure == logging_module.provider_failure_diagnostic(error)
    assert "PROVIDER_CLOSE_SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_full_stack_body_oserror_finalizes_disconnected(caplog):
    sessions = registry()
    provider = ClosingSendFailureProvider()
    body_messages = 0

    send_error = OSError("FULL_STACK_OSERROR_SECRET")

    async def send(message):
        nonlocal body_messages
        if message["type"] != "http.response.body":
            return
        body_messages += 1
        if body_messages == 3:
            raise send_error

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        with pytest.raises(OSError) as raised:
            await _call_full_stack_stream(provider, sessions, send)

    assert raised.value is send_error

    performance = sessions.performance_snapshots().sessions[0].performance
    assert provider.close_calls == 1
    assert performance.current_concurrency == 0
    assert len(performance.recent_requests) == 1
    assert performance.recent_requests[0].outcome == "client_disconnected"
    assert caplog.text.count("performance outcome=client_disconnected") == 1
    assert "unexpected request failure" not in caplog.text
    assert "FULL_STACK_OSERROR_SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_full_stack_body_value_error_retains_client_translation(caplog):
    sessions = registry()
    provider = ClosingSendFailureProvider()
    body_messages = 0

    async def send(message):
        nonlocal body_messages
        if message["type"] != "http.response.body":
            return
        body_messages += 1
        if body_messages == 3:
            raise ValueError("FULL_STACK_VALUE_SECRET")

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        with pytest.raises(ValueError, match="FULL_STACK_VALUE_SECRET"):
            await _call_full_stack_stream(provider, sessions, send)

    performance = sessions.performance_snapshots().sessions[0].performance
    assert provider.close_calls == 1
    assert performance.current_concurrency == 0
    assert len(performance.recent_requests) == 1
    terminal = performance.recent_requests[0]
    assert terminal.outcome == "failed"
    assert terminal.failure is not None
    assert terminal.failure.stage == FailureStage.CLIENT_TRANSLATION
    assert terminal.failure.exception_type == "ValueError"
    assert caplog.text.count("unexpected request failure") == 1
    assert caplog.text.count("performance outcome=failed") == 1
    assert "FULL_STACK_VALUE_SECRET" not in caplog.text
