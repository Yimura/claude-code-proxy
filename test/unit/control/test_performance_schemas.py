from datetime import UTC, datetime
import json
from types import SimpleNamespace

from pydantic import TypeAdapter, ValidationError
import pytest

import claude_code_proxy.control.schemas as schemas_module
from claude_code_proxy.control.schemas import (
    FailureDiagnosticResponse,
    MetricAggregateResponse,
    MetricResponse,
    PerformanceEventResponse,
    PerformanceListResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
    ProcessIdentityResponse,
    RequestPerformanceResponse,
    SessionPerformanceResponse,
    SessionPerformanceViewResponse,
)
from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.observability import SessionMetadata, SessionRegistry


class RegistryClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 1, 1, tzinfo=UTC)
        self.monotonic = 100.0

    def wall_now(self) -> datetime:
        return self.wall

    def monotonic_now(self) -> float:
        return self.monotonic


def metadata(client_id: str, agent_id: str | None = None) -> SessionMetadata:
    return SessionMetadata(
        client_identity=ClientIdentity(client_id, agent_id),
        client_model="claude-opus",
        upstream_model="openai/gpt-5.6-sol",
        provider="openai",
        transport="codex",
        effort="high",
        context_window=1_000_000,
    )


def registry(clock: RegistryClock) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=10,
        secret=b"control-test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
    )


def process_payload() -> dict[str, object]:
    return {"pid": 42, "started_at": "2026-01-02T03:00:00Z"}
def failure_payload() -> dict[str, object]:
    return {"category": "internal", "stage": "route", "code": "safe"}
def metric_payload(status: str = "observed", value: object = 0) -> dict[str, object]:
    return {"status": status, "value": value}
def aggregate_payload(value: object = 0) -> dict[str, object]:
    return {"value": value, "observed_samples": 1,
            "unavailable_samples": 0, "not_applicable_samples": 0}
_REQUEST_METRICS = (
    "duration", "upstream_duration", "ttft", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_creation_tokens", "reasoning_tokens",
    "tool_calls", "retries", "peak_concurrency",
)
_AGGREGATE_METRICS = (
    "input_tokens", "output_tokens", "cache_read_tokens",
    "cache_creation_tokens", "reasoning_tokens", "tool_calls", "retries",
)
def request_performance_payload() -> dict[str, object]:
    return {
        "id": "request-public", "session_id": "session-public",
        "operation": "messages", "outcome": "completed",
        "started_at": "2026-01-02T03:04:05Z",
        "finished_at": "2026-01-02T03:04:06Z",
        **{name: metric_payload() for name in _REQUEST_METRICS},
        "reasoning_continuation": "not_applicable", "failure": None,
    }
def session_performance_payload() -> dict[str, object]:
    request = request_performance_payload()
    return {
        "session_id": "session-public", "requests": 1, "active_requests": [],
        "recent_requests": [request], "outcomes": {"completed": 1},
        **{name: aggregate_payload() for name in _AGGREGATE_METRICS},
        "current_concurrency": 0, "peak_concurrency": 1,
        "latest_request": request_performance_payload(),
    }

def performance_view_payload() -> dict[str, object]:
    return {
        "session": {
            "id": "session-public",
            "state": "idle",
            "active_requests": 0,
            "requests": 1,
            "client_model": "client-model",
            "model": "provider-model",
            "provider": "provider",
            "transport": "transport",
            "effort": "high",
            "context_window": 1000,
            "first_seen": "2026-01-02T03:04:05Z",
            "last_seen": "2026-01-02T03:04:06Z",
            "elapsed_seconds": 1,
            "last_result": "completed",
        },
        "performance": session_performance_payload(),
    }

def performance_list_payload() -> dict[str, object]:
    return {"process": process_payload(),
            "captured_at": "2026-01-02T03:04:06Z", "cursor": 1,
            "sessions": [performance_view_payload()]}

def performance_event_payload() -> dict[str, object]:
    return {
        "process": process_payload(),
        "sequence": 1,
        "occurred_at": "2026-01-02T03:04:06Z",
        "type": "completed",
        "session_id": "session-public",
        "activity": activity_payload(),
        "request": request_performance_payload(),
        "session": session_performance_payload(),
    }

def performance_reset_payload() -> dict[str, object]:
    snapshot = performance_list_payload()
    return {"process": snapshot["process"], "sequence": snapshot["cursor"],
            "occurred_at": "2026-01-02T03:04:06Z", "type": "reset",
            "snapshot": snapshot}


def activity_payload() -> dict[str, object]:
    return dict(performance_view_payload()["session"])


def cursor_payload() -> dict[str, object]:
    return {
        "process": process_payload(),
        "sequence": 1,
        "occurred_at": "2026-01-02T03:04:06Z",
        "type": "cursor",
    }


def test_performance_event_exposes_coherent_activity() -> None:
    payload = performance_event_payload()
    payload["activity"] = activity_payload()

    event = PerformanceEventResponse.model_validate(payload)

    assert event.activity.id == event.session_id
    assert event.activity.requests == event.session.requests
    assert event.activity.active_requests == event.session.current_concurrency


def test_cursor_response_is_strict_frozen_and_discriminated() -> None:
    cursor_type = getattr(schemas_module, "PerformanceCursorResponse", None)
    assert cursor_type is not None
    cursor = cursor_type.model_validate(cursor_payload())
    event = TypeAdapter(schemas_module.PerformanceStreamEvent).validate_python(
        cursor_payload()
    )

    assert event == cursor
    with pytest.raises(ValidationError, match="frozen"):
        cursor.sequence = 2
    payload = cursor_payload()
    payload["session_id"] = "forbidden"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        cursor_type.model_validate(payload)


_PERFORMANCE_SCHEMA_CASES = (
    (ProcessIdentityResponse, process_payload),
    (MetricResponse, metric_payload),
    (MetricAggregateResponse, aggregate_payload),
    (FailureDiagnosticResponse, failure_payload),
    (RequestPerformanceResponse, request_performance_payload),
    (SessionPerformanceResponse, session_performance_payload),
    (SessionPerformanceViewResponse, performance_view_payload),
    (PerformanceListResponse, performance_list_payload),
    (PerformanceEventResponse, performance_event_payload),
    (PerformanceResetResponse, performance_reset_payload),
)

def test_stream_sequences_and_reset_consistency_are_validated() -> None:
    event = performance_event_payload()
    event["sequence"] = 0
    with pytest.raises(ValidationError):
        PerformanceEventResponse.model_validate(event)
    reset = PerformanceResetResponse.model_validate(performance_reset_payload())
    assert reset.sequence == 1
    payload = performance_reset_payload()
    payload["sequence"] = 0
    payload["snapshot"]["cursor"] = 0
    assert PerformanceResetResponse.model_validate(payload).sequence == 0
    for field in ("sequence", "process"):
        payload = performance_reset_payload()
        payload[field] = 2 if field == "sequence" else {
            "pid": 43,
            "started_at": "2026-01-02T03:00:00Z",
        }
        with pytest.raises(ValidationError):
            PerformanceResetResponse.model_validate(payload)

def test_stream_event_alias_discriminates_ordinary_and_reset_events() -> None:
    adapter = TypeAdapter(PerformanceStreamEvent)
    ordinary = adapter.validate_python(performance_event_payload())
    reset = adapter.validate_python(performance_reset_payload())
    assert isinstance(ordinary, PerformanceEventResponse)
    assert isinstance(reset, PerformanceResetResponse)

@pytest.mark.parametrize(
    ("model", "factory"),
    _PERFORMANCE_SCHEMA_CASES,
)
@pytest.mark.parametrize("field", ["prompt", "raw_session_id", "provider_payload"])
def test_performance_schemas_forbid_sensitive_extra_fields(
    model, factory, field: str
) -> None:
    payload = factory()
    payload[field] = "secret"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate(payload)

def test_session_performance_outcomes_are_closed_strict_and_bounded() -> None:
    payload = session_performance_payload()
    source = payload["outcomes"]
    response = SessionPerformanceResponse.model_validate(payload)
    source["active"] = 1
    with pytest.raises(TypeError):
        response.outcomes["active"] = 1
    assert response.model_dump(mode="json")["outcomes"] == {"completed": 1}
    for outcomes in (
        {"active": 1}, {"unknown": 1}, {"completed": True},
        {"completed": "1"}, {"completed": -1},
        {"completed": MAX_CONTROL_INTEGER + 1},
    ):
        payload = session_performance_payload()
        payload["outcomes"] = outcomes
        with pytest.raises(ValidationError):
            SessionPerformanceResponse.model_validate(payload)

def test_real_registry_capture_converts_without_raw_content() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    handle = sessions.begin(
        metadata("raw-session-secret", "raw-agent-secret"), operation="count_tokens"
    )
    sessions.observer(handle).count_tokens(0)
    sessions.finish(handle, "completed")
    capture = sessions.performance_snapshots()
    wrapped = SimpleNamespace(
        process=SimpleNamespace(pid=42, started_at=clock.wall),
        captured_at=capture.captured_at,
        cursor=capture.cursor,
        sessions=capture.sessions,
    )
    response = PerformanceListResponse.model_validate(wrapped, from_attributes=True)
    dumped = response.model_dump(mode="json")
    serialized = json.dumps(dumped)
    performance = response.sessions[0].performance
    assert performance.recent_requests == (performance.latest_request,)
    recent = dumped["sessions"][0]["performance"]["recent_requests"][0]
    assert recent["input_tokens"] == {"status": "observed", "value": 0}
    input_tokens = dumped["sessions"][0]["performance"]["input_tokens"]
    assert input_tokens["observed_samples"] == 1
    assert handle.public_id in serialized
    secrets = (
        "raw-session-secret",
        "prompt",
        "raw_session_id",
        "provider_payload",
    )
    for secret in secrets:
        assert secret not in serialized

@pytest.mark.parametrize(
    ("model", "factory"),
    _PERFORMANCE_SCHEMA_CASES,
)
def test_performance_schema_models_are_frozen(model, factory) -> None:
    instance = model.model_validate(factory())
    field = next(iter(model.model_fields))
    with pytest.raises(ValidationError, match="frozen"):
        setattr(instance, field, None)

@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("view.performance.session_id", "other"),
        ("view.session.requests", 2),
        ("view.session.active_requests", 1),
        ("view.session.state", "active"),
        ("event.session_id", "other"),
        ("event.activity.id", "other"),
        ("event.activity.requests", 2),
        ("event.activity.active_requests", 1),
        ("event.activity.state", "active"),
        ("event.request.session_id", "other"),
        ("event.session.session_id", "other"),
        ("event.request.outcome", "failed"),
        ("event.type", "progress"),
        ("event.request.operation", "count_tokens"),
        ("event.request.id", "other"),
        ("active.request.operation", "count_tokens"),
        ("active.request.id", "other"),
    ],
)
def test_performance_envelopes_reject_inconsistent_lifecycle(path: str, value: object) -> None:
    envelope, *parts = path.split(".")
    is_view = envelope == "view"
    model = SessionPerformanceViewResponse if is_view else PerformanceEventResponse
    factories = {"view": performance_view_payload, "event": performance_event_payload,
                 "active": active_performance_event_payload}
    payload = factories[envelope]()
    target = payload
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValidationError):
        model.model_validate(payload)

@pytest.mark.parametrize(
    ("model", "factory", "path"),
    [
        (RequestPerformanceResponse, request_performance_payload, ("started_at",)),
        (RequestPerformanceResponse, request_performance_payload, ("finished_at",)),
        (PerformanceListResponse, performance_list_payload, ("captured_at",)),
        (PerformanceEventResponse, performance_event_payload, ("occurred_at",)),
        (PerformanceResetResponse, performance_reset_payload, ("occurred_at",)),
    ],
)
def test_performance_envelopes_reject_numeric_datetime_strings(
    model, factory, path: tuple[str, ...]
) -> None:
    payload = factory()
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = "0"
    with pytest.raises(ValidationError):
        model.model_validate(payload)

def performance_view_with_agent_payload() -> dict[str, object]:
    payload = performance_view_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    agent = dict(session)
    agent.update({"id": "agent-public", "parent_id": None})
    session["agents"] = [agent]
    return payload

def active_request_payload() -> dict[str, object]:
    payload = request_performance_payload()
    payload.update({"outcome": "active", "finished_at": None})
    return payload

def active_session_performance_payload() -> dict[str, object]:
    request = active_request_payload()
    payload = session_performance_payload()
    payload.update({"active_requests": [request], "recent_requests": [],
                    "outcomes": {}, "current_concurrency": 1,
                    "latest_request": request})
    return payload

def active_performance_event_payload() -> dict[str, object]:
    payload = performance_event_payload()
    payload.update({"type": "progress", "request": active_request_payload(),
                    "session": active_session_performance_payload()})
    payload["activity"].update(
        {"state": "active", "active_requests": 1, "last_result": None}
    )
    return payload

@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        (("session",), "raw_session_id", "secret"),
        (("session", "agents", 0), "prompt", "secret"),
        (("session",), "first_seen", 0),
        (("session",), "last_seen", "0"),
        (("session", "agents", 0), "first_seen", 0),
        (("session", "agents", 0), "last_seen", "0"),
        (("session",), "client_model", b"model"),
        (("session", "agents", 0), "id", b"agent"),
        (("session",), "elapsed_seconds", float("inf")),
        (("session", "agents", 0), "elapsed_seconds", float("nan")),
        (("session",), "active_requests", True),
        (("session",), "requests", "1"),
        (("session", "agents", 0), "requests", MAX_CONTROL_INTEGER + 1),
        (("session",), "elapsed_seconds", -1),
    ],
)
def test_performance_activity_models_reject_unsafe_nested_values(
    path: tuple[str | int, ...], field: str, value: object
) -> None:
    payload = performance_view_with_agent_payload()
    target = payload
    for part in path:
        target = target[part]
    target[field] = value
    with pytest.raises(ValidationError):
        SessionPerformanceViewResponse.model_validate(payload)

@pytest.mark.parametrize(
    "updates",
    [
        {"outcome": "active"},
        {"finished_at": None},
        {"failure": failure_payload()},
        {"outcome": "active", "finished_at": None, "failure": failure_payload()},
    ],
)
def test_request_performance_rejects_incoherent_lifecycle(
    updates: dict[str, object]
) -> None:
    payload = request_performance_payload()
    payload.update(updates)
    with pytest.raises(ValidationError):
        RequestPerformanceResponse.model_validate(payload)

def test_request_performance_accepts_active_and_failed_without_diagnostic() -> None:
    assert RequestPerformanceResponse.model_validate(active_request_payload()).outcome == "active"
    failed = request_performance_payload()
    failed.update({"outcome": "failed", "failure": None})
    assert RequestPerformanceResponse.model_validate(failed).outcome == "failed"

def test_session_performance_rejects_incoherent_snapshots() -> None:
    invalid = []
    active = active_session_performance_payload()
    active["active_requests"][0]["session_id"] = "other"
    invalid.append(active)
    terminal_active = session_performance_payload()
    terminal_active.update({"active_requests": [request_performance_payload()], "recent_requests": [], "outcomes": {}, "current_concurrency": 1})
    invalid.append(terminal_active)
    active_recent = session_performance_payload()
    live = active_request_payload()
    active_recent.update({"requests": 0, "recent_requests": [live], "outcomes": {}, "latest_request": live})
    invalid.append(active_recent)
    for field, value in (("current_concurrency", 1), ("requests", 2), ("latest_request", None)):
        payload = session_performance_payload()
        payload[field] = value
        invalid.append(payload)
    peak = active_session_performance_payload()
    peak["peak_concurrency"] = 0
    invalid.append(peak)
    latest = session_performance_payload()
    latest["latest_request"] = request_performance_payload()
    latest["latest_request"]["operation"] = "count_tokens"
    invalid.append(latest)
    empty = session_performance_payload()
    empty.update({"requests": 0, "recent_requests": [], "outcomes": {}})
    invalid.append(empty)
    for payload in invalid:
        with pytest.raises(ValidationError):
            SessionPerformanceResponse.model_validate(payload)


@pytest.mark.parametrize("location", ["active", "recent", "combined"])
def test_session_performance_rejects_duplicate_request_ids(location: str) -> None:
    payload = session_performance_payload()
    if location == "active":
        request = active_request_payload()
        payload.update({"requests": 2, "active_requests": [request, request],
                        "recent_requests": [], "outcomes": {},
                        "current_concurrency": 2, "peak_concurrency": 2,
                        "latest_request": request})
    elif location == "recent":
        request = request_performance_payload()
        payload.update({"requests": 2, "recent_requests": [request, request],
                        "outcomes": {"completed": 2}, "latest_request": request})
    else:
        payload.update({"requests": 2,
                        "active_requests": [active_request_payload()],
                        "outcomes": {"completed": 1}, "current_concurrency": 1})
    with pytest.raises(ValidationError):
        SessionPerformanceResponse.model_validate(payload)
