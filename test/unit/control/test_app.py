import asyncio
import json
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import TypeAdapter, ValidationError

import claude_code_proxy.control.app as control_app_module
from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.control.schemas import (
    FailureDiagnosticResponse,
    HealthResponse,
    MetricAggregateResponse,
    MetricResponse,
    PerformanceEventResponse,
    PerformanceListResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
    ProcessIdentityResponse,
    RequestPerformanceResponse,
    SessionCounts,
    SessionListResponse,
    SessionPerformanceResponse,
    SessionPerformanceViewResponse,
    SessionResponse,
)
from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.event_journal import EventJournal, Subscription
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.observability import (
    SessionMetadata,
    SessionRegistry,
    SessionSnapshot,
)


class RegistryClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 1, 1, tzinfo=UTC)
        self.monotonic = 100.0

    def wall_now(self) -> datetime:
        return self.wall

    def monotonic_now(self) -> float:
        return self.monotonic

    def advance(self, seconds: float = 1.0) -> None:
        self.wall += timedelta(seconds=seconds)
        self.monotonic += seconds

def metadata(
    client_id: str,
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
    *,
    client_model: str = "claude-opus",
    upstream_model: str = "openai/gpt-5.6-sol",
    provider: str = "openai",
    transport: str = "codex",
    effort: str = "high",
    context_window: int | None = 1_000_000,
) -> SessionMetadata:
    return SessionMetadata(
        client_identity=ClientIdentity(
            client_id,
            agent_id,
            parent_agent_id,
        ),
        client_model=client_model,
        upstream_model=upstream_model,
        provider=provider,
        transport=transport,
        effort=effort,
        context_window=context_window,
    )

def registry(
    clock: RegistryClock, inactive_limit: int = 10
) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=inactive_limit,
        secret=b"control-test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
    )

def filtered_registry() -> tuple[SessionRegistry, dict[str, str]]:
    clock = RegistryClock()
    sessions = registry(clock)
    active = sessions.begin(metadata("raw-active"))
    clock.advance()
    idle = sessions.begin(
        metadata(
            "raw-idle",
            client_model="claude-sonnet",
            upstream_model="vertex/claude-sonnet-5",
            provider="vertex",
            transport="litellm",
            effort="medium",
            context_window=None,
        )
    )
    sessions.finish(idle, "completed")
    clock.advance()
    failed = sessions.begin(
        metadata(
            "raw-failed",
            client_model="claude-haiku",
            upstream_model="gemini/gemini-pro",
            provider="gemini",
            transport="litellm",
            effort="low",
            context_window=200_000,
        )
    )
    sessions.finish(failed, "failed")
    return sessions, {
        "active": active.public_id,
        "idle": idle.public_id,
        "failed": failed.public_id,
    }

async def request(app, path: str, params=None):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://control"
    ) as client:
        return await client.get(path, params=params)

async def test_session_response_includes_agent_snapshots() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    handle = sessions.begin(metadata("session", "agent", "parent"))
    app = create_control_app(
        sessions,
        application_version="1.0",
        pid=123,
    )
    response = await request(app, "/v1/sessions")
    assert response.status_code == 200
    agent = response.json()["sessions"][0]["agents"][0]
    assert agent["id"] == handle.agent_public_id
    assert agent["parent_id"] == handle.parent_agent_public_id
    assert "client_session_id" not in agent
    assert "agent_id" not in agent

async def test_health_counts_roots_not_agent_rows() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    sessions.begin(metadata("session", "first"))
    sessions.begin(metadata("session", "second"))
    app = create_control_app(
        sessions,
        application_version="1.0",
        pid=123,
    )
    response = await request(app, "/v1/health")
    assert response.json()["capabilities"] == [
        "sessions",
        "agents",
        "performance",
        "performance_events",
    ]
    assert response.json()["sessions"] == {"active": 1, "retained": 1}

async def test_health_reports_exact_version_process_time_limit_and_counts() -> None:
    registry_clock = RegistryClock()
    sessions = registry(registry_clock, inactive_limit=7)
    sessions.begin(metadata("raw-active"))
    inactive = sessions.begin(metadata("raw-inactive"))
    sessions.finish(inactive, "completed")
    started = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    now = started + timedelta(seconds=12.5)
    app = create_control_app(
        sessions,
        started_at=started,
        application_version="9.8.7",
        pid=4321,
        clock=lambda: now,
    )
    response = await request(app, "/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "protocol_version": 1,
        "application_version": "9.8.7",
        "pid": 4321,
        "started_at": "2026-01-02T03:04:05Z",
        "uptime_seconds": 12.5,
        "capabilities": [
            "sessions",
            "agents",
            "performance",
            "performance_events",
        ],
        "sessions": {"active": 1, "retained": 2},
        "inactive_limit": 7,
    }

async def test_health_clamps_uptime_when_clock_precedes_start() -> None:
    clock = RegistryClock()
    started = datetime(2026, 1, 2, tzinfo=UTC)
    app = create_control_app(
        registry(clock),
        started_at=started,
        application_version="1.0",
        pid=1,
        clock=lambda: started - timedelta(seconds=1),
    )
    response = await request(app, "/v1/health")
    assert response.json()["uptime_seconds"] == 0

async def test_health_uses_installed_distribution_version(monkeypatch) -> None:
    clock = RegistryClock()
    requested_distributions: list[str] = []
    def version(distribution: str) -> str:
        requested_distributions.append(distribution)
        return "4.5.6"
    monkeypatch.setattr(control_app_module.metadata, "version", version)
    app = create_control_app(
        registry(clock),
        started_at=clock.wall,
        pid=1,
        clock=clock.wall_now,
    )
    response = await request(app, "/v1/health")
    assert response.json()["application_version"] == "4.5.6"
    assert requested_distributions == ["anthropic-proxy"]

async def test_control_app_defaults_are_resolved_at_construction(monkeypatch) -> None:
    registry_clock = RegistryClock()
    started = datetime(2026, 4, 5, 6, 7, 8, tzinfo=UTC)
    current = started + timedelta(seconds=3)
    times = iter((started, current))
    requested_distributions: list[str] = []
    monkeypatch.setattr(control_app_module, "_utc_now", lambda: next(times))
    monkeypatch.setattr(control_app_module.os, "getpid", lambda: 2468)
    def version(distribution: str) -> str:
        requested_distributions.append(distribution)
        return "7.8.9"
    monkeypatch.setattr(control_app_module.metadata, "version", version)
    app = create_control_app(registry(registry_clock))
    response = await request(app, "/v1/health")
    assert response.status_code == 200
    assert response.json()["started_at"] == "2026-04-05T06:07:08Z"
    assert response.json()["uptime_seconds"] == 3
    assert response.json()["application_version"] == "7.8.9"
    assert response.json()["pid"] == 2468
    assert requested_distributions == ["anthropic-proxy"]

async def test_sessions_preserve_registry_order_and_one_capture_time() -> None:
    registry_clock = RegistryClock()
    sessions = registry(registry_clock)
    first = sessions.begin(metadata("raw-first"))
    registry_clock.advance()
    second = sessions.begin(metadata("raw-second"))
    captured_at = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
    app = create_control_app(
        sessions,
        started_at=captured_at,
        application_version="1.0",
        pid=1,
        clock=lambda: captured_at,
    )
    response = await request(app, "/v1/sessions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["captured_at"] == "2026-02-03T04:05:06Z"
    assert [item["id"] for item in payload["sessions"]] == [
        second.public_id,
        first.public_id,
    ]

@pytest.mark.parametrize(
    ("entry", "expected_name"),
    [
        ("id={id_prefix}", "idle"),
        ("session_id=raw-idle", "idle"),
        ("state=FAILED", "failed"),
        ("provider=VERTEX", "idle"),
        ("transport=CODEX", "active"),
        ("model=CLAUDE-SONNET-5", "idle"),
        ("effort=LOW", "failed"),
    ],
)
async def test_sessions_support_each_filter_key(
    entry: str, expected_name: str
) -> None:
    sessions, ids = filtered_registry()
    value = entry.format(id_prefix=ids["idle"][:12].upper())
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    response = await request(app, "/v1/sessions", [("filter", value)])
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["sessions"]] == [
        ids[expected_name]
    ]

async def test_session_id_filter_never_returns_raw_id() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    raw_id = "raw-idle"
    response = await request(
        app,
        "/v1/sessions",
        [("filter", f"session_id={raw_id}")],
    )
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["sessions"]] == [
        ids["idle"]
    ]
    assert raw_id not in response.text

async def test_repeated_filter_key_is_or_and_different_keys_are_and() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    or_response = await request(
        app,
        "/v1/sessions",
        [("filter", "state=idle"), ("filter", "state=failed")],
    )
    and_response = await request(
        app,
        "/v1/sessions",
        [("filter", "transport=litellm"), ("filter", "effort=medium")],
    )
    assert {item["id"] for item in or_response.json()["sessions"]} == {
        ids["idle"],
        ids["failed"],
    }
    assert [item["id"] for item in and_response.json()["sessions"]] == [
        ids["idle"]
    ]

@pytest.mark.parametrize(
    "entry",
    [
        "missing-separator",
        " =value",
        "key=   ",
        "unknown=value",
        "state=idle=extra",
        "state==idle",
    ],
)
async def test_malformed_filter_entries_return_422(entry: str) -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    response = await request(app, "/v1/sessions", [("filter", entry)])
    assert response.status_code == 422

async def test_filter_entry_length_accepts_256_and_rejects_257_characters() -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    exactly_256 = "model=" + "x" * 250
    exactly_257 = "model=" + "x" * 251
    accepted = await request(
        app, "/v1/sessions", [("filter", exactly_256)]
    )
    rejected = await request(
        app, "/v1/sessions", [("filter", exactly_257)]
    )
    assert len(exactly_256) == 256
    assert accepted.status_code == 200
    assert accepted.json()["sessions"] == []
    assert len(exactly_257) == 257
    assert rejected.status_code == 422

async def test_invalid_session_id_filter_does_not_echo_raw_value() -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    raw_id = "sensitive-session-" + "x" * 240
    response = await request(
        app,
        "/v1/sessions",
        [("filter", f"session_id={raw_id}")],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid session filter"}
    assert raw_id not in response.text

async def test_exactly_32_filter_entries_are_accepted() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    entries = [("filter", "state=idle") for _ in range(32)]
    response = await request(app, "/v1/sessions", entries)
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["sessions"]] == [ids["idle"]]

async def test_filter_whitespace_is_normalized() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    response = await request(
        app, "/v1/sessions", [("filter", " state = idle ")]
    )
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["sessions"]] == [ids["idle"]]

async def test_more_than_32_filter_entries_returns_422() -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    entries = [("filter", "state=idle") for _ in range(33)]
    response = await request(app, "/v1/sessions", entries)
    assert response.status_code == 422

async def test_ambiguous_id_prefix_returns_safe_422() -> None:
    registry_clock = RegistryClock()
    sessions = registry(registry_clock, inactive_limit=20)
    seen: set[str] = set()
    collision = None
    for index in range(17):
        prefix = sessions.begin(metadata(f"raw-collision-{index}")).public_id[0]
        if prefix in seen:
            collision = prefix
            break
        seen.add(prefix)
    assert collision is not None
    app = create_control_app(
        sessions,
        started_at=registry_clock.wall,
        application_version="1.0",
        pid=1,
    )
    response = await request(
        app, "/v1/sessions", [("filter", f"id={collision}")]
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Session ID prefix is ambiguous"}
    serialized = response.text
    assert "raw-collision" not in serialized
    assert all(snapshot.id not in serialized for snapshot in sessions.snapshots())

async def test_session_json_excludes_raw_ids_and_internal_fields() -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    response = await request(app, "/v1/sessions")
    assert response.status_code == 200
    serialized = json.dumps(response.json())
    assert "raw-active" not in serialized
    assert "raw-idle" not in serialized
    assert "raw-failed" not in serialized
    expected_fields = {
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
    assert all(set(item) == expected_fields for item in response.json()["sessions"])

@pytest.mark.parametrize(
    "path",
    [
        "/docs",
        "/openapi.json",
        "/redoc",
        "/",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/api/hello",
    ],
)
async def test_control_app_exposes_only_versioned_control_routes(path: str) -> None:
    clock = RegistryClock()
    app = create_control_app(
        registry(clock),
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
    )
    response = await request(app, path)
    assert response.status_code == 404

def test_health_schemas_are_frozen_and_protocol_version_is_literal_one() -> None:
    counts = SessionCounts(active=1, retained=2)
    values = {
        "application_version": "1.0",
        "pid": 123,
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        "uptime_seconds": 4.0,
        "sessions": counts,
        "inactive_limit": 5,
    }
    health = HealthResponse(**values)
    assert health.protocol_version == 1
    with pytest.raises(ValidationError, match="frozen"):
        counts.active = 2
    with pytest.raises(ValidationError, match="frozen"):
        health.pid = 456
    with pytest.raises(ValidationError, match="Input should be 1"):
        HealthResponse(protocol_version=2, **values)

def test_session_response_validates_snapshot_attributes_and_is_frozen() -> None:
    snapshot = SessionSnapshot(
        id="a" * 64,
        state="idle",
        active_requests=0,
        requests=3,
        client_model="claude-opus",
        model="gpt-5.6-sol",
        provider="openai",
        transport="codex",
        effort="high",
        context_window=1_000_000,
        first_seen=datetime(2026, 1, 1, tzinfo=UTC),
        last_seen=datetime(2026, 1, 2, tzinfo=UTC),
        elapsed_seconds=2.5,
        last_result="completed",
    )
    response = SessionResponse.model_validate(snapshot)
    assert response.model_dump() == snapshot.__dict__
    assert "client_session_id" not in SessionResponse.model_fields
    with pytest.raises(ValidationError, match="frozen"):
        response.requests = 4

def test_session_list_response_stores_a_frozen_tuple() -> None:
    snapshot = SessionSnapshot(
        id="b" * 64,
        state="active",
        active_requests=1,
        requests=1,
        client_model="claude-sonnet",
        model="gpt-5.6-sol",
        provider="openai",
        transport="codex",
        effort="medium",
        context_window=None,
        first_seen=datetime(2026, 1, 1, tzinfo=UTC),
        last_seen=datetime(2026, 1, 1, tzinfo=UTC),
        elapsed_seconds=1.0,
        last_result=None,
    )
    session = SessionResponse.model_validate(snapshot)
    response = SessionListResponse(
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        sessions=[session],
    )
    assert response.sessions == (session,)
    assert isinstance(response.sessions, tuple)
    with pytest.raises(ValidationError, match="frozen"):
        response.sessions = ()

async def test_control_app_normalizes_aware_datetimes_and_rejects_naive_ones() -> None:
    clock = RegistryClock()
    offset = timezone(timedelta(hours=2))
    app = create_control_app(
        registry(clock),
        started_at=datetime(2026, 1, 1, 2, tzinfo=offset),
        application_version="1.0",
        pid=1,
        clock=lambda: datetime(2026, 1, 1, 3, tzinfo=offset),
    )
    response = await request(app, "/v1/health")
    assert response.json()["started_at"] == "2026-01-01T00:00:00Z"
    assert response.json()["uptime_seconds"] == 3600
    with pytest.raises(ValueError, match="started_at must be timezone-aware"):
        create_control_app(
            registry(clock),
            started_at=datetime(2026, 1, 1),
            application_version="1.0",
            pid=1,
        )
    naive_clock_app = create_control_app(
        registry(clock),
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
        clock=lambda: datetime(2026, 1, 1),
    )
    with pytest.raises(ValueError, match="clock result must be timezone-aware"):
        await request(naive_clock_app, "/v1/health")

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
        "request": request_performance_payload(),
        "session": session_performance_payload(),
    }

def performance_reset_payload() -> dict[str, object]:
    snapshot = performance_list_payload()
    return {"process": snapshot["process"], "sequence": snapshot["cursor"],
            "occurred_at": "2026-01-02T03:04:06Z", "type": "reset",
            "snapshot": snapshot}

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


async def test_performance_snapshot_wraps_atomic_capture_and_process_identity() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    handle = sessions.begin(metadata("raw-performance"))
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        application_version="1.0",
        pid=2468,
    )

    response = await request(app, "/v1/performance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["process"] == {
        "pid": 2468,
        "started_at": "2026-01-02T03:04:05Z",
    }
    assert payload["captured_at"] == "2026-01-01T00:00:00Z"
    assert payload["cursor"] == 1
    assert payload["sessions"][0]["session"]["id"] == handle.public_id
    assert payload["sessions"][0]["performance"]["session_id"] == handle.public_id
    PerformanceListResponse.model_validate(payload)


async def test_performance_snapshot_empty_registry_is_valid() -> None:
    clock = RegistryClock()
    app = create_control_app(
        registry(clock),
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
    )

    response = await request(app, "/v1/performance")

    assert response.status_code == 200
    assert response.json()["cursor"] == 0
    assert response.json()["sessions"] == []


async def test_performance_snapshot_reuses_filter_semantics_and_raw_session_resolution() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )

    by_raw_session = await request(
        app,
        "/v1/performance",
        [("filter", "session_id=raw-idle")],
    )
    repeated_or = await request(
        app,
        "/v1/performance",
        [("filter", "state=idle"), ("filter", "state=failed")],
    )
    different_and = await request(
        app,
        "/v1/performance",
        [("filter", "transport=litellm"), ("filter", "effort=medium")],
    )

    assert [
        item["session"]["id"] for item in by_raw_session.json()["sessions"]
    ] == [ids["idle"]]
    assert {
        item["session"]["id"] for item in repeated_or.json()["sessions"]
    } == {ids["idle"], ids["failed"]}
    assert [
        item["session"]["id"] for item in different_and.json()["sessions"]
    ] == [ids["idle"]]
    assert "raw-idle" not in by_raw_session.text


async def test_performance_snapshot_rejects_invalid_and_ambiguous_filters_generically() -> None:
    clock = RegistryClock()
    sessions = registry(clock, inactive_limit=20)
    prefixes: dict[str, int] = {}
    collision = None
    for index in range(17):
        prefix = sessions.begin(metadata(f"raw-performance-{index}")).public_id[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
        if prefixes[prefix] == 2:
            collision = prefix
            break
    assert collision is not None
    app = create_control_app(
        sessions,
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
    )

    invalid = await request(
        app,
        "/v1/performance",
        [("filter", "unknown=value")],
    )
    ambiguous = await request(
        app,
        "/v1/performance",
        [("filter", f"id={collision}")],
    )

    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "Invalid session filter"}
    assert ambiguous.status_code == 422
    assert ambiguous.json() == {"detail": "Session ID prefix is ambiguous"}
    assert "raw-performance" not in invalid.text + ambiguous.text


async def test_performance_snapshot_excludes_raw_ids_and_provider_content() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    sessions.begin(metadata("raw-private-session"))
    app = create_control_app(
        sessions,
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
    )

    response = await request(app, "/v1/performance")

    assert response.status_code == 200
    serialized = response.text
    for marker in (
        "raw-private-session",
        "raw_session_id",
        "provider_payload",
        "prompt",
        "content",
    ):
        assert marker not in serialized


def stream_registry(
    clock: RegistryClock,
    *,
    capacity: int = 4096,
    subscriber_capacity: int = 64,
) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=10,
        secret=b"control-test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
        events=EventJournal(capacity, subscriber_capacity),
    )


async def performance_stream_response(
    app,
    *,
    filters: list[str] | None = None,
    after: str | None = None,
    pid: str | None = None,
    started_at: str | None = None,
):
    route = next(route for route in app.routes if route.path == "/v1/performance/events")
    return await route.endpoint(
        filter=filters,
        after=after,
        pid=pid,
        started_at=started_at,
    )


async def next_stream_json(response) -> dict[str, object]:
    frame = await anext(response.body_iterator)
    assert isinstance(frame, str)
    assert frame.endswith("\n")
    assert not frame.endswith("\n\n")
    return json.loads(frame)


def control_app_for_stream(
    sessions: SessionRegistry,
    *,
    heartbeat_interval: float = 15.0,
    pid: int = 42,
    started_at: datetime = datetime(2026, 1, 2, 3, tzinfo=UTC),
):
    return create_control_app(
        sessions,
        started_at=started_at,
        application_version="1.0",
        pid=pid,
        heartbeat_interval=heartbeat_interval,
    )


async def test_performance_stream_initial_reset_precedes_immediate_live_event() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    assert response.media_type == "application/x-ndjson"
    assert sessions.events.subscriber_count == 1

    handle = sessions.begin(metadata("raw-live"))
    reset = await next_stream_json(response)
    event = await next_stream_json(response)

    assert reset["type"] == "reset"
    assert reset["sequence"] == reset["snapshot"]["cursor"] == 0
    assert event["type"] == "request_started"
    assert event["sequence"] == 1
    assert event["session_id"] == handle.public_id
    assert event["request"]["id"] == handle.request_id
    assert event["process"] == reset["process"]
    PerformanceResetResponse.model_validate(reset)
    PerformanceEventResponse.model_validate(event)
    await response.body_iterator.aclose()
    assert sessions.events.subscriber_count == 0


async def test_performance_stream_matching_process_replays_retained_then_live() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    first = sessions.begin(metadata("first"))
    started = datetime(2026, 1, 2, 3, tzinfo=UTC)
    app = control_app_for_stream(sessions, started_at=started)
    response = await performance_stream_response(
        app,
        after="0",
        pid="42",
        started_at=started.isoformat(),
    )
    second = sessions.begin(metadata("second"))

    replay = await next_stream_json(response)
    live = await next_stream_json(response)

    assert replay["type"] == "request_started"
    assert replay["sequence"] == 1
    assert replay["request"]["id"] == first.request_id
    assert live["sequence"] == 2
    assert live["request"]["id"] == second.request_id
    await response.body_iterator.aclose()


async def test_performance_stream_after_current_waits_live_without_reset() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("existing"))
    started = datetime(2026, 1, 2, 3, tzinfo=UTC)
    app = control_app_for_stream(
        sessions,
        heartbeat_interval=0,
        started_at=started,
    )
    response = await performance_stream_response(
        app,
        after="1",
        pid="42",
        started_at=started.isoformat(),
    )

    heartbeat = await anext(response.body_iterator)
    live_handle = sessions.begin(metadata("live"))
    live = await next_stream_json(response)

    assert heartbeat == "\n"
    assert live["type"] == "request_started"
    assert live["sequence"] == 2
    assert live["request"]["id"] == live_handle.request_id
    await response.body_iterator.aclose()


async def test_performance_stream_stale_cursor_gets_current_reset() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock, capacity=1)
    sessions.begin(metadata("first"))
    sessions.begin(metadata("second"))
    started = datetime(2026, 1, 2, 3, tzinfo=UTC)
    app = control_app_for_stream(sessions, started_at=started)

    response = await performance_stream_response(
        app,
        after="0",
        pid="42",
        started_at=started.isoformat(),
    )
    reset = await next_stream_json(response)

    assert reset["type"] == "reset"
    assert reset["sequence"] == 2
    assert reset["snapshot"]["cursor"] == 2
    await response.body_iterator.aclose()


@pytest.mark.parametrize(
    ("pid", "started_at"),
    [
        (None, None),
        ("41", "2026-01-02T03:00:00+00:00"),
        ("42", "2026-01-02T03:00:01+00:00"),
        (None, "2026-01-02T03:00:00+00:00"),
        ("42", None),
    ],
)
async def test_performance_stream_missing_or_mismatched_process_forces_reset(
    pid: str | None,
    started_at: str | None,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("existing"))
    app = control_app_for_stream(sessions)

    response = await performance_stream_response(
        app,
        after="0",
        pid=pid,
        started_at=started_at,
    )
    first = await next_stream_json(response)

    assert first["type"] == "reset"
    assert first["sequence"] == 1
    assert first["process"] == {
        "pid": 42,
        "started_at": "2026-01-02T03:00:00Z",
    }
    await response.body_iterator.aclose()


async def test_performance_stream_filters_initial_snapshot_without_raw_id() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    selected = sessions.begin(metadata("raw-selected"))
    sessions.begin(metadata("raw-other"))
    app = control_app_for_stream(sessions)

    response = await performance_stream_response(
        app,
        filters=["session_id=raw-selected"],
    )
    reset = await next_stream_json(response)
    serialized = json.dumps(reset)

    assert [
        item["session"]["id"] for item in reset["snapshot"]["sessions"]
    ] == [selected.public_id]
    assert "raw-selected" not in serialized
    assert "raw-other" not in serialized
    await response.body_iterator.aclose()


async def test_performance_stream_overflow_resets_and_continues_on_fresh_subscription() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock, subscriber_capacity=1)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    initial = await next_stream_json(response)
    assert initial["type"] == "reset"

    sessions.begin(metadata("first"))
    sessions.begin(metadata("second"))
    assert sessions.events.subscriber_count == 0
    reset = await next_stream_json(response)

    assert reset["type"] == "reset"
    assert reset["sequence"] == 2
    assert sessions.events.subscriber_count == 1
    continued = sessions.begin(metadata("continued"))
    event = await next_stream_json(response)
    assert event["sequence"] == 3
    assert event["request"]["id"] == continued.request_id
    assert sessions.events.subscriber_count == 1
    await response.body_iterator.aclose()
    assert sessions.events.subscriber_count == 0


async def test_performance_stream_heartbeat_is_blank_and_carries_no_cursor() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    app = control_app_for_stream(sessions, heartbeat_interval=0)
    response = await performance_stream_response(app)
    await next_stream_json(response)

    heartbeat = await anext(response.body_iterator)

    assert heartbeat == "\n"
    assert all(marker not in heartbeat for marker in ("sequence", "journal", "state"))
    await response.body_iterator.aclose()


async def test_performance_stream_cancelled_generator_closes_subscription_once(
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    close_calls = 0
    original_close = Subscription.close

    def counted_close(subscription) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(subscription)

    monkeypatch.setattr(Subscription, "close", counted_close)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    await next_stream_json(response)
    assert sessions.events.subscriber_count == 1

    with pytest.raises(asyncio.CancelledError):
        await response.body_iterator.athrow(asyncio.CancelledError())

    assert sessions.events.subscriber_count == 0
    assert close_calls == 1


async def test_performance_stream_send_failure_preserves_error_and_closes_once(
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    close_calls = 0
    original_close = Subscription.close

    def failing_close(subscription) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(subscription)
        raise RuntimeError("close failed")

    monkeypatch.setattr(Subscription, "close", failing_close)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    blocked = asyncio.Event()

    async def receive():
        await blocked.wait()
        return {"type": "http.disconnect"}

    async def failing_send(message):
        raise RuntimeError("send failed before body")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/v1/performance/events",
        "raw_path": b"/v1/performance/events",
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
        "root_path": "",
    }

    with pytest.raises(BaseException, match="send failed before body"):
        await response(scope, receive, failing_send)

    assert sessions.events.subscriber_count == 0
    assert close_calls == 1


@pytest.mark.parametrize(
    "after",
    ["true", "false", "-1", "+1", str(MAX_CONTROL_INTEGER + 1)],
)
async def test_performance_stream_rejects_invalid_cursor_with_generic_422(
    after: str,
) -> None:
    clock = RegistryClock()
    app = control_app_for_stream(stream_registry(clock))

    response = await request(app, "/v1/performance/events", {"after": after})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}
    assert len(response.content) < 100


@pytest.mark.parametrize(
    "started_at",
    ["0", "not-a-date", "2026-01-02T03:00:00"],
)
async def test_performance_stream_rejects_invalid_resume_time_with_generic_422(
    started_at: str,
) -> None:
    clock = RegistryClock()
    app = control_app_for_stream(stream_registry(clock))

    response = await request(
        app,
        "/v1/performance/events",
        {"after": "0", "pid": "42", "started_at": started_at},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}


async def test_performance_stream_rejects_future_cursor_before_subscribing() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    app = control_app_for_stream(sessions)

    response = await request(
        app,
        "/v1/performance/events",
        {
            "after": "1",
            "pid": "42",
            "started_at": "2026-01-02T03:00:00Z",
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}
    assert sessions.events.subscriber_count == 0


async def test_performance_stream_rejects_ambiguous_filter_before_subscribing() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    prefixes: dict[str, int] = {}
    collision = None
    for index in range(17):
        prefix = sessions.begin(metadata(f"raw-stream-{index}")).public_id[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
        if prefixes[prefix] == 2:
            collision = prefix
            break
    assert collision is not None
    app = control_app_for_stream(sessions)
    before = sessions.events.subscriber_count

    response = await request(
        app,
        "/v1/performance/events",
        [("filter", f"id={collision}")],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Session ID prefix is ambiguous"}
    assert sessions.events.subscriber_count == before


async def test_performance_stream_filters_live_events_server_side() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("raw-selected", provider="vertex"))
    sessions.begin(metadata("raw-other", provider="openai"))
    app = control_app_for_stream(sessions, heartbeat_interval=0)
    response = await performance_stream_response(
        app,
        filters=["provider=vertex"],
    )
    await next_stream_json(response)

    sessions.begin(metadata("raw-other", provider="openai"))
    skipped = await anext(response.body_iterator)
    selected = sessions.begin(metadata("raw-selected", provider="vertex"))
    event = await next_stream_json(response)

    assert skipped == "\n"
    assert event["session_id"] == selected.public_id
    assert event["request"]["id"] == selected.request_id
    assert "raw-selected" not in json.dumps(event)
    assert "raw-other" not in json.dumps(event)
    await response.body_iterator.aclose()


async def test_performance_stream_without_after_valid_identity_still_resets() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("existing"))
    app = control_app_for_stream(sessions)

    response = await performance_stream_response(
        app,
        pid="42",
        started_at="2026-01-02T03:00:00Z",
    )
    reset = await next_stream_json(response)

    assert reset["type"] == "reset"
    assert reset["sequence"] == 1
    await response.body_iterator.aclose()


@pytest.mark.parametrize(
    ("pid", "started_at"),
    [
        ("not-an-integer", "2026-01-02T03:00:00Z"),
        ("42", "not-a-date"),
    ],
)
async def test_performance_stream_without_after_rejects_malformed_identity(
    pid: str,
    started_at: str,
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)

    def unexpected_subscription(*args, **kwargs):
        pytest.fail("malformed identity reached subscription")

    monkeypatch.setattr(sessions, "subscribe_performance", unexpected_subscription)
    app = control_app_for_stream(sessions)

    with pytest.raises(HTTPException) as captured:
        await performance_stream_response(
            app,
            pid=pid,
            started_at=started_at,
        )

    assert captured.value.status_code == 422
    assert captured.value.detail == "Invalid performance event request"


@pytest.mark.parametrize(
    "started_at",
    [
        "0001-01-01T00:00:00+14:00",
        "9999-12-31T23:59:59-14:00",
    ],
)
async def test_performance_stream_rejects_utc_overflow_boundaries(
    started_at: str,
) -> None:
    clock = RegistryClock()
    app = control_app_for_stream(stream_registry(clock))
    transport = ASGITransport(app=app, raise_app_exceptions=False)

    async with AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.get(
            "/v1/performance/events",
            params={"after": "0", "pid": "42", "started_at": started_at},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}


@pytest.mark.parametrize("pid", ["true", "-1", "0", "+42", "9223372036854775808"])
async def test_performance_stream_rejects_invalid_resume_pid_with_generic_422(
    pid: str,
) -> None:
    clock = RegistryClock()
    app = control_app_for_stream(stream_registry(clock))

    response = await request(
        app,
        "/v1/performance/events",
        {
            "after": "0",
            "pid": pid,
            "started_at": "2026-01-02T03:00:00Z",
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}
