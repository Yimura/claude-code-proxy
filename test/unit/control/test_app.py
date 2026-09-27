import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

import claude_code_proxy.control.app as control_app_module
from claude_code_proxy.control.app import create_control_app as _create_control_app
from claude_code_proxy.control.schemas import (
    HealthResponse,
    SessionCounts,
    SessionListResponse,
    SessionResponse,
)
from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.observability import (
    SessionMetadata,
    SessionRegistry,
    SessionSnapshot,
)
from test.unit.control.app_test_support import (
    create_test_control_app as create_control_app,
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
        "orchestration_authorizations",
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
            "orchestration_authorizations",
        ],
        "orchestration_mode": "advisory",
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


def test_orchestration_control_schemas_are_strict_frozen_and_bounded() -> None:
    from claude_code_proxy.control.schemas import (
        OrchestrationAuthorizationRequest,
        OrchestrationAuthorizationResponse,
        OrchestrationRevocationRequest,
    )

    request = OrchestrationAuthorizationRequest(
        session_id="raw/session",
        max_depth=2,
        duration_seconds=1,
    )
    assert OrchestrationAuthorizationRequest(
        session_id="x" * 256,
        max_depth=2,
        duration_seconds=1,
    ).session_id == "x" * 256
    with pytest.raises(ValidationError):
        OrchestrationAuthorizationRequest(
            session_id="x" * 257,
            max_depth=2,
            duration_seconds=1,
        )
    revocation = OrchestrationRevocationRequest(session_id="../raw-session")
    assert OrchestrationRevocationRequest(session_id="x" * 256).session_id == "x" * 256
    with pytest.raises(ValidationError):
        OrchestrationRevocationRequest(session_id="x" * 257)
    response = OrchestrationAuthorizationResponse(
        session_id="a" * 64,
        max_depth=8,
        remaining_seconds=0.0,
    )
    with pytest.raises(ValidationError, match="frozen"):
        request.max_depth = 3
    with pytest.raises(ValidationError, match="frozen"):
        response.max_depth = 3
    with pytest.raises(ValidationError, match="frozen"):
        revocation.session_id = "other"
    for field, value in (
        ("max_depth", True),
        ("max_depth", 2.0),
        ("max_depth", "2"),
        ("duration_seconds", True),
        ("duration_seconds", 1.0),
        ("duration_seconds", "1"),
    ):
        payload = {
            "session_id": "raw-session",
            "max_depth": 2,
            "duration_seconds": 1,
            field: value,
        }
        with pytest.raises(ValidationError):
            OrchestrationAuthorizationRequest.model_validate(payload)
    with pytest.raises(ValidationError):
        OrchestrationAuthorizationRequest.model_validate(
            {
                "session_id": "raw-session",
                "max_depth": 2,
                "duration_seconds": 1,
                "extra": "rejected",
            }
        )
    with pytest.raises(ValidationError):
        OrchestrationRevocationRequest.model_validate(
            {"session_id": "raw-session", "extra": "rejected"}
        )
    for remaining in (-1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            OrchestrationAuthorizationResponse(
                session_id="a" * 64,
                max_depth=2,
                remaining_seconds=remaining,
            )


async def test_private_orchestration_routes_hash_raw_session_and_list_active_rows() -> None:
    from claude_code_proxy.config import CodexOrchestrationMode
    from claude_code_proxy.public_identity import PublicIdentity
    from claude_code_proxy.providers.codex.orchestration_registry import OrchestrationRegistry

    identity = PublicIdentity(secret=b"private-control")
    orchestration = OrchestrationRegistry(identity=identity, monotonic_clock=lambda: 100.0)
    app = create_control_app(
        registry(RegistryClock()),
        application_version="1.0",
        pid=1,
        orchestration_registry=orchestration,
        orchestration_mode=CodexOrchestrationMode.ENFORCE,
    )
    raw_ids = (
        "sensitive-raw-session",
        "folder/child",
        ".",
        "..",
        "./nested",
        "../sibling",
    )
    endpoint = "/v1/orchestration/authorizations"
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        for raw in raw_ids:
            allowed = await client.put(
                endpoint,
                json={
                    "session_id": raw,
                    "max_depth": 4,
                    "duration_seconds": 60,
                },
            )
            listed = await client.get(endpoint)
            revoked = await client.request(
                "DELETE", endpoint, json={"session_id": raw}
            )
            revoked_again = await client.request(
                "DELETE", endpoint, json={"session_id": raw}
            )

            public_id = identity.public_id(raw)
            assert allowed.status_code == 200
            assert allowed.json() == {
                "session_id": public_id,
                "max_depth": 4,
                "remaining_seconds": 60.0,
            }
            assert listed.json() == {"authorizations": [allowed.json()]}
            assert json.dumps(raw) not in allowed.text + listed.text
            assert revoked.status_code == revoked_again.status_code == 204
            assert revoked.content == revoked_again.content == b""


async def test_private_health_reports_orchestration_mode_and_capability() -> None:
    from claude_code_proxy.config import CodexOrchestrationMode

    app = create_control_app(
        registry(RegistryClock()),
        application_version="1.0",
        pid=1,
        orchestration_mode=CodexOrchestrationMode.ADVISORY,
    )
    response = await request(app, "/v1/health")

    assert response.json()["orchestration_mode"] == "advisory"
    assert "orchestration_authorizations" in response.json()["capabilities"]
    assert response.json()["protocol_version"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "raw-secret", "max_depth": True, "duration_seconds": 60},
        {"session_id": "raw-secret", "max_depth": 2.0, "duration_seconds": 60},
        {"session_id": "raw-secret", "max_depth": "2", "duration_seconds": 60},
        {"session_id": "raw-secret", "max_depth": 2, "duration_seconds": True},
        {"session_id": "raw-secret", "max_depth": 2, "duration_seconds": 60.0},
        {"session_id": "raw-secret", "max_depth": 2, "duration_seconds": "60"},
        {
            "session_id": "raw-secret",
            "max_depth": 2,
            "duration_seconds": 60,
            "raw": "secret-marker",
        },
    ],
)
async def test_private_authorization_validation_error_is_generic(payload) -> None:
    app = create_control_app(registry(RegistryClock()), application_version="1.0", pid=1)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://control") as client:
        response = await client.put(
            "/v1/orchestration/authorizations",
            json=payload,
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid control request"}
    assert "raw-secret" not in response.text
    assert "secret-marker" not in response.text


def test_control_app_requires_shared_orchestration_dependencies() -> None:
    with pytest.raises(TypeError, match="orchestration_registry"):
        _create_control_app(
            registry(RegistryClock()),
            application_version="1.0",
            pid=1,
        )


async def test_authorization_body_accepts_exact_byte_ceiling_and_rejects_one_over() -> None:
    from claude_code_proxy.control.app import MAX_AUTHORIZATION_REQUEST_BYTES

    app = create_control_app(
        registry(RegistryClock()),
        application_version="1.0",
        pid=1,
    )
    payload = json.dumps({
        "session_id": "bounded-session",
        "max_depth": 2,
        "duration_seconds": 60,
    }).encode()
    exact = payload + b" " * (MAX_AUTHORIZATION_REQUEST_BYTES - len(payload))
    over = exact + b" "

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        accepted = await client.put(
            "/v1/orchestration/authorizations",
            content=exact,
            headers={"content-type": "application/json"},
        )
        rejected = await client.put(
            "/v1/orchestration/authorizations",
            content=over,
            headers={"content-type": "application/json"},
        )

    assert len(exact) == MAX_AUTHORIZATION_REQUEST_BYTES
    assert accepted.status_code == 200
    assert rejected.status_code == 413
    assert rejected.json() == {"detail": "Authorization request body too large"}


async def test_authorization_body_ceiling_rejects_huge_and_chunked_bodies_privately() -> None:
    from claude_code_proxy.control.app import MAX_AUTHORIZATION_REQUEST_BYTES

    app = create_control_app(
        registry(RegistryClock()),
        application_version="1.0",
        pid=1,
    )
    marker = "RAW_BODY_PRIVACY_MARKER"
    huge = json.dumps({"session_id": marker + "x" * 1_000_000}).encode()

    async def chunks():
        yield b'{"session_id":"chunked-' + marker.encode()
        yield b"x" * MAX_AUTHORIZATION_REQUEST_BYTES
        yield b'"}'

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        content_length = await client.put(
            "/v1/orchestration/authorizations",
            content=huge,
            headers={"content-type": "application/json"},
        )
        chunked = await client.put(
            "/v1/orchestration/authorizations",
            content=chunks(),
            headers={"content-type": "application/json"},
        )

    assert content_length.status_code == chunked.status_code == 413
    exposed = content_length.text + chunked.text
    assert marker not in exposed
    assert content_length.json() == chunked.json() == {
        "detail": "Authorization request body too large"
    }


async def test_authorization_capacity_error_is_fixed_and_does_not_echo_session() -> None:
    from claude_code_proxy.config import CodexOrchestrationMode
    from claude_code_proxy.public_identity import PublicIdentity
    from claude_code_proxy.providers.codex.orchestration_registry import OrchestrationRegistry

    orchestration = OrchestrationRegistry(
        identity=PublicIdentity(secret=b"capacity-test"),
        authorization_capacity=1,
    )
    app = create_control_app(
        registry(RegistryClock()),
        application_version="1.0",
        pid=1,
        orchestration_registry=orchestration,
        orchestration_mode=CodexOrchestrationMode.ENFORCE,
    )
    endpoint = "/v1/orchestration/authorizations"
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        first = await client.put(
            endpoint,
            json={
                "session_id": "first-session",
                "max_depth": 2,
                "duration_seconds": 60,
            },
        )
        second = await client.put(
            endpoint,
            json={
                "session_id": "CAPACITY_RAW_SESSION_MARKER",
                "max_depth": 2,
                "duration_seconds": 60,
            },
        )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {
        "detail": "Orchestration authorization capacity reached"
    }
    assert "CAPACITY_RAW_SESSION_MARKER" not in second.text
