from datetime import UTC, datetime, timedelta
import json

from fastapi import HTTPException
import pytest

from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.event_journal import EventJournal
from claude_code_proxy.observability import SessionMetadata, SessionRegistry


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


def metadata(client_id: str, *, provider: str = "openai") -> SessionMetadata:
    return SessionMetadata(
        client_identity=ClientIdentity(client_id),
        client_model="claude-opus",
        upstream_model=f"{provider}/provider-model",
        provider=provider,
        transport="codex",
        effort="high",
        context_window=1_000_000,
    )


def registry(
    clock: RegistryClock,
    *,
    inactive_limit: int = 10,
    subscriber_capacity: int = 64,
) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=inactive_limit,
        secret=b"control-test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
        events=EventJournal(subscriber_capacity=subscriber_capacity),
    )


def colliding_raw_id(
    sessions: SessionRegistry,
    prefix: str,
    excluded_public_id: str,
) -> str:
    for index in range(100):
        candidate = f"late-collision-{index}"
        public_id = sessions.public_id(candidate)
        if public_id != excluded_public_id and public_id.startswith(prefix):
            return candidate
    raise AssertionError("failed to find deterministic public ID collision")


def control_app(sessions: SessionRegistry):
    return create_control_app(
        sessions,
        started_at=datetime(2026, 1, 2, 3, tzinfo=UTC),
        application_version="1.0",
        pid=42,
    )


async def stream_response(
    app,
    *,
    filters: list[str],
    after: str | None = None,
):
    route = next(route for route in app.routes if route.path == "/v1/performance/events")
    return await route.endpoint(
        filter=filters,
        after=None if after is None else [after],
        pid=None if after is None else ["42"],
        started_at=None if after is None else ["2026-01-02T03:00:00Z"],
    )


async def next_json(response) -> dict[str, object]:
    frame = await anext(response.body_iterator)
    assert isinstance(frame, str)
    return json.loads(frame)


async def test_subscription_exposes_immutable_exact_event_filters() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    selected = sessions.begin(metadata("selected"))

    state = sessions.subscribe_performance(
        {"id": (selected.public_id[:8],)},
        after=None,
    )

    assert state.event_filters == {"id": (selected.public_id,)}
    with pytest.raises(TypeError):
        state.event_filters["id"] = ("other",)
    state.subscription.close()


async def test_unique_id_prefix_is_frozen_against_late_collision() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    selected = sessions.begin(metadata("selected"))
    prefix = selected.public_id[0]
    late_raw_id = colliding_raw_id(sessions, prefix, selected.public_id)
    response = await stream_response(control_app(sessions), filters=[f"id={prefix}"])
    await next_json(response)

    sessions.begin(metadata(late_raw_id))
    matching = sessions.begin(metadata("selected"))
    cursor = await next_json(response)
    event = await next_json(response)

    assert cursor["type"] == "cursor"
    assert cursor["sequence"] == 2
    assert event["session_id"] == matching.public_id
    await response.body_iterator.aclose()


async def test_replay_only_evicted_id_prefix_resolves_and_matches() -> None:
    clock = RegistryClock()
    sessions = registry(clock, inactive_limit=0)
    selected = sessions.begin(metadata("evicted"))
    sessions.finish(selected, "completed")
    assert sessions.snapshots() == []

    response = await stream_response(
        control_app(sessions),
        filters=[f"id={selected.public_id[:8]}"],
        after="0",
    )
    first = await next_json(response)
    second = await next_json(response)

    assert [first["sequence"], second["sequence"]] == [1, 2]
    assert first["session_id"] == second["session_id"] == selected.public_id
    await response.body_iterator.aclose()


async def test_unmatched_id_prefix_remains_stable_match_none() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    future_raw_id = "future-session"
    prefix = sessions.public_id(future_raw_id)[:8]
    response = await stream_response(
        control_app(sessions),
        filters=[f"id={prefix}"],
    )
    await next_json(response)

    future = sessions.begin(metadata(future_raw_id))
    cursor = await next_json(response)

    assert cursor["type"] == "cursor"
    assert cursor["sequence"] == 1
    assert "session_id" not in cursor
    assert future.public_id not in json.dumps(cursor)
    await response.body_iterator.aclose()


async def test_id_prefix_ambiguity_across_current_and_replay_is_leak_free() -> None:
    clock = RegistryClock()
    sessions = registry(clock, inactive_limit=0)
    replay_only = sessions.begin(metadata("replay-only"))
    sessions.finish(replay_only, "completed")
    raw_current = colliding_raw_id(
        sessions,
        replay_only.public_id[0],
        replay_only.public_id,
    )
    sessions.begin(metadata(raw_current))
    before = sessions.events.subscriber_count

    with pytest.raises(HTTPException) as captured:
        await stream_response(
            control_app(sessions),
            filters=[f"id={replay_only.public_id[0]}"],
            after="0",
        )

    assert captured.value.status_code == 422
    assert captured.value.detail == "Session ID prefix is ambiguous"
    assert sessions.events.subscriber_count == before


async def test_overflow_resubscription_keeps_frozen_id_selection() -> None:
    clock = RegistryClock()
    sessions = registry(clock, subscriber_capacity=1)
    selected = sessions.begin(metadata("selected"))
    prefix = selected.public_id[0]
    late_raw_id = colliding_raw_id(sessions, prefix, selected.public_id)
    response = await stream_response(control_app(sessions), filters=[f"id={prefix}"])
    await next_json(response)

    sessions.begin(metadata(late_raw_id))
    sessions.begin(metadata("overflow-trigger"))
    reset = await next_json(response)
    assert reset["type"] == "reset"
    assert [item["session"]["id"] for item in reset["snapshot"]["sessions"]] == [
        selected.public_id
    ]

    late = sessions.begin(metadata(late_raw_id))
    cursor = await next_json(response)
    matching = sessions.begin(metadata("selected"))
    event = await next_json(response)

    assert cursor["type"] == "cursor"
    assert cursor["sequence"] < event["sequence"]
    assert event["session_id"] == matching.public_id
    assert late.public_id != event["session_id"]
    await response.body_iterator.aclose()
