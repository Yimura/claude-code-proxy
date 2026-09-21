from datetime import UTC, datetime, timedelta

import pytest

import claude_code_proxy.observability as observability_module
from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.event_journal import EventJournal
from claude_code_proxy.observability import (
    SessionMetadata,
    SessionRegistry,
)


class Clock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 1, 1, tzinfo=UTC)
        self.monotonic = 100.0

    def wall_now(self) -> datetime:
        return self.wall

    def monotonic_now(self) -> float:
        return self.monotonic

    def advance(self) -> None:
        self.wall += timedelta(seconds=1)
        self.monotonic += 1


def metadata(
    session_id: str,
    agent_id: str | None = None,
    *,
    provider: str = "openai",
) -> SessionMetadata:
    return SessionMetadata(
        client_identity=ClientIdentity(session_id, agent_id),
        client_model="claude-opus",
        upstream_model="openai/gpt-5.6-sol",
        provider=provider,
        transport="codex",
        effort="high",
        context_window=1_000_000,
    )


def registry(clock: Clock, *, limit: int = 10) -> SessionRegistry:
    return SessionRegistry(
        limit,
        secret=b"gate-test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
        events=EventJournal(),
        performance_enabled=False,
    )


def test_direct_registry_default_keeps_performance_enabled() -> None:
    assert SessionRegistry(1).performance_enabled is True


@pytest.mark.parametrize("value", [None, 0, 1, "true", object()])
def test_registry_requires_strict_performance_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="performance_enabled"):
        SessionRegistry(1, performance_enabled=value)  # type: ignore[arg-type]


def test_disabled_begin_does_not_allocate_performance_or_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    sessions = registry(clock)

    def fail_allocation(*args: object, **kwargs: object) -> None:
        pytest.fail("RequestPerformance was allocated")

    monkeypatch.setattr(observability_module, "RequestPerformance", fail_allocation)
    handle = sessions.begin(metadata("session"), operation="count_tokens")

    assert handle.operation == "count_tokens"
    assert handle.started_monotonic == 100.0
    assert sessions.events.current_sequence == 0
    assert next(iter(sessions._records.values())).performance is None


def test_disabled_registry_tracks_overlap_agents_filters_and_finish() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata("session", "agent-a"))
    clock.advance()
    second = sessions.begin(metadata("session", "agent-b"))

    active = sessions.snapshots({"provider": ["OPENAI"]})[0]
    assert active.active_requests == 2
    assert active.requests == 2
    assert {agent.requests for agent in active.agents} == {1}
    assert {agent.active_requests for agent in active.agents} == {1}

    clock.advance()
    assert sessions.finish(first, "completed") is None
    partial = sessions.snapshots()[0]
    assert partial.active_requests == 1
    assert partial.last_result == "completed"

    clock.advance()
    assert sessions.finish(second, "failed") is None
    terminal = sessions.snapshots({"state": ["FAILED"]})[0]
    assert terminal.active_requests == 0
    assert terminal.requests == 2
    assert terminal.last_result == "failed"
    assert sessions.counts() == (0, 1)
    assert sessions.events.current_sequence == 0


def test_disabled_registry_evicts_inactive_rows() -> None:
    clock = Clock()
    sessions = registry(clock, limit=1)
    first = sessions.begin(metadata("first"))
    sessions.finish(first, "completed")
    clock.advance()
    second = sessions.begin(metadata("second"))
    sessions.finish(second, "completed")

    snapshots = sessions.snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].id == second.public_id


def test_disabled_performance_interfaces_are_unavailable() -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata("session"))

    with pytest.raises(observability_module.PerformanceUnavailable):
        sessions.observer(handle)
    with pytest.raises(observability_module.PerformanceUnavailable):
        sessions.performance_snapshots()
    with pytest.raises(observability_module.PerformanceUnavailable):
        sessions.subscribe_performance(None, None)


def test_registry_logging_policy_defaults_to_collection_policy() -> None:
    assert SessionRegistry(1).performance_logging_enabled is True
    assert SessionRegistry(
        1, performance_enabled=False
    ).performance_logging_enabled is False


def test_registry_accepts_collector_without_terminal_logging() -> None:
    sessions = SessionRegistry(
        1,
        performance_enabled=True,
        performance_logging_enabled=False,
    )

    assert sessions.performance_enabled is True
    assert sessions.performance_logging_enabled is False


@pytest.mark.parametrize("value", [0, 1, "true", object()])
def test_registry_requires_strict_logging_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="performance_logging_enabled"):
        SessionRegistry(
            1,
            performance_logging_enabled=value,  # type: ignore[arg-type]
        )


def test_registry_rejects_logging_without_collection() -> None:
    with pytest.raises(ValueError, match="requires performance collection"):
        SessionRegistry(
            1,
            performance_enabled=False,
            performance_logging_enabled=True,
        )
