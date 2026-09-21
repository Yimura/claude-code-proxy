from datetime import UTC, datetime, timedelta

from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.event_journal import EventJournal
from claude_code_proxy.observability import SessionMetadata, SessionRegistry


class Clock:
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
    session_id: str | None = "sensitive-session",
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
    **changes,
) -> SessionMetadata:
    values = {
        "client_identity": ClientIdentity(
            session_id,
            agent_id,
            parent_agent_id,
        ),
        "client_model": "claude-opus",
        "upstream_model": "openai/gpt-5.6-sol",
        "provider": "openai",
        "transport": "codex",
        "effort": "high",
        "context_window": 1_000_000,
    }
    values.update(changes)
    return SessionMetadata(**values)


def registry(
    clock: Clock,
    inactive_limit: int = 10,
    events: EventJournal | None = None,
) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=inactive_limit,
        secret=b"test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
        events=events,
    )
