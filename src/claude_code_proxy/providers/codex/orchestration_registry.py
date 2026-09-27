"""Observed Codex agent lineage and bounded nesting authorization state."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import math
import threading
import time


MAX_LINEAGE_RECORDS_PER_SESSION = 1024


class ParentState(StrEnum):
    OBSERVED = "observed"
    ABSENT = "absent"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    CYCLIC = "cyclic"


class AuthorizationStatus(StrEnum):
    ABSENT = "absent"
    ACTIVE = "active"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class NestingAuthorization:
    session_id: str
    status: AuthorizationStatus
    max_depth: int | None
    authorized_at: float | None
    expires_at: float | None


@dataclass(frozen=True, slots=True)
class ObservedLineage:
    session_id: str
    agent_id: str
    parent_agent_id: str | None
    parent_state: ParentState
    depth: int | None
    first_observed_at: float
    last_observed_at: float


@dataclass(slots=True)
class _LineageRecord:
    parent_agent_id: str | None
    first_observed_at: float
    last_observed_at: float
    conflicting_parent: bool = False
    parent_state: ParentState = ParentState.UNKNOWN
    depth: int | None = None


class OrchestrationRegistry:
    """Retain immutable observations keyed only by opaque public identities."""

    def __init__(
        self, *, monotonic_clock: Callable[[], float] | None = None
    ) -> None:
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._lineages: dict[str, dict[str, _LineageRecord]] = {}
        self._authorizations: dict[str, NestingAuthorization] = {}
        self._generations: dict[str, int] = {}
        self._next_generation = 0
        self._lock = threading.Lock()

    def observe_lineage(
        self,
        session_id: str,
        agent_id: str | None,
        parent_agent_id: str | None,
    ) -> ObservedLineage | None:
        if agent_id is None:
            return None
        session = _identifier("session_id", session_id)
        agent = _identifier("agent_id", agent_id)
        parent = _optional_identifier("parent_agent_id", parent_agent_id)
        now = _finite_time(self._monotonic_clock())
        with self._lock:
            records = self._lineages.setdefault(session, {})
            record = records.get(agent)
            if record is None and len(records) >= MAX_LINEAGE_RECORDS_PER_SESSION:
                self._bump_generation_locked(session)
                return ObservedLineage(
                    session,
                    agent,
                    parent,
                    ParentState.UNKNOWN,
                    None,
                    now,
                    now,
                )
            if record is None:
                record = _LineageRecord(parent, now, now)
                records[agent] = record
            else:
                record.last_observed_at = now
                if record.parent_agent_id != parent:
                    record.conflicting_parent = True
            self._bump_generation_locked(session)
            self._resolve_session(records, now)
            return _snapshot(session, agent, record)

    def lineage(
        self, session_id: str, agent_id: str | None
    ) -> ObservedLineage | None:
        if agent_id is None:
            return None
        session = _identifier("session_id", session_id)
        agent = _identifier("agent_id", agent_id)
        with self._lock:
            record = self._lineages.get(session, {}).get(agent)
            if record is None:
                return None
            return _snapshot(session, agent, record)

    def authorize(
        self, session_id: str, *, max_depth: int, duration: int | float
    ) -> NestingAuthorization:
        session = _identifier("session_id", session_id)
        depth = _authorization_depth(max_depth)
        seconds = _authorization_duration(duration)
        now = _finite_time(self._monotonic_clock())
        authorization = NestingAuthorization(
            session,
            AuthorizationStatus.ACTIVE,
            depth,
            now,
            now + seconds,
        )
        with self._lock:
            self._authorizations[session] = authorization
            self._bump_generation_locked(session)
        return authorization

    def authorization(self, session_id: str) -> NestingAuthorization:
        session = _identifier("session_id", session_id)
        now = _finite_time(self._monotonic_clock())
        with self._lock:
            authorization = self._authorizations.get(session)
            if authorization is None:
                return NestingAuthorization(
                    session, AuthorizationStatus.ABSENT, None, None, None
                )
            if authorization.status is AuthorizationStatus.EXPIRED:
                return authorization
            assert authorization.expires_at is not None
            if now < authorization.expires_at:
                return authorization
            expired = NestingAuthorization(
                session,
                AuthorizationStatus.EXPIRED,
                authorization.max_depth,
                authorization.authorized_at,
                authorization.expires_at,
            )
            self._authorizations[session] = expired
            self._bump_generation_locked(session)
            return expired

    def revoke(self, session_id: str) -> bool:
        session = _identifier("session_id", session_id)
        with self._lock:
            removed = self._authorizations.pop(session, None) is not None
            if removed:
                self._bump_generation_locked(session)
            return removed

    def session_generation(self, session_id: str) -> int:
        session = _identifier("session_id", session_id)
        with self._lock:
            return self._generations.get(session, 0)

    def remove_session_if_generation(
        self, session_id: str, generation: int
    ) -> bool:
        session = _identifier("session_id", session_id)
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        with self._lock:
            if self._generations.get(session) != generation:
                return False
            self._remove_session_locked(session)
            return True

    def remove_session(self, session_id: str) -> None:
        session = _identifier("session_id", session_id)
        with self._lock:
            self._remove_session_locked(session)

    def _remove_session_locked(self, session_id: str) -> None:
        self._lineages.pop(session_id, None)
        self._authorizations.pop(session_id, None)
        self._generations.pop(session_id, None)

    def _bump_generation_locked(self, session_id: str) -> int:
        self._next_generation += 1
        self._generations[session_id] = self._next_generation
        return self._next_generation

    @staticmethod
    def _resolve_session(
        records: dict[str, _LineageRecord], now: float
    ) -> None:
        resolved: dict[str, tuple[ParentState, int | None]] = {}
        visit_state: dict[str, int] = {}
        for start in records:
            if start in resolved:
                continue
            path: list[str] = []
            current = start
            while current not in resolved:
                if visit_state.get(current) == 1:
                    for agent_id in path:
                        resolved[agent_id] = (ParentState.CYCLIC, None)
                        visit_state[agent_id] = 2
                    path.clear()
                    break
                visit_state[current] = 1
                path.append(current)
                record = records[current]
                terminal = _terminal_resolution(record, records)
                if terminal is not None:
                    resolved[current] = terminal
                    visit_state[current] = 2
                    path.pop()
                    break
                assert record.parent_agent_id is not None
                current = record.parent_agent_id

            while path:
                agent_id = path.pop()
                if agent_id in resolved:
                    continue
                parent_id = records[agent_id].parent_agent_id
                assert parent_id is not None
                parent_state, parent_depth = resolved[parent_id]
                resolved[agent_id] = _child_resolution(
                    parent_state, parent_depth
                )
                visit_state[agent_id] = 2

        for agent_id, record in records.items():
            state, depth = resolved[agent_id]
            if state != record.parent_state or depth != record.depth:
                record.last_observed_at = now
            record.parent_state = state
            record.depth = depth


def _terminal_resolution(
    record: _LineageRecord,
    records: dict[str, _LineageRecord],
) -> tuple[ParentState, int | None] | None:
    if record.conflicting_parent:
        return ParentState.AMBIGUOUS, None
    if record.parent_agent_id is None:
        return ParentState.ABSENT, 1
    if record.parent_agent_id not in records:
        return ParentState.UNKNOWN, None
    return None


def _child_resolution(
    parent_state: ParentState, parent_depth: int | None
) -> tuple[ParentState, int | None]:
    if parent_state in {ParentState.ABSENT, ParentState.OBSERVED}:
        assert parent_depth is not None
        return ParentState.OBSERVED, parent_depth + 1
    if parent_state is ParentState.CYCLIC:
        return ParentState.CYCLIC, None
    if parent_state is ParentState.AMBIGUOUS:
        return ParentState.AMBIGUOUS, None
    return ParentState.UNKNOWN, None


def _snapshot(
    session_id: str, agent_id: str, record: _LineageRecord
) -> ObservedLineage:
    return ObservedLineage(
        session_id=session_id,
        agent_id=agent_id,
        parent_agent_id=record.parent_agent_id,
        parent_state=record.parent_state,
        depth=record.depth,
        first_observed_at=record.first_observed_at,
        last_observed_at=record.last_observed_at,
    )


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or not value.isprintable():
        raise ValueError(f"{name} must be a nonblank printable string")
    return value


def _optional_identifier(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _identifier(name, value)


def _authorization_depth(value: object) -> int:
    if type(value) is not int or not 2 <= value <= 8:
        raise ValueError("max_depth must be an integer between 2 and 8")
    return value


def _authorization_duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration must be finite and between 1 and 86400 seconds")
    seconds = float(value)
    if not math.isfinite(seconds) or not 1 <= seconds <= 86400:
        raise ValueError("duration must be finite and between 1 and 86400 seconds")
    return seconds


def _finite_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("monotonic clock must be finite")
    sampled = float(value)
    if not math.isfinite(sampled):
        raise ValueError("monotonic clock must be finite")
    return sampled
