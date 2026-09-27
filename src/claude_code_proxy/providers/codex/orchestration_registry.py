"""Observed Codex agent lineage and bounded nesting authorization state."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import math
import threading
import time


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
            if record is None:
                record = _LineageRecord(parent, now, now)
                records[agent] = record
            else:
                record.last_observed_at = now
                if record.parent_agent_id != parent:
                    record.conflicting_parent = True
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
            return expired

    def revoke(self, session_id: str) -> bool:
        session = _identifier("session_id", session_id)
        with self._lock:
            return self._authorizations.pop(session, None) is not None

    def remove_session(self, session_id: str) -> None:
        session = _identifier("session_id", session_id)
        with self._lock:
            self._lineages.pop(session, None)
            self._authorizations.pop(session, None)

    @staticmethod
    def _resolve_session(
        records: dict[str, _LineageRecord], now: float
    ) -> None:
        memo: dict[str, tuple[ParentState, int | None]] = {}

        def resolve(
            agent_id: str, path: frozenset[str]
        ) -> tuple[ParentState, int | None]:
            cached = memo.get(agent_id)
            if cached is not None:
                return cached
            record = records[agent_id]
            if record.conflicting_parent:
                result = (ParentState.AMBIGUOUS, None)
            elif record.parent_agent_id is None:
                result = (ParentState.ABSENT, 1)
            elif agent_id in path:
                result = (ParentState.CYCLIC, None)
            elif record.parent_agent_id not in records:
                result = (ParentState.UNKNOWN, None)
            else:
                parent_state, parent_depth = resolve(
                    record.parent_agent_id, path | {agent_id}
                )
                result = _child_resolution(parent_state, parent_depth)
            memo[agent_id] = result
            return result

        for agent_id, record in records.items():
            state, depth = resolve(agent_id, frozenset())
            if state != record.parent_state or depth != record.depth:
                record.last_observed_at = now
            record.parent_state = state
            record.depth = depth


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
