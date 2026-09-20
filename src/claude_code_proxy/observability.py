"""Thread-safe, privacy-preserving runtime session observations."""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import hmac
import secrets
import threading
import time
from typing import Literal
import uuid

from .domain.models import ClientIdentity
from .limits import MAX_CONTROL_INTEGER
from .text_safety import scalar_text

SessionState = Literal["active", "idle", "failed"]
SessionResult = Literal["completed", "failed"]
SessionFilters = Mapping[str, Sequence[str]]

_FILTER_FIELDS = frozenset(
    {
        "id",
        "session_id",
        "state",
        "provider",
        "transport",
        "model",
        "effort",
    }
)


@dataclass(frozen=True)
class SessionMetadata:
    client_identity: ClientIdentity = field(repr=False)
    client_model: str
    upstream_model: str
    provider: str
    transport: str
    effort: str
    context_window: int | None

    def __post_init__(self) -> None:
        for name in (
            "client_model",
            "upstream_model",
            "provider",
            "transport",
            "effort",
        ):
            object.__setattr__(self, name, scalar_text(getattr(self, name)))


@dataclass(frozen=True)
class ObservationHandle:
    key: str
    request_id: str
    public_id: str
    started_monotonic: float
    is_new: bool
    request_scoped: bool
    agent_key: str | None = None
    agent_public_id: str | None = None
    parent_agent_public_id: str | None = None
    agent_is_new: bool = False


@dataclass(frozen=True)
class AgentSnapshot:
    id: str
    parent_id: str | None
    state: SessionState
    active_requests: int
    requests: int
    client_model: str
    model: str
    provider: str
    transport: str
    effort: str
    context_window: int | None
    first_seen: datetime
    last_seen: datetime
    elapsed_seconds: float
    last_result: SessionResult | None


@dataclass(frozen=True)
class SessionSnapshot:
    id: str
    state: SessionState
    active_requests: int
    requests: int
    client_model: str
    model: str
    provider: str
    transport: str
    effort: str
    context_window: int | None
    first_seen: datetime
    last_seen: datetime
    elapsed_seconds: float
    last_result: SessionResult | None
    agents: tuple[AgentSnapshot, ...] = ()


class InvalidSessionFilter(ValueError):
    """Raised when a session snapshot filter is malformed."""


class AmbiguousSessionId(ValueError):
    """Raised when a public ID prefix identifies multiple sessions."""


@dataclass(frozen=True)
class _SnapshotMetadata:
    client_model: str
    model: str
    provider: str
    transport: str
    effort: str
    context_window: int | None

    @classmethod
    def from_session(cls, metadata: SessionMetadata) -> "_SnapshotMetadata":
        return cls(
            client_model=metadata.client_model,
            model=_strip_model_prefix(metadata.upstream_model),
            provider=metadata.provider,
            transport=metadata.transport,
            effort=metadata.effort,
            context_window=metadata.context_window,
        )


@dataclass
class _ActivityRecord:
    public_id: str
    metadata: _SnapshotMetadata
    first_seen: datetime
    last_seen: datetime
    requests: int
    active: dict[str, float]
    latest_duration: float = 0.0
    last_result: SessionResult | None = None


@dataclass
class _AgentRecord(_ActivityRecord):
    parent_public_id: str | None = None


@dataclass
class _SessionRecord(_ActivityRecord):
    agents: dict[str, _AgentRecord] = field(default_factory=dict)


class SessionRegistry:
    """Track request lifecycle by logical session without exposing client IDs."""

    def __init__(
        self,
        inactive_limit: int,
        secret: bytes | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if (
            type(inactive_limit) is not int
            or not 0 <= inactive_limit <= MAX_CONTROL_INTEGER
        ):
            raise ValueError(
                "inactive_limit must be an integer between 0 and "
                f"{MAX_CONTROL_INTEGER}"
            )
        self._inactive_limit = inactive_limit
        self._secret = secrets.token_bytes(32) if secret is None else secret
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._records: dict[str, _SessionRecord] = {}
        self._inactive: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def inactive_limit(self) -> int:
        return self._inactive_limit

    def public_id(self, identifier: str) -> str:
        """Return this process's stable opaque ID for an identifier."""
        normalized = identifier.strip()
        return hmac.new(
            self._secret,
            normalized.encode(errors="surrogatepass"),
            hashlib.sha256,
        ).hexdigest()

    def public_agent_id(self, root_identifier: str, agent_id: str) -> str:
        normalized_root = root_identifier.strip()
        normalized_agent = agent_id.strip()
        return self.public_id(f"agent:{normalized_root}\0{normalized_agent}")

    def _agent_identity(
        self,
        root_identifier: str,
        identity: ClientIdentity,
    ) -> tuple[str, str, str | None] | None:
        agent_id = (identity.agent_id or "").strip()
        if not agent_id:
            return None
        public_id = self.public_agent_id(root_identifier, agent_id)
        parent_id = (identity.parent_agent_id or "").strip()
        parent_public_id = (
            self.public_agent_id(root_identifier, parent_id)
            if parent_id
            else None
        )
        return f"agent:{public_id}", public_id, parent_public_id

    @staticmethod
    def _begin_agent(
        session: _SessionRecord,
        identity: tuple[str, str, str | None] | None,
        metadata: _SnapshotMetadata,
        request_id: str,
        started: float,
        seen_at: datetime,
    ) -> tuple[str | None, str | None, str | None, bool]:
        if identity is None:
            return None, None, None, False
        key, public_id, parent_public_id = identity
        agent = session.agents.get(key)
        is_new = agent is None
        if agent is None:
            agent = _AgentRecord(
                public_id=public_id,
                parent_public_id=parent_public_id,
                metadata=metadata,
                first_seen=seen_at,
                last_seen=seen_at,
                requests=0,
                active={},
            )
            session.agents[key] = agent
        agent.parent_public_id = parent_public_id
        _begin_activity(agent, metadata, request_id, started, seen_at)
        return key, public_id, parent_public_id, is_new

    def begin(self, metadata: SessionMetadata) -> ObservationHandle:
        identity = metadata.client_identity
        normalized_id = (identity.session_id or "").strip()
        request_id = uuid.uuid4().hex
        request_scoped = not normalized_id
        root_identifier = (
            f"request:{request_id}" if request_scoped else normalized_id
        )
        public_id = self.public_id(root_identifier)
        key = (
            f"request:{request_id}"
            if request_scoped
            else f"session:{public_id}"
        )
        snapshot_metadata = _SnapshotMetadata.from_session(metadata)
        agent_identity = self._agent_identity(root_identifier, identity)

        with self._lock:
            started = self._monotonic_clock()
            seen_at = self._wall_clock()
            record = self._records.get(key)
            is_new = record is None
            if record is None:
                record = _SessionRecord(
                    public_id=public_id,
                    metadata=snapshot_metadata,
                    first_seen=seen_at,
                    last_seen=seen_at,
                    requests=0,
                    active={},
                )
                self._records[key] = record
            _begin_activity(
                record,
                snapshot_metadata,
                request_id,
                started,
                seen_at,
            )
            self._inactive.pop(key, None)
            agent_key, agent_public_id, parent_public_id, agent_is_new = (
                self._begin_agent(
                    record,
                    agent_identity,
                    snapshot_metadata,
                    request_id,
                    started,
                    seen_at,
                )
            )

        return ObservationHandle(
            key=key,
            request_id=request_id,
            public_id=public_id,
            started_monotonic=started,
            is_new=is_new,
            request_scoped=request_scoped,
            agent_key=agent_key,
            agent_public_id=agent_public_id,
            parent_agent_public_id=parent_public_id,
            agent_is_new=agent_is_new,
        )

    def finish(self, handle: ObservationHandle, result: SessionResult) -> None:
        """Finish one request; repeated or unknown handles are ignored."""
        with self._lock:
            record = self._records.get(handle.key)
            if record is None:
                return
            finished_at = self._monotonic_clock()
            seen_at = self._wall_clock()
            if not _finish_activity(
                record,
                handle.request_id,
                finished_at,
                seen_at,
                result,
            ):
                return
            if handle.agent_key is not None:
                agent = record.agents.get(handle.agent_key)
                if agent is not None:
                    _finish_activity(
                        agent,
                        handle.request_id,
                        finished_at,
                        seen_at,
                        result,
                    )
            if not record.active:
                self._inactive[handle.key] = None
                self._inactive.move_to_end(handle.key)
                self._evict_inactive()

    def snapshots(
        self,
        filters: SessionFilters | None = None,
    ) -> list[SessionSnapshot]:
        """Copy snapshots and optionally filter their public fields."""
        normalized_filters = _validate_filters(filters)
        with self._lock:
            now = self._monotonic_clock()
            snapshots = [
                _to_snapshot(record, now) for record in self._records.values()
            ]

        if normalized_filters is not None:
            session_ids = normalized_filters.get("session_id")
            exact_ids = (
                {self.public_id(value) for value in session_ids}
                if session_ids is not None
                else None
            )
            snapshots = _filter_snapshots(
                snapshots,
                normalized_filters,
                exact_ids,
            )
        return sorted(snapshots, key=lambda item: item.last_seen, reverse=True)

    def counts(self) -> tuple[int, int]:
        """Return ``(active logical rows, total retained rows)``."""
        with self._lock:
            active = sum(bool(record.active) for record in self._records.values())
            return active, len(self._records)

    def _evict_inactive(self) -> None:
        while len(self._inactive) > self._inactive_limit:
            oldest_key, _ = self._inactive.popitem(last=False)
            del self._records[oldest_key]


def _begin_activity(
    record: _ActivityRecord,
    metadata: _SnapshotMetadata,
    request_id: str,
    started: float,
    seen_at: datetime,
) -> None:
    record.metadata = metadata
    record.last_seen = max(record.last_seen, seen_at)
    record.requests += 1
    record.active[request_id] = started


def _finish_activity(
    record: _ActivityRecord,
    request_id: str,
    finished_at: float,
    seen_at: datetime,
    result: SessionResult,
) -> bool:
    started = record.active.pop(request_id, None)
    if started is None:
        return False
    record.latest_duration = max(0.0, finished_at - started)
    record.last_seen = max(record.last_seen, seen_at)
    record.last_result = result
    return True


def _strip_model_prefix(model: str) -> str:
    _, separator, unprefixed = model.partition("/")
    return unprefixed if separator else model


def _to_snapshot(record: _SessionRecord, now: float) -> SessionSnapshot:
    active_requests = len(record.active)
    state = _session_state(record)
    elapsed = _elapsed_seconds(record, now)
    metadata = record.metadata
    return SessionSnapshot(
        id=record.public_id,
        state=state,
        active_requests=active_requests,
        requests=record.requests,
        client_model=metadata.client_model,
        model=metadata.model,
        provider=metadata.provider,
        transport=metadata.transport,
        effort=metadata.effort,
        context_window=metadata.context_window,
        first_seen=record.first_seen,
        last_seen=record.last_seen,
        elapsed_seconds=elapsed,
        last_result=record.last_result,
        agents=tuple(
            _to_agent_snapshot(agent, now)
            for agent in sorted(
                record.agents.values(),
                key=lambda item: (item.last_seen, item.public_id),
                reverse=True,
            )
        ),
    )


def _to_agent_snapshot(record: _AgentRecord, now: float) -> AgentSnapshot:
    metadata = record.metadata
    return AgentSnapshot(
        id=record.public_id,
        parent_id=record.parent_public_id,
        state=_session_state(record),
        active_requests=len(record.active),
        requests=record.requests,
        client_model=metadata.client_model,
        model=metadata.model,
        provider=metadata.provider,
        transport=metadata.transport,
        effort=metadata.effort,
        context_window=metadata.context_window,
        first_seen=record.first_seen,
        last_seen=record.last_seen,
        elapsed_seconds=_elapsed_seconds(record, now),
        last_result=record.last_result,
    )


def _session_state(record: _ActivityRecord) -> SessionState:
    if record.active:
        return "active"
    if record.last_result == "failed":
        return "failed"
    return "idle"


def _elapsed_seconds(record: _ActivityRecord, now: float) -> float:
    if not record.active:
        return record.latest_duration
    return max(0.0, now - min(record.active.values()))


def _validate_filters(
    filters: SessionFilters | None,
) -> dict[str, tuple[str, ...]] | None:
    if filters is None:
        return None
    if not isinstance(filters, Mapping) or not filters:
        raise InvalidSessionFilter("filters must contain at least one key")

    normalized: dict[str, tuple[str, ...]] = {}
    for key, values in filters.items():
        if not isinstance(key, str) or not key.strip() or key not in _FILTER_FIELDS:
            raise InvalidSessionFilter(f"unsupported session filter {key!r}")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise InvalidSessionFilter(f"filter {key!r} requires a sequence of values")
        entries = tuple(values)
        invalid_value = any(
            not isinstance(value, str)
            or not value
            or (key == "session_id" and not value.strip())
            for value in entries
        )
        if not entries or invalid_value:
            raise InvalidSessionFilter(
                f"filter {key!r} requires non-empty values"
            )
        normalized[key] = entries
    return normalized


def _filter_snapshots(
    snapshots: list[SessionSnapshot],
    filters: dict[str, tuple[str, ...]],
    exact_ids: set[str] | None,
) -> list[SessionSnapshot]:
    selected_ids = _resolve_id_filters(snapshots, filters.get("id"))
    matches = snapshots
    if selected_ids is not None:
        matches = [item for item in matches if item.id in selected_ids]
    if exact_ids is not None:
        matches = [item for item in matches if item.id in exact_ids]

    for key, values in filters.items():
        if key in {"id", "session_id"}:
            continue
        accepted = {value.casefold() for value in values}
        matches = [
            item
            for item in matches
            if str(getattr(item, key)).casefold() in accepted
        ]
    return matches


def _resolve_id_filters(
    snapshots: list[SessionSnapshot],
    prefixes: tuple[str, ...] | None,
) -> set[str] | None:
    if prefixes is None:
        return None

    selected: set[str] = set()
    for prefix in prefixes:
        folded = prefix.casefold()
        matches = [
            item.id
            for item in snapshots
            if item.id.casefold().startswith(folded)
        ]
        if len(matches) > 1:
            raise AmbiguousSessionId(f"session ID prefix {prefix!r} is ambiguous")
        selected.update(matches)
    return selected
