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

from .domain.models import (
    ClientIdentity,
    CompletionResponse,
    StreamEvent,
    TokenUsage,
    ToolUseStart,
)
from .event_journal import EventJournal, EventType, JournalEvent, Subscription
from .failures import FailureDiagnostic
from .limits import MAX_CONTROL_INTEGER
from .performance import (
    OperationKind,
    ReasoningContinuation,
    RequestOutcome,
    RequestPerformance,
    RequestPerformanceSnapshot,
    RequestTelemetryObserver,
    SessionPerformance,
    SessionPerformanceSnapshot,
)
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
    operation: OperationKind = "messages"
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


@dataclass(frozen=True, slots=True)
class SessionPerformanceView:
    session: SessionSnapshot
    performance: SessionPerformanceSnapshot


@dataclass(frozen=True, slots=True)
class PerformanceCapture:
    captured_at: datetime
    cursor: int
    sessions: tuple[SessionPerformanceView, ...]


@dataclass(frozen=True, slots=True)
class PerformanceSubscription:
    initial: PerformanceCapture | None
    subscription: Subscription


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


@dataclass(kw_only=True)
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
    performance: SessionPerformance
    agents: dict[str, _AgentRecord] = field(default_factory=dict)
    progress_at: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _BeginContext:
    request_id: str
    key: str
    root_identifier: str
    public_id: str
    request_scoped: bool


class SessionRegistry:
    """Track request lifecycle by logical session without exposing client IDs."""

    def __init__(
        self,
        inactive_limit: int,
        secret: bytes | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        events: EventJournal | None = None,
    ) -> None:
        _validate_inactive_limit(inactive_limit)
        self._inactive_limit = inactive_limit
        self._secret = secrets.token_bytes(32) if secret is None else secret
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._events = events or EventJournal(4096, 64)
        self._records: dict[str, _SessionRecord] = {}
        self._inactive: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def inactive_limit(self) -> int:
        return self._inactive_limit

    @property
    def events(self) -> EventJournal:
        return self._events

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

    def begin(
        self,
        metadata: SessionMetadata,
        *,
        operation: OperationKind = "messages",
        started_at: datetime | None = None,
        started_monotonic: float | None = None,
    ) -> ObservationHandle:
        context = self._begin_context(metadata)
        with self._lock:
            monotonic = (
                self._monotonic_clock()
                if started_monotonic is None
                else started_monotonic
            )
            wall = self._wall_clock() if started_at is None else started_at
            request = RequestPerformance(
                context.request_id,
                context.public_id,
                operation,
                wall,
                monotonic,
                1,
            )
            self._ensure_event_capacity_locked()
            handle, record = self._start_locked(
                context, metadata, request, wall, monotonic
            )
            self._publish_event_locked(
                record, request, "request_started", wall, monotonic
            )
            record.progress_at[request.request_id] = monotonic
            return handle

    def _begin_context(self, metadata: SessionMetadata) -> "_BeginContext":
        identity = metadata.client_identity
        normalized_id = (identity.session_id or "").strip()
        request_id = uuid.uuid4().hex
        request_scoped = not normalized_id
        root = f"request:{request_id}" if request_scoped else normalized_id
        public_id = self.public_id(root)
        key = f"request:{request_id}" if request_scoped else f"session:{public_id}"
        return _BeginContext(
            request_id, key, root, public_id, request_scoped
        )

    def _start_locked(
        self,
        context: "_BeginContext",
        metadata: SessionMetadata,
        request: RequestPerformance,
        wall: datetime,
        monotonic: float,
    ) -> tuple[ObservationHandle, _SessionRecord]:
        snapshot_metadata = _SnapshotMetadata.from_session(metadata)
        record = self._records.get(context.key)
        is_new = record is None
        if record is None:
            record = _new_session_record(
                context.public_id, snapshot_metadata, wall
            )
        record.performance.start(request)
        _begin_activity(
            record, snapshot_metadata, context.request_id, monotonic, wall
        )
        self._inactive.pop(context.key, None)
        agent = self._begin_agent(
            record,
            self._agent_identity(context.root_identifier, metadata.client_identity),
            snapshot_metadata,
            context.request_id,
            monotonic,
            wall,
        )
        if is_new:
            self._records[context.key] = record
        return _to_handle(context, request, is_new, agent), record

    def observer(self, handle: ObservationHandle) -> RequestTelemetryObserver:
        with self._lock:
            if self._request_locked(handle) is None:
                raise ValueError("observation handle is unknown or finalized")
        return RequestTelemetryObserver(self, handle)

    def upstream_started(self, handle: ObservationHandle) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            now = self._monotonic_clock()
            if request.mark_upstream_started(now):
                self._publish_progress_locked(
                    record, request, self._wall_clock(), now
                )

    def upstream_finished(self, handle: ObservationHandle) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            now = self._monotonic_clock()
            if request.mark_upstream_finished(now):
                self._publish_progress_locked(
                    record, request, self._wall_clock(), now
                )

    def stream_event(
        self, handle: ObservationHandle, event: StreamEvent
    ) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            now = self._monotonic_clock()
            first_output = request.observe_stream_event(event, now)
            occurred_at = self._wall_clock()
            immediate = self._publish_stream_immediate_locked(
                record, request, event, first_output, occurred_at, now
            )
            if not immediate:
                self._publish_progress_locked(
                    record, request, occurred_at, now
                )

    def _publish_stream_immediate_locked(
        self,
        record: _SessionRecord,
        request: RequestPerformance,
        event: StreamEvent,
        first_output: bool,
        occurred_at: datetime,
        now: float,
    ) -> bool:
        published = False
        if first_output:
            self._publish_immediate_locked(
                record, request, "first_output", occurred_at, now
            )
            published = True
        if isinstance(event, ToolUseStart):
            self._publish_immediate_locked(
                record, request, "tool_use", occurred_at, now
            )
            published = True
        return published

    def response(
        self, handle: ObservationHandle, response: CompletionResponse
    ) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            now = self._monotonic_clock()
            had_output = request.snapshot(now).ttft.status == "observed"
            request.observe_response(response, now)
            has_output = request.snapshot(now).ttft.status == "observed"
            occurred_at = self._wall_clock()
            if has_output and not had_output:
                self._publish_immediate_locked(
                    record, request, "first_output", occurred_at, now
                )
                return
            self._publish_progress_locked(record, request, occurred_at, now)

    def count_tokens(self, handle: ObservationHandle, value: int) -> None:
        _validate_count_tokens(value)
        usage = TokenUsage(
            value, 0, observed_fields=frozenset({"input_tokens"})
        )
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            request.record_usage(usage)
            now = self._monotonic_clock()
            self._publish_progress_locked(
                record, request, self._wall_clock(), now
            )

    def mark_retries_supported(self, handle: ObservationHandle) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is not None:
                target[1].mark_retries_supported()

    def record_retry(self, handle: ObservationHandle) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            request.record_retry()
            now = self._monotonic_clock()
            self._publish_immediate_locked(
                record, request, "retry", self._wall_clock(), now
            )

    def set_reasoning_continuation(
        self,
        handle: ObservationHandle,
        value: ReasoningContinuation,
    ) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            request.set_reasoning_continuation(value)
            now = self._monotonic_clock()
            self._publish_progress_locked(
                record, request, self._wall_clock(), now
            )

    def finish(
        self,
        handle: ObservationHandle,
        result: RequestOutcome,
        failure: FailureDiagnostic | None = None,
    ) -> RequestPerformanceSnapshot | None:
        _validate_terminal_outcome(result)
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return None
            record, request = target
            finished_at = self._wall_clock()
            finished_monotonic = self._monotonic_clock()
            self._ensure_event_capacity_locked()
            request.finish(
                result, finished_at, finished_monotonic, failure
            )
            terminal = record.performance.add_finalized(request)
            if terminal is None:
                raise RuntimeError("request finalization invariant violated")
            self._finish_base_locked(
                record, handle, finished_at, finished_monotonic, result
            )
            self._publish_event_locked(
                record, request, result, finished_at, finished_monotonic
            )
            record.progress_at.pop(handle.request_id, None)
            self._retain_finished_locked(handle, record)
            return terminal

    def _finish_base_locked(
        self,
        record: _SessionRecord,
        handle: ObservationHandle,
        finished_at: datetime,
        finished_monotonic: float,
        outcome: RequestOutcome,
    ) -> None:
        base_result: SessionResult = (
            "completed" if outcome == "completed" else "failed"
        )
        if not _finish_activity(
            record,
            handle.request_id,
            finished_monotonic,
            finished_at,
            base_result,
        ):
            raise RuntimeError("base request finalization invariant violated")
        if handle.agent_key is None:
            return
        agent = record.agents.get(handle.agent_key)
        if agent is not None:
            _finish_activity(
                agent,
                handle.request_id,
                finished_monotonic,
                finished_at,
                base_result,
            )

    def _retain_finished_locked(
        self, handle: ObservationHandle, record: _SessionRecord
    ) -> None:
        if record.active:
            return
        self._inactive[handle.key] = None
        self._inactive.move_to_end(handle.key)
        self._evict_inactive()

    def snapshots(
        self,
        filters: SessionFilters | None = None,
    ) -> list[SessionSnapshot]:
        """Copy snapshots and optionally filter their public fields."""
        normalized = _validate_filters(filters)
        with self._lock:
            return self._session_snapshots_locked(normalized)

    def performance_snapshots(
        self, filters: SessionFilters | None = None
    ) -> PerformanceCapture:
        normalized = _validate_filters(filters)
        with self._lock:
            return self._performance_capture_locked(normalized)

    def subscribe_performance(
        self,
        filters: SessionFilters | None,
        after: int | None,
    ) -> PerformanceSubscription:
        normalized = _validate_filters(filters)
        with self._lock:
            subscription = self._events.subscribe(after)
            initial = None
            if after is None or subscription.reset_required:
                initial = self._performance_capture_locked(normalized)
            return PerformanceSubscription(initial, subscription)

    def _performance_capture_locked(
        self, filters: dict[str, tuple[str, ...]] | None
    ) -> PerformanceCapture:
        now = self._monotonic_clock()
        captured_at = self._wall_clock()
        sessions = self._session_snapshots_at_locked(filters, now)
        records = {record.public_id: record for record in self._records.values()}
        views = tuple(
            SessionPerformanceView(
                session,
                records[session.id].performance.snapshot(now),
            )
            for session in sessions
        )
        return PerformanceCapture(
            captured_at, self._events.current_sequence, views
        )

    def _session_snapshots_locked(
        self, filters: dict[str, tuple[str, ...]] | None
    ) -> list[SessionSnapshot]:
        return self._session_snapshots_at_locked(
            filters, self._monotonic_clock()
        )

    def _session_snapshots_at_locked(
        self,
        filters: dict[str, tuple[str, ...]] | None,
        now: float,
    ) -> list[SessionSnapshot]:
        snapshots = [
            _to_snapshot(record, now) for record in self._records.values()
        ]
        if filters is not None:
            session_ids = filters.get("session_id")
            exact_ids = (
                {self.public_id(value) for value in session_ids}
                if session_ids is not None
                else None
            )
            snapshots = _filter_snapshots(snapshots, filters, exact_ids)
        return sorted(
            snapshots, key=lambda item: item.last_seen, reverse=True
        )

    def counts(self) -> tuple[int, int]:
        """Return ``(active logical rows, total retained rows)``."""
        with self._lock:
            active = sum(bool(record.active) for record in self._records.values())
            return active, len(self._records)

    def _request_locked(
        self, handle: ObservationHandle
    ) -> tuple[_SessionRecord, RequestPerformance] | None:
        if not isinstance(handle, ObservationHandle):
            return None
        record = self._records.get(handle.key)
        if record is None or record.public_id != handle.public_id:
            return None
        request = record.performance.request(handle.request_id)
        if request is None or request.is_terminal:
            return None
        return record, request

    def _publish_progress_locked(
        self,
        record: _SessionRecord,
        request: RequestPerformance,
        occurred_at: datetime,
        now: float,
    ) -> None:
        last = record.progress_at.get(request.request_id)
        if last is not None and now - last < 0.25:
            return
        self._publish_event_locked(
            record, request, "progress", occurred_at, now
        )
        record.progress_at[request.request_id] = now

    def _publish_immediate_locked(
        self,
        record: _SessionRecord,
        request: RequestPerformance,
        event_type: EventType,
        occurred_at: datetime,
        now: float,
    ) -> None:
        self._publish_event_locked(
            record, request, event_type, occurred_at, now
        )
        record.progress_at[request.request_id] = now

    def _publish_event_locked(
        self,
        record: _SessionRecord,
        request: RequestPerformance,
        event_type: EventType,
        occurred_at: datetime,
        now: float,
    ) -> None:
        event = JournalEvent(
            sequence=0,
            occurred_at=occurred_at,
            type=event_type,
            session_id=record.public_id,
            request_id=request.request_id,
            request=request.snapshot(now),
            session=record.performance.snapshot(now),
        )
        self._events.publish(event)

    def _ensure_event_capacity_locked(self) -> None:
        if self._events.current_sequence >= MAX_CONTROL_INTEGER:
            raise ValueError("event sequence exceeds the control limit")

    def _evict_inactive(self) -> None:
        while len(self._inactive) > self._inactive_limit:
            oldest_key, _ = self._inactive.popitem(last=False)
            del self._records[oldest_key]


def _validate_inactive_limit(inactive_limit: object) -> None:
    if (
        type(inactive_limit) is not int
        or not 0 <= inactive_limit <= MAX_CONTROL_INTEGER
    ):
        raise ValueError(
            "inactive_limit must be an integer between 0 and "
            f"{MAX_CONTROL_INTEGER}"
        )


def _new_session_record(
    public_id: str, metadata: _SnapshotMetadata, seen_at: datetime
) -> _SessionRecord:
    return _SessionRecord(
        public_id=public_id,
        metadata=metadata,
        first_seen=seen_at,
        last_seen=seen_at,
        requests=0,
        active={},
        performance=SessionPerformance(public_id),
    )


def _to_handle(
    context: _BeginContext,
    request: RequestPerformance,
    is_new: bool,
    agent: tuple[str | None, str | None, str | None, bool],
) -> ObservationHandle:
    agent_key, agent_public_id, parent_public_id, agent_is_new = agent
    return ObservationHandle(
        key=context.key,
        request_id=context.request_id,
        public_id=context.public_id,
        operation=request.operation,
        started_monotonic=request.started_monotonic,
        is_new=is_new,
        request_scoped=context.request_scoped,
        agent_key=agent_key,
        agent_public_id=agent_public_id,
        parent_agent_public_id=parent_public_id,
        agent_is_new=agent_is_new,
    )


def _validate_terminal_outcome(outcome: object) -> None:
    if outcome not in {
        "completed",
        "failed",
        "cancelled",
        "client_disconnected",
    }:
        raise ValueError("finish requires a terminal outcome")


def _validate_count_tokens(value: object) -> None:
    if type(value) is not int or not 0 <= value <= MAX_CONTROL_INTEGER:
        raise ValueError(
            "count_tokens value must be an integer between 0 and "
            f"{MAX_CONTROL_INTEGER}"
        )

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
        _validate_filter_key(key)
        normalized[key] = _validate_filter_values(key, values)
    return normalized


def _validate_filter_key(key: object) -> None:
    if not isinstance(key, str) or not key.strip() or key not in _FILTER_FIELDS:
        raise InvalidSessionFilter(f"unsupported session filter {key!r}")


def _validate_filter_values(
    key: str, values: object
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise InvalidSessionFilter(
            f"filter {key!r} requires a sequence of values"
        )
    entries = tuple(values)
    if not entries or any(_invalid_filter_value(key, value) for value in entries):
        raise InvalidSessionFilter(
            f"filter {key!r} requires non-empty values"
        )
    return entries


def _invalid_filter_value(key: str, value: object) -> bool:
    if not isinstance(value, str) or not value:
        return True
    return key == "session_id" and not value.strip()


def _filter_snapshots(
    snapshots: list[SessionSnapshot],
    filters: dict[str, tuple[str, ...]],
    exact_ids: set[str] | None,
) -> list[SessionSnapshot]:
    selected_ids = _resolve_id_filters(snapshots, filters.get("id"))
    matches = _filter_snapshot_ids(snapshots, selected_ids, exact_ids)
    for key, values in filters.items():
        if key not in {"id", "session_id"}:
            matches = _filter_snapshot_field(matches, key, values)
    return matches


def _filter_snapshot_ids(
    snapshots: list[SessionSnapshot],
    selected_ids: set[str] | None,
    exact_ids: set[str] | None,
) -> list[SessionSnapshot]:
    matches = snapshots
    if selected_ids is not None:
        matches = [item for item in matches if item.id in selected_ids]
    if exact_ids is not None:
        matches = [item for item in matches if item.id in exact_ids]
    return matches


def _filter_snapshot_field(
    snapshots: list[SessionSnapshot],
    key: str,
    values: tuple[str, ...],
) -> list[SessionSnapshot]:
    accepted = {value.casefold() for value in values}
    return [
        item
        for item in snapshots
        if str(getattr(item, key)).casefold() in accepted
    ]

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
