"""Thread-safe, privacy-preserving runtime session observations."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import hmac
import secrets
import threading
import time
import uuid

from .domain.models import ClientIdentity, CompletionResponse, StreamEvent
from .domain.models import TokenUsage, ToolUseStart
from .event_journal import EventJournal, EventReservation, EventType, JournalEvent, SessionEventIdentity, Subscription
from .failures import FailureDiagnostic
from .finalization import FinalizationResult, validate_finalization
from .limits import MAX_CONTROL_INTEGER
from .performance import OperationKind, ReasoningContinuation, RequestOutcome, RequestPerformance, RequestPerformanceSnapshot
from .performance import RequestTelemetryObserver, SessionPerformance, SessionPerformanceSnapshot, validate_clock_sample
from .performance_filters import (
    AmbiguousSessionId,
    FilterMap as SessionFilters,
    InvalidSessionFilter as InvalidSessionFilter,
    mutable_performance_filters,
    prepare_performance_filters,
    validate_performance_filters,
)
from .session_inventory import (
    SessionResult,
    SessionState,
    begin_activity as _begin_activity,
    elapsed_seconds as _elapsed_seconds,
    finish_activity as _finish_activity,
    session_state as _session_state,
    strip_model_prefix as _strip_model_prefix,
)
from .session_snapshot_filters import filter_snapshots as _filter_snapshots
from .text_safety import (
    TELEMETRY_ATTRIBUTE_MAX_LENGTH,
    TELEMETRY_MODEL_MAX_LENGTH,
    retained_telemetry_text,
)


class PerformanceUnavailable(RuntimeError):
    """Raised when performance telemetry was not enabled at startup."""

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
        limits = {
            "client_model": TELEMETRY_MODEL_MAX_LENGTH,
            "upstream_model": TELEMETRY_MODEL_MAX_LENGTH,
            "provider": TELEMETRY_ATTRIBUTE_MAX_LENGTH,
            "transport": TELEMETRY_ATTRIBUTE_MAX_LENGTH,
            "effort": TELEMETRY_ATTRIBUTE_MAX_LENGTH,
        }
        for name, limit in limits.items():
            retained = retained_telemetry_text(
                getattr(self, name), max_length=limit
            )
            object.__setattr__(self, name, retained)


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
    operation: OperationKind = "messages"


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
    event_filters: SessionFilters | None


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
    performance: SessionPerformance | None
    agents: dict[str, _AgentRecord] = field(default_factory=dict)
    progress_at: dict[str, float] | None = None


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
        performance_enabled: bool = True,
        performance_logging_enabled: bool | None = None,
    ) -> None:
        _validate_inactive_limit(inactive_limit)
        _validate_performance_enabled(performance_enabled)
        logging_enabled = (
            performance_enabled
            if performance_logging_enabled is None
            else performance_logging_enabled
        )
        _validate_performance_logging_enabled(logging_enabled)
        if logging_enabled and not performance_enabled:
            raise ValueError(
                "performance logging requires performance collection"
            )
        self._inactive_limit = inactive_limit
        self._performance_enabled = performance_enabled
        self._performance_logging_enabled = logging_enabled
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

    @property
    def performance_enabled(self) -> bool:
        return self._performance_enabled

    @property
    def performance_logging_enabled(self) -> bool:
        return self._performance_logging_enabled

    def sample_clocks(self) -> tuple[datetime, float]:
        """Sample and validate the registry's clock domain atomically."""
        with self._lock:
            wall = self._wall_clock()
            monotonic = validate_clock_sample(wall, self._monotonic_clock())
            return wall, monotonic

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
        self, root_identifier: str, identity: ClientIdentity,
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
                started_monotonic
                if started_monotonic is not None
                else self._monotonic_clock()
            )
            wall = self._wall_clock() if started_at is None else started_at
            if not self._performance_enabled:
                _validate_operation(operation)
                validate_clock_sample(wall, monotonic)
                handle, _ = self._start_locked(
                    context, metadata, None, operation, wall, monotonic
                )
                return handle
            request = RequestPerformance(
                context.request_id,
                context.public_id,
                operation,
                wall,
                monotonic,
                1,
            )
            with self._events.reserve(1) as reservation:
                handle, record = self._start_locked(
                    context, metadata, request, operation, wall, monotonic
                )
                self._commit_events_locked(
                    reservation, record, request,
                    ("request_started",), wall, monotonic,
                )
                return handle

    def _begin_context(self, metadata: SessionMetadata) -> "_BeginContext":
        identity = metadata.client_identity
        normalized_id = (identity.session_id or "").strip()
        request_id = uuid.uuid4().hex
        request_scoped = not normalized_id
        root = f"request:{request_id}" if request_scoped else normalized_id
        public_id = self.public_id(root)
        key = f"request:{request_id}" if request_scoped else f"session:{public_id}"
        return _BeginContext(request_id, key, root, public_id, request_scoped)

    def _start_locked(
        self,
        context: "_BeginContext",
        metadata: SessionMetadata,
        request: RequestPerformance | None,
        operation: OperationKind,
        wall: datetime,
        monotonic: float,
    ) -> tuple[ObservationHandle, _SessionRecord]:
        snapshot_metadata = _SnapshotMetadata.from_session(metadata)
        record = self._records.get(context.key)
        is_new = record is None
        if record is None:
            record = _new_session_record(
                context.public_id, snapshot_metadata, wall,
                performance_enabled=self._performance_enabled,
            )
        if request is not None:
            assert record.performance is not None
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
        return _to_handle(
            context, operation, monotonic, is_new, agent
        ), record

    def observer(self, handle: ObservationHandle) -> RequestTelemetryObserver:
        self._require_performance()
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
            occurred_at, now = self._sample_event_clocks_locked(request)
            publish = (
                request.would_mark_upstream_started()
                and self._progress_due(record, request, now)
            )
            self._update_upstream_locked(
                record, request, request.mark_upstream_started,
                occurred_at, now, publish,
            )

    def upstream_finished(self, handle: ObservationHandle) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            occurred_at, now = self._sample_event_clocks_locked(request)
            publish = (
                request.would_mark_upstream_finished()
                and self._progress_due(record, request, now)
            )
            self._update_upstream_locked(
                record, request, request.mark_upstream_finished,
                occurred_at, now, publish,
            )

    def _update_upstream_locked(
        self, record: _SessionRecord, request: RequestPerformance,
        mutation: Callable[[float], bool], occurred_at: datetime,
        now: float, publish: bool,
    ) -> None:
        if not publish:
            mutation(now)
            return
        with self._events.reserve(1) as reservation:
            if mutation(now):
                self._commit_events_locked(
                    reservation, record, request, ("progress",),
                    occurred_at, now,
                )

    def stream_event(
        self, handle: ObservationHandle, event: StreamEvent
    ) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            occurred_at, now = self._sample_event_clocks_locked(request)
            event_types = self._stream_event_plan(record, request, event, now)
            if not event_types:
                request.observe_stream_event(event, now)
                return
            with self._events.reserve(len(event_types)) as reservation:
                request.observe_stream_event(event, now)
                self._commit_events_locked(
                    reservation, record, request, event_types,
                    occurred_at, now,
                )

    def _stream_event_plan(
        self, record: _SessionRecord, request: RequestPerformance,
        event: StreamEvent, now: float,
    ) -> tuple[EventType, ...]:
        immediate: list[EventType] = []
        if request.would_mark_stream_output(event):
            immediate.append("first_output")
        if isinstance(event, ToolUseStart):
            immediate.append("tool_use")
        if immediate:
            return tuple(immediate)
        if self._progress_due(record, request, now):
            return ("progress",)
        return ()

    def response(
        self, handle: ObservationHandle, response: CompletionResponse
    ) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            occurred_at, now = self._sample_event_clocks_locked(request)
            if request.would_mark_response_output(response):
                event_types: tuple[EventType, ...] = ("first_output",)
            elif self._progress_due(record, request, now):
                event_types = ("progress",)
            else:
                event_types = ()
            self._observe_response_locked(
                record, request, response, occurred_at, now, event_types
            )

    def _observe_response_locked(
        self, record: _SessionRecord, request: RequestPerformance,
        response: CompletionResponse, occurred_at: datetime, now: float,
        event_types: tuple[EventType, ...],
    ) -> None:
        if not event_types:
            request.observe_response(response, now)
            return
        with self._events.reserve(len(event_types)) as reservation:
            request.observe_response(response, now)
            self._commit_events_locked(
                reservation, record, request, event_types,
                occurred_at, now,
            )

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
            occurred_at, now = self._sample_event_clocks_locked(request)
            if not self._progress_due(record, request, now):
                request.record_usage(usage)
                return
            with self._events.reserve(1) as reservation:
                request.record_usage(usage)
                self._commit_events_locked(
                    reservation, record, request, ("progress",),
                    occurred_at, now,
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
            occurred_at, now = self._sample_event_clocks_locked(request)
            with self._events.reserve(1) as reservation:
                request.record_retry()
                self._commit_events_locked(
                    reservation, record, request, ("retry",),
                    occurred_at, now,
                )

    def set_reasoning_continuation(
        self, handle: ObservationHandle, value: ReasoningContinuation,
    ) -> None:
        with self._lock:
            target = self._request_locked(handle)
            if target is None:
                return
            record, request = target
            occurred_at, now = self._sample_event_clocks_locked(request)
            if not self._progress_due(record, request, now):
                request.set_reasoning_continuation(value)
                return
            with self._events.reserve(1) as reservation:
                request.set_reasoning_continuation(value)
                self._commit_events_locked(
                    reservation, record, request, ("progress",),
                    occurred_at, now,
                )

    def finish(
        self,
        handle: ObservationHandle,
        result: RequestOutcome,
        failure: FailureDiagnostic | None = None,
    ) -> RequestPerformanceSnapshot | None:
        return self.finish_with_status(handle, result, failure).performance

    def finish_with_status(
        self,
        handle: ObservationHandle,
        result: RequestOutcome,
        failure: FailureDiagnostic | None = None,
    ) -> FinalizationResult:
        validate_finalization(result, failure)
        with self._lock:
            if not self._performance_enabled:
                return self._finish_without_performance_locked(handle, result)
            return self._finish_with_performance_locked(
                handle, result, failure
            )

    def _finish_with_performance_locked(
        self,
        handle: ObservationHandle,
        result: RequestOutcome,
        failure: FailureDiagnostic | None,
    ) -> FinalizationResult:
        target = self._request_locked(handle)
        if target is None:
            return FinalizationResult(False, None)
        record, request = target
        finished_at = self._wall_clock()
        finished_monotonic = self._monotonic_clock()
        with self._events.reserve(1) as reservation:
            request.finish(result, finished_at, finished_monotonic, failure)
            assert record.performance is not None
            terminal = record.performance.add_finalized(request)
            if terminal is None:
                raise RuntimeError("request finalization invariant violated")
            self._finish_base_locked(
                record, handle, finished_at, finished_monotonic, result
            )
            self._commit_events_locked(
                reservation, record, request, (result,),
                finished_at, finished_monotonic,
            )
            assert record.progress_at is not None
            record.progress_at.pop(handle.request_id, None)
            self._retain_finished_locked(handle, record)
            return FinalizationResult(True, terminal)

    def _finish_without_performance_locked(
        self, handle: ObservationHandle, result: RequestOutcome
    ) -> FinalizationResult:
        record = self._active_record_locked(handle)
        if record is None:
            return FinalizationResult(False, None)
        finished_at = self._wall_clock()
        finished_monotonic = validate_clock_sample(
            finished_at, self._monotonic_clock()
        )
        self._finish_base_locked(
            record, handle, finished_at, finished_monotonic, result
        )
        self._retain_finished_locked(handle, record)
        return FinalizationResult(True, None)

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
        normalized = validate_performance_filters(filters)
        with self._lock:
            return self._session_snapshots_at_locked(
                normalized, self._monotonic_clock())

    def performance_snapshots(
        self, filters: SessionFilters | None = None
    ) -> PerformanceCapture:
        self._require_performance()
        normalized = validate_performance_filters(filters)
        with self._lock:
            return self._performance_capture_locked(normalized)

    def subscribe_performance(
        self,
        filters: SessionFilters | None,
        after: int | None,
        *,
        resolved: bool = False,
    ) -> PerformanceSubscription:
        self._require_performance()
        normalized = validate_performance_filters(filters)
        with self._lock:
            subscription = self._events.subscribe(after)
            try:
                event_filters = prepare_performance_filters(
                    normalized,
                    (record.public_id for record in self._records.values()),
                    subscription.replay,
                    self.public_id,
                    resolved=resolved,
                )
                initial = None
                if after is None or subscription.reset_required:
                    initial = self._performance_capture_locked(
                        mutable_performance_filters(event_filters)
                    )
                return PerformanceSubscription(
                    initial, subscription, event_filters
                )
            except BaseException:
                subscription.close()
                raise

    def _performance_capture_locked(
        self, filters: dict[str, tuple[str, ...]] | None
    ) -> PerformanceCapture:
        captured_at = self._wall_clock()
        now = validate_clock_sample(captured_at, self._monotonic_clock())
        sessions = self._session_snapshots_at_locked(filters, now)
        records = {record.public_id: record for record in self._records.values()}
        views = tuple(
            SessionPerformanceView(
                session,
                _performance(records[session.id]).snapshot(now),
            )
            for session in sessions
        )
        return PerformanceCapture(captured_at, self._events.current_sequence, views)

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
        return sorted(snapshots, key=lambda item: item.last_seen, reverse=True)

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
        record = self._active_record_locked(handle)
        if record is None or record.performance is None:
            return None
        request = record.performance.request(handle.request_id)
        if request is None or request.is_terminal:
            return None
        return record, request

    def _active_record_locked(
        self, handle: ObservationHandle
    ) -> _SessionRecord | None:
        if not isinstance(handle, ObservationHandle):
            return None
        record = self._records.get(handle.key)
        if record is None or record.public_id != handle.public_id:
            return None
        if handle.request_id not in record.active:
            return None
        return record

    def _require_performance(self) -> None:
        if not self._performance_enabled:
            raise PerformanceUnavailable("performance telemetry is disabled")

    def _sample_event_clocks_locked(
        self, request: RequestPerformance
    ) -> tuple[datetime, float]:
        occurred_at = self._wall_clock()
        now = validate_clock_sample(occurred_at, self._monotonic_clock())
        request.snapshot(now)
        return occurred_at, now

    @staticmethod
    def _progress_due(
        record: _SessionRecord, request: RequestPerformance, now: float
    ) -> bool:
        assert record.progress_at is not None
        last = record.progress_at.get(request.request_id)
        return last is None or now - last >= 0.25

    def _commit_events_locked(
        self, reservation: EventReservation, record: _SessionRecord,
        request: RequestPerformance, event_types: tuple[EventType, ...],
        occurred_at: datetime, now: float,
    ) -> tuple[JournalEvent, ...]:
        events = tuple(
            self._event_locked(record, request, kind, occurred_at, now)
            for kind in event_types
        )
        published = reservation.publish(events)
        assert record.progress_at is not None
        record.progress_at[request.request_id] = now
        return published

    @staticmethod
    def _event_locked(
        record: _SessionRecord, request: RequestPerformance,
        event_type: EventType, occurred_at: datetime, now: float,
    ) -> JournalEvent:
        return JournalEvent(
            sequence=0, occurred_at=occurred_at, type=event_type,
            session_id=record.public_id, request_id=request.request_id,
            activity=SessionEventIdentity.from_snapshot(_to_snapshot(record, now)),
            request=request.snapshot(now),
            session=_performance(record).snapshot(now),
        )

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


def _validate_performance_enabled(value: object) -> None:
    if type(value) is not bool:
        raise ValueError("performance_enabled must be a boolean")


def _validate_performance_logging_enabled(value: object) -> None:
    if type(value) is not bool:
        raise ValueError(
            "performance_logging_enabled must be a boolean"
        )


def _validate_operation(operation: object) -> None:
    if operation not in {"messages", "count_tokens"}:
        raise ValueError("invalid operation")


def _performance(record: _SessionRecord) -> SessionPerformance:
    if record.performance is None:
        raise PerformanceUnavailable("performance telemetry is disabled")
    return record.performance


def _new_session_record(
    public_id: str, metadata: _SnapshotMetadata, seen_at: datetime,
    *, performance_enabled: bool,
) -> _SessionRecord:
    return _SessionRecord(
        public_id=public_id,
        metadata=metadata,
        first_seen=seen_at,
        last_seen=seen_at,
        requests=0,
        active={},
        performance=(
            SessionPerformance(public_id) if performance_enabled else None
        ),
        progress_at={} if performance_enabled else None,
    )


def _to_handle(
    context: _BeginContext,
    operation: OperationKind,
    started_monotonic: float,
    is_new: bool,
    agent: tuple[str | None, str | None, str | None, bool],
) -> ObservationHandle:
    agent_key, agent_public_id, parent_public_id, agent_is_new = agent
    return ObservationHandle(
        key=context.key,
        request_id=context.request_id,
        public_id=context.public_id,
        operation=operation,
        started_monotonic=started_monotonic,
        is_new=is_new,
        request_scoped=context.request_scoped,
        agent_key=agent_key,
        agent_public_id=agent_public_id,
        parent_agent_public_id=parent_public_id,
        agent_is_new=agent_is_new,
    )


def _validate_count_tokens(value: object) -> None:
    if type(value) is not int or not 0 <= value <= MAX_CONTROL_INTEGER:
        raise ValueError(
            "count_tokens value must be an integer between 0 and "
            f"{MAX_CONTROL_INTEGER}"
        )



def _to_snapshot(record: _SessionRecord, now: float) -> SessionSnapshot:
    metadata = record.metadata
    return SessionSnapshot(
        id=record.public_id,
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
