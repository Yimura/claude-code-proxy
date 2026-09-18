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

from .limits import MAX_CONTROL_INTEGER
from .text_safety import scalar_text

SessionState = Literal["active", "idle", "failed"]
SessionResult = Literal["completed", "failed"]
SessionFilters = Mapping[str, Sequence[str]]

_FILTER_FIELDS = frozenset(
    {"id", "state", "provider", "transport", "model", "effort"}
)


@dataclass(frozen=True)
class SessionMetadata:
    client_session_id: str | None = field(repr=False)
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
class _SessionRecord:
    public_id: str
    metadata: _SnapshotMetadata
    first_seen: datetime
    last_seen: datetime
    requests: int
    active: dict[str, float]
    latest_duration: float = 0.0
    last_result: SessionResult | None = None


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

    def begin(self, metadata: SessionMetadata) -> ObservationHandle:
        normalized_id = (metadata.client_session_id or "").strip()
        request_id = uuid.uuid4().hex
        request_scoped = not normalized_id
        identity = f"request:{request_id}" if request_scoped else normalized_id
        public_id = self.public_id(identity)
        key = f"request:{request_id}" if request_scoped else f"session:{public_id}"
        snapshot_metadata = _SnapshotMetadata.from_session(metadata)

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
            record.metadata = snapshot_metadata
            record.last_seen = max(record.last_seen, seen_at)
            record.requests += 1
            record.active[request_id] = started
            self._inactive.pop(key, None)

        return ObservationHandle(
            key=key,
            request_id=request_id,
            public_id=public_id,
            started_monotonic=started,
            is_new=is_new,
            request_scoped=request_scoped,
        )

    def finish(self, handle: ObservationHandle, result: SessionResult) -> None:
        """Finish one request; repeated or unknown handles are ignored."""
        with self._lock:
            record = self._records.get(handle.key)
            if record is None:
                return
            started = record.active.get(handle.request_id)
            if started is None:
                return

            finished_at = self._monotonic_clock()
            seen_at = self._wall_clock()
            del record.active[handle.request_id]
            record.latest_duration = max(0.0, finished_at - started)
            record.last_seen = max(record.last_seen, seen_at)
            record.last_result = result
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
            snapshots = _filter_snapshots(snapshots, normalized_filters)
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
    )


def _session_state(record: _SessionRecord) -> SessionState:
    if record.active:
        return "active"
    if record.last_result == "failed":
        return "failed"
    return "idle"


def _elapsed_seconds(record: _SessionRecord, now: float) -> float:
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
            not isinstance(value, str) or not value for value in entries
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
) -> list[SessionSnapshot]:
    selected_ids = _resolve_id_filters(snapshots, filters.get("id"))
    matches = snapshots
    if selected_ids is not None:
        matches = [item for item in matches if item.id in selected_ids]

    for key, values in filters.items():
        if key == "id":
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
