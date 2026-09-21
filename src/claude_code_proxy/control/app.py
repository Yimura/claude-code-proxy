"""Construction of the isolated, versioned local control application."""

import json
import math
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..event_journal import OVERFLOW, JournalEvent
from ..limits import MAX_CONTROL_INTEGER
from ..performance_filters import FILTER_FIELDS
from ..observability import (
    AmbiguousSessionId,
    InvalidSessionFilter,
    PerformanceCapture,
    PerformanceSubscription,
    SessionFilters,
    SessionRegistry,
)
from .schemas import (
    HealthResponse,
    PerformanceCursorResponse,
    PerformanceEventResponse,
    PerformanceListResponse,
    PerformanceResetResponse,
    ProcessIdentityResponse,
    SessionCounts,
    SessionListResponse,
    SessionResponse,
)
from .streaming import OwnedPerformanceStreamingResponse, SubscriptionOwner

_DISTRIBUTION_NAME = "anthropic-proxy"
_MAX_FILTER_ENTRIES = 32
_MAX_FILTER_ENTRY_LENGTH = 256
_FilterQuery = Annotated[list[str] | None, Query()]
_ResumeQuery = Annotated[list[str] | None, Query()]


@dataclass(frozen=True)
class _ControlContext:
    sessions: SessionRegistry
    started_at: datetime
    application_version: str
    pid: int
    clock: Callable[[], datetime]
    heartbeat_interval: float


def create_control_app(
    sessions: SessionRegistry,
    started_at: datetime | None = None,
    application_version: str | None = None,
    pid: int | None = None,
    clock: Callable[[], datetime] | None = None,
    heartbeat_interval: float = 15.0,
) -> FastAPI:
    """Create the isolated local control API with explicit dependencies."""
    context = _ControlContext(
        sessions=sessions,
        started_at=_aware_utc(
            _utc_now() if started_at is None else started_at,
            "started_at",
        ),
        application_version=(
            _installed_application_version()
            if application_version is None
            else application_version
        ),
        pid=os.getpid() if pid is None else pid,
        clock=_utc_now if clock is None else clock,
        heartbeat_interval=_validate_heartbeat_interval(heartbeat_interval),
    )
    application = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    _register_control_routes(application, context)
    return application


def _register_control_routes(
    application: FastAPI,
    context: _ControlContext,
) -> None:
    @application.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return _health_response(context)

    @application.get("/v1/sessions", response_model=SessionListResponse)
    def list_sessions(filter: _FilterQuery = None) -> SessionListResponse:
        return _session_list_response(context, filter)

    @application.get("/v1/performance", response_model=PerformanceListResponse)
    def list_performance(filter: _FilterQuery = None) -> PerformanceListResponse:
        return _performance_list_response(context, filter)

    @application.get("/v1/performance/events")
    async def stream_performance(
        filter: _FilterQuery = None,
        after: _ResumeQuery = None,
        pid: _ResumeQuery = None,
        started_at: _ResumeQuery = None,
    ) -> StreamingResponse:
        return _performance_event_response(
            context,
            filter,
            after,
            pid,
            started_at,
        )


def _health_response(context: _ControlContext) -> HealthResponse:
    now = _aware_utc(context.clock(), "clock result")
    active, retained = context.sessions.counts()
    return HealthResponse(
        application_version=context.application_version,
        pid=context.pid,
        started_at=context.started_at,
        uptime_seconds=max(0.0, (now - context.started_at).total_seconds()),
        sessions=SessionCounts(active=active, retained=retained),
        inactive_limit=context.sessions.inactive_limit,
    )


def _session_list_response(
    context: _ControlContext,
    entries: list[str] | None,
) -> SessionListResponse:
    captured_at = _aware_utc(context.clock(), "clock result")
    try:
        parsed_filters = _parse_filters(entries)
        snapshots = context.sessions.snapshots(parsed_filters)
    except InvalidSessionFilter:
        raise HTTPException(
            status_code=422,
            detail="Invalid session filter",
        ) from None
    except AmbiguousSessionId:
        raise HTTPException(
            status_code=422,
            detail="Session ID prefix is ambiguous",
        ) from None
    return SessionListResponse(
        captured_at=captured_at,
        sessions=tuple(
            SessionResponse.model_validate(snapshot) for snapshot in snapshots
        ),
    )


def _performance_list_response(
    context: _ControlContext,
    entries: list[str] | None,
) -> PerformanceListResponse:
    try:
        filters = _parse_filters(entries)
        capture = context.sessions.performance_snapshots(filters)
    except InvalidSessionFilter:
        raise HTTPException(
            status_code=422,
            detail="Invalid session filter",
        ) from None
    except AmbiguousSessionId:
        raise HTTPException(
            status_code=422,
            detail="Session ID prefix is ambiguous",
        ) from None
    return _performance_capture_response(context, capture)


def _performance_event_response(
    context: _ControlContext,
    entries: list[str] | None,
    after: list[str] | None,
    pid: list[str] | None,
    started_at: list[str] | None,
) -> StreamingResponse:
    try:
        filters = _parse_filters(entries)
        effective_after = _effective_after(context, after, pid, started_at)
        state = context.sessions.subscribe_performance(filters, effective_after)
    except InvalidSessionFilter:
        raise HTTPException(422, "Invalid session filter") from None
    except AmbiguousSessionId:
        raise HTTPException(422, "Session ID prefix is ambiguous") from None
    except ValueError:
        raise HTTPException(
            422,
            "Invalid performance event request",
        ) from None
    owner = SubscriptionOwner(state.subscription)
    frames = _performance_frames(context, state, owner)
    return OwnedPerformanceStreamingResponse(frames, owner)


async def _performance_frames(
    context: _ControlContext,
    state: PerformanceSubscription,
    owner: SubscriptionOwner,
) -> AsyncIterator[str]:
    original: BaseException | None = None
    try:
        if state.initial is not None:
            yield _reset_line(context, state.initial)
        for event in state.subscription.replay:
            yield _filtered_event_line(context, state.event_filters, event)
        while True:
            item = await owner.subscription.receive(context.heartbeat_interval)
            if item is None:
                yield "\n"
                continue
            if item is OVERFLOW:
                state = _reset_after_overflow(context, state, owner)
                yield _reset_line(context, state.initial)
                continue
            yield _filtered_event_line(context, state.event_filters, item)
    except BaseException as error:
        original = error
        raise
    finally:
        owner.close(original)


def _filtered_event_line(
    context: _ControlContext,
    filters: SessionFilters | None,
    event: JournalEvent,
) -> str:
    if _event_matches(filters, event):
        return _event_line(context, event)
    return _cursor_line(context, event)


def _event_matches(
    filters: SessionFilters | None,
    event: JournalEvent,
) -> bool:
    if filters is None:
        return True
    for key, values in filters.items():
        accepted = {value.casefold() for value in values}
        if key == "id":
            if event.activity.id.casefold() not in accepted:
                return False
            continue
        if str(getattr(event.activity, key)).casefold() not in accepted:
            return False
    return True


def _reset_after_overflow(
    context: _ControlContext,
    previous: PerformanceSubscription,
    owner: SubscriptionOwner,
) -> PerformanceSubscription:
    owner.release()
    state = context.sessions.subscribe_performance(
        previous.event_filters,
        after=None,
        resolved=True,
    )
    owner.replace(state.subscription)
    if state.initial is None:
        owner.release()
        raise RuntimeError("fresh performance subscription requires a reset")
    return state


def _event_line(context: _ControlContext, event: JournalEvent) -> str:
    response = PerformanceEventResponse(
        process=_process_identity(context),
        sequence=event.sequence,
        occurred_at=event.occurred_at,
        type=event.type,
        session_id=event.session_id,
        activity=event.activity,
        request=event.request,
        session=event.session,
    )
    return _ndjson(response)


def _cursor_line(context: _ControlContext, event: JournalEvent) -> str:
    response = PerformanceCursorResponse(
        process=_process_identity(context),
        sequence=event.sequence,
        occurred_at=event.occurred_at,
        type="cursor",
    )
    return _ndjson(response)


def _reset_line(
    context: _ControlContext,
    capture: PerformanceCapture | None,
) -> str:
    if capture is None:
        raise RuntimeError("performance reset requires a capture")
    snapshot = _performance_capture_response(context, capture)
    response = PerformanceResetResponse(
        process=snapshot.process,
        sequence=snapshot.cursor,
        occurred_at=snapshot.captured_at,
        type="reset",
        snapshot=snapshot,
    )
    return _ndjson(response)


def _ndjson(
    response: PerformanceEventResponse
    | PerformanceCursorResponse
    | PerformanceResetResponse,
) -> str:
    payload = response.model_dump(mode="json")
    return json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"


def _performance_capture_response(
    context: _ControlContext,
    capture: PerformanceCapture,
) -> PerformanceListResponse:
    return PerformanceListResponse(
        process=_process_identity(context),
        captured_at=capture.captured_at,
        cursor=capture.cursor,
        sessions=capture.sessions,
    )


def _effective_after(
    context: _ControlContext,
    after: list[str] | None,
    pid: list[str] | None,
    started_at: list[str] | None,
) -> int | None:
    cursor = _parse_query_integer(_single_query_value(after), minimum=0)
    process_pid = _parse_query_integer(_single_query_value(pid), minimum=1)
    process_started_at = _parse_resume_datetime(
        _single_query_value(started_at)
    )
    if cursor is None:
        return None
    if process_pid != context.pid or process_started_at != context.started_at:
        return None
    return cursor


def _single_query_value(values: list[str] | None) -> str | None:
    if values is None:
        return None
    if len(values) != 1:
        raise ValueError("resume query parameter must occur at most once")
    return values[0]


def _parse_query_integer(value: str | None, *, minimum: int) -> int | None:
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError("query integer must use ASCII decimal digits")
    parsed = int(value)
    if not minimum <= parsed <= MAX_CONTROL_INTEGER:
        raise ValueError("query integer is outside the control range")
    return parsed


def _parse_resume_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return _aware_utc(parsed, "resume datetime")
    except (ValueError, OverflowError) as error:
        raise ValueError(
            "resume datetime must be an aware ISO value in the UTC range"
        ) from error


def _process_identity(context: _ControlContext) -> ProcessIdentityResponse:
    return ProcessIdentityResponse(
        pid=context.pid,
        started_at=context.started_at,
    )


def _parse_filters(entries: list[str] | None) -> SessionFilters | None:
    """Parse at most 32 entries of at most 256 characters each."""
    if entries is None:
        return None
    if len(entries) > _MAX_FILTER_ENTRIES:
        raise InvalidSessionFilter("too many filter entries")

    aggregated: dict[str, list[str]] = {}
    for entry in entries:
        if len(entry) > _MAX_FILTER_ENTRY_LENGTH:
            raise InvalidSessionFilter("filter entry is too long")
        if entry.count("=") != 1:
            raise InvalidSessionFilter("malformed filter entry")
        raw_key, raw_value = entry.split("=", 1)
        key = raw_key.strip()
        value = raw_value.strip()
        if not key or not value or key not in FILTER_FIELDS:
            raise InvalidSessionFilter("malformed filter entry")
        aggregated.setdefault(key, []).append(value)
    return {key: tuple(values) for key, values in aggregated.items()}


def _installed_application_version() -> str:
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"Cannot determine installed version for {_DISTRIBUTION_NAME!r}"
        ) from error


def _validate_heartbeat_interval(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("heartbeat interval must be finite and positive")
    try:
        interval = float(value)
    except (OverflowError, ValueError):
        raise ValueError("heartbeat interval must be finite and non-negative") from None
    if interval <= 0 or not math.isfinite(interval):
        raise ValueError("heartbeat interval must be finite and positive")
    return interval


def _aware_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)
