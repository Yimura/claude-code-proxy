"""Construction of the isolated, versioned local control application."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from ..observability import (
    AmbiguousSessionId,
    InvalidSessionFilter,
    SessionFilters,
    SessionRegistry,
)
from .schemas import (
    HealthResponse,
    SessionCounts,
    SessionListResponse,
    SessionResponse,
)

_DISTRIBUTION_NAME = "anthropic-proxy"
_MAX_FILTER_ENTRIES = 32
_MAX_FILTER_ENTRY_LENGTH = 256
_SUPPORTED_FILTERS = frozenset(
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
_FilterQuery = Annotated[list[str] | None, Query()]


@dataclass(frozen=True)
class _ControlContext:
    sessions: SessionRegistry
    started_at: datetime
    application_version: str
    pid: int
    clock: Callable[[], datetime]


def create_control_app(
    sessions: SessionRegistry,
    started_at: datetime | None = None,
    application_version: str | None = None,
    pid: int | None = None,
    clock: Callable[[], datetime] | None = None,
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
        if not key or not value or key not in _SUPPORTED_FILTERS:
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


def _aware_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)
