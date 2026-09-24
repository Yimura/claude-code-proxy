"""Responsive, literal-text widgets for live performance telemetry."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Iterable, Protocol

from rich.text import Text
from textual.widgets import DataTable, Static

from ..control.schemas import RequestPerformanceResponse
from .formatting import (
    format_aggregate,
    format_cache_ratio,
    format_latest_metric,
    format_request_elapsed,
    format_session_tokens,
    metric_detail,
    safe_cell,
)
from .state import StateDelta, TuiState, visible_session_ids


class WidthMode(str, Enum):
    """Column sets selected from the terminal width."""

    WIDE = "wide"
    MEDIUM = "medium"
    NARROW = "narrow"


def width_mode(width: int) -> WidthMode:
    """Map terminal width to the documented responsive mode."""
    if width >= 120:
        return WidthMode.WIDE
    if width >= 90:
        return WidthMode.MEDIUM
    return WidthMode.NARROW


_COLUMNS: dict[str, tuple[str, int]] = {
    "session": ("SESSION", 14),
    "model": ("MODEL", 18),
    "state": ("STATE / PHASE", 12),
    "effort": ("EFFORT", 8),
    "context": ("CONTEXT", 9),
    "requests": ("REQS", 6),
    "active": ("ACTIVE", 7),
    "elapsed": ("ELAPSED", 9),
    "ttft": ("TTFT", 8),
    "tokens": ("TOKENS IN / OUT", 16),
    "cache": ("CACHE", 8),
    "tools": ("TOOLS", 7),
    "retries": ("RETRIES", 8),
    "result": ("RESULT", 13),
}
_MODE_COLUMNS = {
    WidthMode.WIDE: tuple(_COLUMNS),
    WidthMode.MEDIUM: (
        "session", "model", "state", "requests", "active", "elapsed",
        "ttft", "tokens", "tools", "result",
    ),
    WidthMode.NARROW: (
        "session", "state", "elapsed", "ttft", "result",
    ),
}


class _Phase(Protocol):
    value: str


class ConnectionStatusLike(Protocol):
    phase: _Phase
    attempt: int


_REQUEST_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("operation", "OPERATION", 12),
    ("outcome", "OUTCOME", 18),
    ("elapsed", "ELAPSED", 10),
    ("ttft", "TTFT", 9),
    ("tokens", "TOKENS IN / OUT", 17),
    ("tools", "TOOLS", 8),
    ("retries", "RETRIES", 8),
)


class ConnectionHeader(Static):
    """Connection and snapshot summary rendered without markup."""

    def __init__(self, *, id: str = "connection-header") -> None:
        super().__init__(id=id, markup=False)

    def update_status(
        self,
        state: TuiState,
        status: ConnectionStatusLike,
        *,
        visible_count: int,
        now: datetime | None = None,
    ) -> None:
        self.update(
            _header_text(
                state,
                status,
                visible_count=visible_count,
                now=now or datetime.now(UTC),
            )
        )


def _header_text(
    state: TuiState,
    status: ConnectionStatusLike,
    *,
    visible_count: int,
    now: datetime,
) -> Text:
    retry = ""
    if status.phase.value == "reconnecting":
        retry = f" · attempt {status.attempt}"
    process = state.process
    process_text = "PID — · started — · uptime —"
    if process is not None:
        uptime = max(0, int((now - process.started_at).total_seconds()))
        process_text = (
            f"PID {process.pid} · started {process.started_at:%H:%M:%S} "
            f"· uptime {_duration(uptime)}"
        )
    active = sum(item.session.active_requests for item in state.sessions.values())
    sort = state.sort
    summary = (
        f"{status.phase.value.upper()}{retry} · {process_text} · "
        f"cursor {state.cursor} · {visible_count}/{len(state.sessions)} sessions "
        f"· {active} active · sort {sort.field.value} {sort.direction.value}"
    )
    return safe_cell(summary)


class SessionTable(DataTable[Text]):
    """Keyed logical-session table with responsive whole-column sets."""

    def __init__(self, *, id: str = "session-table") -> None:
        super().__init__(id=id, cursor_type="row", zebra_stripes=True)
        self.mode: WidthMode | None = None
        self._order: tuple[str, ...] = ()
        self._state = TuiState.empty()

    @property
    def selected_key(self) -> str | None:
        if not self.rows or self.cursor_row < 0 or self.cursor_row >= len(self.rows):
            return None
        return tuple(self.rows)[self.cursor_row].value

    def select_key(self, identifier: str | None) -> None:
        if identifier is None:
            return
        keys = tuple(key.value for key in self.rows)
        if identifier in keys:
            self.move_cursor(row=keys.index(identifier), column=0, animate=False)

    def sync_state(
        self,
        state: TuiState,
        delta: StateDelta,
        *,
        mode: WidthMode,
        now: datetime | None = None,
    ) -> None:
        selected = state.selected_session_id
        if not delta.selection_changed:
            selected = self.selected_key or selected
        order = visible_session_ids(state)
        rebuild = mode is not self.mode or order != self._order
        self._state = state
        if rebuild:
            self._rebuild(state, order, mode, now)
        else:
            self._update_changed(state, delta.changed_session_ids, now)
        self.select_key(selected)

    def refresh_active(self, state: TuiState, now: datetime) -> None:
        active = {
            identifier
            for identifier in self._order
            if state.sessions[identifier].session.active_requests
        }
        self._update_changed(state, active, now)

    def _rebuild(
        self,
        state: TuiState,
        order: tuple[str, ...],
        mode: WidthMode,
        now: datetime | None,
    ) -> None:
        self.clear(columns=True)
        for key in _MODE_COLUMNS[mode]:
            label, column_width = _COLUMNS[key]
            self.add_column(safe_cell(label), key=key, width=column_width)
        for identifier in order:
            cells = _session_cells(state, identifier, mode, now)
            self.add_row(*cells, key=identifier, height=1)
        self.mode = mode
        self._order = order

    def _update_changed(
        self,
        state: TuiState,
        identifiers: Iterable[str],
        now: datetime | None,
    ) -> None:
        if self.mode is None:
            return
        visible = set(self._order)
        columns = _MODE_COLUMNS[self.mode]
        for identifier in identifiers:
            if identifier not in visible:
                continue
            cells = _session_cells(state, identifier, self.mode, now)
            for column, value in zip(columns, cells, strict=True):
                self.update_cell(identifier, column, value)


def _session_cells(
    state: TuiState,
    identifier: str,
    mode: WidthMode,
    now: datetime | None,
) -> tuple[Text, ...]:
    view = state.sessions[identifier]
    session = view.session
    performance = view.performance
    latest = performance.latest_request
    values = {
        "session": session.id,
        "model": session.model,
        "state": state.phase_for(identifier),
        "effort": session.effort,
        "context": session.context_window if session.context_window is not None else "—",
        "requests": session.requests,
        "active": session.active_requests,
        "elapsed": format_request_elapsed(latest, now=now),
        "ttft": format_latest_metric(None if latest is None else latest.ttft),
        "tokens": format_session_tokens(performance),
        "cache": format_cache_ratio(performance),
        "tools": format_aggregate(performance.tool_calls),
        "retries": format_aggregate(performance.retries),
        "result": session.last_result or "—",
    }
    return tuple(safe_cell(values[column]) for column in _MODE_COLUMNS[mode])


class RequestTable(DataTable[Text]):
    """Active-first, recent-request table keyed by safe request ID."""

    def __init__(self, *, id: str = "request-table") -> None:
        super().__init__(id=id, cursor_type="row", zebra_stripes=True)
        self._request_ids: tuple[str, ...] = ()
        self._columns_ready = False

    @property
    def selected_key(self) -> str | None:
        if not self.rows or self.cursor_row < 0 or self.cursor_row >= len(self.rows):
            return None
        return tuple(self.rows)[self.cursor_row].value

    def sync_state(self, state: TuiState, *, now: datetime | None = None) -> None:
        selected_session = state.selected_session_id
        requests: tuple[RequestPerformanceResponse, ...] = ()
        if selected_session is not None:
            performance = state.sessions[selected_session].performance
            requests = performance.active_requests + performance.recent_requests
        identifiers = tuple(item.id for item in requests)
        selected = state.selected_request_id or self.selected_key
        if identifiers != self._request_ids:
            self._rebuild(requests, now)
        else:
            self._update_requests(requests, now)
        self.select_key(selected)

    def select_key(self, identifier: str | None) -> None:
        if identifier in self._request_ids:
            self.move_cursor(row=self._request_ids.index(identifier), column=0)

    def refresh_active(self, state: TuiState, now: datetime) -> None:
        selected = state.selected_session_id
        if selected is None:
            return
        self._update_requests(state.sessions[selected].performance.active_requests, now)

    def _rebuild(
        self,
        requests: tuple[RequestPerformanceResponse, ...],
        now: datetime | None,
    ) -> None:
        self.clear(columns=True)
        for key, label, column_width in _REQUEST_COLUMNS:
            self.add_column(safe_cell(label), key=key, width=column_width)
        for request in requests:
            self.add_row(*_request_cells(request, now), key=request.id, height=1)
        self._columns_ready = True
        self._request_ids = tuple(item.id for item in requests)

    def _update_requests(
        self,
        requests: Iterable[RequestPerformanceResponse],
        now: datetime | None,
    ) -> None:
        if not self._columns_ready:
            return
        columns = tuple(item[0] for item in _REQUEST_COLUMNS)
        retained = set(self._request_ids)
        for request in requests:
            if request.id not in retained:
                continue
            for column, value in zip(
                columns, _request_cells(request, now), strict=True
            ):
                self.update_cell(request.id, column, value)


def _request_cells(
    request: RequestPerformanceResponse,
    now: datetime | None,
) -> tuple[Text, ...]:
    tokens = f"{metric_detail(request.input_tokens)} / {metric_detail(request.output_tokens)}"
    values = (
        request.operation,
        request.outcome,
        format_request_elapsed(request, now=now),
        format_latest_metric(request.ttft),
        tokens,
        metric_detail(request.tool_calls),
        metric_detail(request.retries),
    )
    return tuple(safe_cell(value) for value in values)



def _duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
