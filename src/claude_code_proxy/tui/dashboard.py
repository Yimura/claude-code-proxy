"""Textual application orchestration for the live analytics dashboard."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Footer, Static

from ..cli_common import terminal_text
from ..control.schemas import PerformanceResetResponse
from . import app as app_core
from .details import RequestDetails, SessionDetails
from .screens import (
    FilterScreen,
    HelpScreen,
    RequestDetailScreen,
    SearchScreen,
    SessionDetailScreen,
    SortScreen,
)
from .state import (
    FilterField,
    FilterTerm,
    SortDirection,
    SortField,
    SortSpec,
    StateDelta,
    TuiState,
    apply_stream_event,
    clear_query,
    select_request,
    select_session,
    visible_session_ids,
    with_filter,
    with_search,
    with_sort,
)
from .widgets import (
    ConnectionHeader,
    RequestTable,
    SessionTable,
    WidthMode,
    width_mode,
)

_SHORT_HEIGHT = 22
_FILTER_FIELDS = frozenset(
    {"id", "state", "provider", "transport", "model", "effort"}
)


class TuiApp(App[app_core.AppResult]):
    """Live, responsive performance dashboard."""

    CSS_PATH = "tui.tcss"
    TITLE = "Claude Code Proxy · Live Sessions"
    BINDINGS = [
        ("q", "quit_clean", "Quit"),
        ("ctrl+c", "quit_clean", "Quit"),
        ("/", "search", "Search"),
        ("f", "filter", "Filter"),
        ("s", "sort", "Sort"),
        ("c", "clear", "Clear"),
        ("?", "help", "Help"),
        ("j", "select_next", "Down"),
        ("k", "select_previous", "Up"),
        ("tab", "focus_next", "Next pane"),
        ("shift+tab", "focus_previous", "Previous pane"),
        ("enter", "details", "Details"),
        ("escape", "back", "Back"),
    ]

    def __init__(
        self,
        socket_path: Path,
        *,
        pump: app_core.ConnectionPump | None = None,
        start_stream: bool = True,
    ) -> None:
        super().__init__()
        self.socket_path = socket_path
        self.pending = app_core.PendingEvents()
        self.state = TuiState.empty()
        self.connection_status = app_core.ConnectionStatus(
            app_core.ConnectionPhase.CONNECTING
        )
        self.width_mode = WidthMode.NARROW
        self._pump = pump or app_core.ConnectionPump(socket_path)
        self._start_stream = start_stream
        self._stream_stopped = False

    def compose(self) -> ComposeResult:
        yield ConnectionHeader()
        with Vertical(id="dashboard"):
            yield Static("", id="empty-state", markup=False)
            yield SessionTable()
            with Container(id="details-pane"):
                yield SessionDetails()
                yield RequestTable()
                yield RequestDetails()
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.1, self.drain_pending)
        self.set_interval(1.0, self.refresh_clock)
        self._apply_responsive_layout()
        self._refresh_all(StateDelta(replace_all=True))
        self.query_one(SessionTable).focus()
        if self._start_stream:
            self._run_stream()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_responsive_layout()
        self._refresh_all(StateDelta(order_changed=True))

    def on_unmount(self) -> None:
        self.stop_stream()

    @work(thread=True, name="performance-stream", exclusive=True, exit_on_error=False)
    def _run_stream(self) -> None:
        result = self._pump.run(
            self.pending.offer,
            lambda status: self.call_from_thread(self.accept_status, status),
        )
        self.call_from_thread(self._finish_stream, result)

    def stop_stream(self) -> None:
        """Idempotently stop the control stream and close its active client."""
        if self._stream_stopped:
            return
        self._stream_stopped = True
        self._pump.stop()

    def drain_pending(self) -> None:
        """Apply the coalesced stream batch in sequence and redraw once."""
        merged = StateDelta()
        for event in self.pending.drain():
            self.state, delta = apply_stream_event(self.state, event)
            merged = _merge_delta(merged, delta)
            if isinstance(event, PerformanceResetResponse):
                self.connection_status = app_core.ConnectionStatus(
                    app_core.ConnectionPhase.CONNECTED
                )
        if merged != StateDelta():
            self._refresh_all(merged)

    def accept_status(self, status: app_core.ConnectionStatus) -> None:
        """Apply connection state while retaining stale snapshot rows."""
        self.connection_status = status
        self._set_connection_class()
        self._refresh_header()
        self._refresh_empty_state()

    def refresh_clock(self, *, now: datetime | None = None) -> None:
        """Refresh local wall-clock values without rebuilding rows."""
        observed = now or datetime.now(UTC)
        self.query_one(SessionTable).refresh_active(self.state, observed)
        self.query_one(RequestTable).refresh_active(self.state, observed)
        self._refresh_header(observed)

    def _refresh_all(self, delta: StateDelta) -> None:
        sessions = self.query_one(SessionTable)
        sessions.sync_state(self.state, delta, mode=self.width_mode)
        sessions.select_key(self.state.selected_session_id)
        self.query_one(SessionDetails).sync_state(self.state)
        self.query_one(RequestTable).sync_state(self.state)
        self.query_one(RequestDetails).sync_state(self.state)
        self._refresh_header()
        self._refresh_empty_state()
        self._set_connection_class()

    def _apply_responsive_layout(self) -> None:
        self.width_mode = width_mode(self.size.width)
        show_details = (
            self.width_mode is not WidthMode.NARROW
            and self.size.height >= _SHORT_HEIGHT
        )
        self.query_one("#details-pane").display = show_details
        self.query_one(SessionDetails).display = show_details
        self.query_one(RequestTable).display = show_details
        self.query_one(RequestDetails).display = show_details
        for mode in WidthMode:
            self.screen.set_class(mode is self.width_mode, mode.value)
        self.screen.set_class(self.size.height < _SHORT_HEIGHT, "short")

    def _refresh_header(self, now: datetime | None = None) -> None:
        self.query_one(ConnectionHeader).update_status(
            self.state,
            self.connection_status,
            visible_count=len(visible_session_ids(self.state)),
            now=now,
        )

    def _refresh_empty_state(self) -> None:
        empty = self.query_one("#empty-state", Static)
        message = _empty_message(self.state, self.connection_status)
        empty.update(terminal_text(message))
        empty.display = bool(message)

    def _set_connection_class(self) -> None:
        disconnected = self.connection_status.phase in {
            app_core.ConnectionPhase.DISCONNECTED,
            app_core.ConnectionPhase.RECONNECTING,
        }
        self.screen.set_class(disconnected, "disconnected")

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if isinstance(event.data_table, SessionTable):
            self._select_session(cast(str, event.row_key.value))
        if isinstance(event.data_table, RequestTable):
            self._select_request(cast(str, event.row_key.value))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if isinstance(event.data_table, SessionTable):
            self.action_details()
            return
        if isinstance(event.data_table, RequestTable):
            self.push_screen(RequestDetailScreen(self.state))

    def _select_session(self, identifier: str) -> None:
        updated = select_session(self.state, identifier)
        if updated is self.state:
            return
        self.state = updated
        self.query_one(SessionDetails).sync_state(self.state)
        self.query_one(RequestTable).sync_state(self.state)
        self.query_one(RequestDetails).sync_state(self.state)

    def _select_request(self, identifier: str) -> None:
        self.state = select_request(self.state, identifier)
        self.query_one(RequestDetails).sync_state(self.state)

    def action_select_next(self) -> None:
        table = _focused_table(self)
        if table is not None:
            table.action_cursor_down()

    def action_select_previous(self) -> None:
        table = _focused_table(self)
        if table is not None:
            table.action_cursor_up()

    def action_details(self) -> None:
        if isinstance(self.focused, RequestTable):
            self.push_screen(RequestDetailScreen(self.state))
            return
        if self.state.selected_session_id is None:
            return
        if self.width_mode is WidthMode.NARROW:
            self.push_screen(SessionDetailScreen(self.state))
            return
        self.query_one(RequestTable).focus()

    def action_back(self) -> None:
        if isinstance(self.focused, RequestTable):
            self.query_one(SessionTable).focus()

    def action_search(self) -> None:
        self.push_screen(SearchScreen(self.state.search, self.apply_search))

    def apply_search(self, value: str) -> None:
        self.state = with_search(self.state, value)
        self._refresh_all(StateDelta(order_changed=True))

    def action_filter(self) -> None:
        self.push_screen(FilterScreen(self.apply_filter))

    def apply_filter(self, field: str, value: str) -> None:
        if field == "session_id":
            self.run_worker(
                self.resolve_session_filter(value),
                name="resolve-session-filter",
                exclusive=False,
                exit_on_error=False,
            )
            return
        if field not in _FILTER_FIELDS:
            return
        term = FilterTerm(cast(FilterField, field), value)
        self.state = with_filter(self.state, term)
        self._refresh_all(StateDelta(order_changed=True))

    async def resolve_session_filter(self, raw_value: str) -> None:
        """Resolve one raw identifier without retaining or displaying it."""
        try:
            identifiers = await asyncio.to_thread(
                self._resolve_safe_session_ids, raw_value
            )
        except Exception:
            self.notify("Session lookup unavailable", severity="error", markup=False)
            return
        if not identifiers:
            self.notify("No session matched", severity="warning", markup=False)
            return
        for identifier in identifiers:
            self.state = with_filter(
                self.state,
                FilterTerm(
                    "id", identifier, exact_id=True, source="session_id"
                ),
            )
        self._refresh_all(StateDelta(order_changed=True))

    def _resolve_safe_session_ids(self, raw_value: str) -> tuple[str, ...]:
        with app_core.ControlClient(self.socket_path) as client:
            result = client.performance(("session_id=" + raw_value,))
        return tuple(item.session.id for item in result.sessions)

    def action_sort(self) -> None:
        self.push_screen(
            SortScreen(
                self.state.sort.field,
                self.state.sort.direction,
                self.apply_sort,
            )
        )

    def apply_sort(self, field: str, direction: str) -> None:
        try:
            spec = SortSpec(SortField(field), SortDirection(direction))
        except ValueError:
            return
        self.state = with_sort(self.state, spec)
        self._refresh_all(StateDelta(order_changed=True))

    def action_clear(self) -> None:
        self.state = clear_query(self.state)
        self._refresh_all(StateDelta(order_changed=True))

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_quit_clean(self) -> None:
        self.stop_stream()
        self.exit(app_core.AppResult(0))

    def _finish_stream(self, result: app_core.AppResult) -> None:
        self.stop_stream()
        self.exit(result)


def _merge_delta(left: StateDelta, right: StateDelta) -> StateDelta:
    return StateDelta(
        changed_session_ids=left.changed_session_ids | right.changed_session_ids,
        order_changed=left.order_changed or right.order_changed,
        selection_changed=left.selection_changed or right.selection_changed,
        replace_all=left.replace_all or right.replace_all,
    )


def _focused_table(app: TuiApp) -> DataTable | None:
    focused = app.focused
    if isinstance(focused, (SessionTable, RequestTable)):
        return focused
    return None


def _empty_message(
    state: TuiState,
    status: app_core.ConnectionStatus,
) -> str:
    if state.sessions and not visible_session_ids(state):
        return "No sessions match the current search and filters"
    if state.sessions:
        return ""
    messages = {
        app_core.ConnectionPhase.CONNECTING: "Connecting to performance stream…",
        app_core.ConnectionPhase.CONNECTED: "No sessions retained",
        app_core.ConnectionPhase.DISCONNECTED: "Disconnected · no retained sessions",
        app_core.ConnectionPhase.RECONNECTING: "Reconnecting · no retained sessions",
    }
    return messages[status.phase]


def run_tui(socket_path: Path) -> app_core.AppResult:
    """Run Textual and return only after terminal restoration."""
    result = TuiApp(socket_path).run()
    if isinstance(result, app_core.AppResult):
        return result
    return app_core.AppResult(0)
