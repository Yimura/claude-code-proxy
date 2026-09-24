"""Literal-content overlays and detail screens for the TUI."""

from __future__ import annotations

from collections.abc import Callable

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Label, Select, Static

from .details import RequestDetails, SessionDetails
from .formatting import safe_cell
from .state import SortDirection, SortField, TuiState
from .widgets import RequestTable


class _DismissableModal(ModalScreen[None]):
    BINDINGS = [("escape", "close", "Back")]

    def action_close(self) -> None:
        self.dismiss()


class _DismissableScreen(Screen[None]):
    BINDINGS = [("escape", "close", "Back")]

    def action_close(self) -> None:
        self.dismiss()


class SearchScreen(_DismissableModal):
    """Quick-search overlay that applies each edit locally."""

    def __init__(
        self,
        value: str,
        callback: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._initial_value = value
        self._callback = callback

    def compose(self) -> ComposeResult:
        with Vertical(classes="overlay-card"):
            yield Label(safe_cell("Quick search"))
            yield Input(
                value=self._initial_value,
                placeholder="safe ID, model, state, provider…",
                id="search-value",
            )

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Changed)
    def search_changed(self, event: Input.Changed) -> None:
        self._callback(event.value)

    @on(Input.Submitted)
    def search_submitted(self) -> None:
        self.dismiss()


_FILTER_OPTIONS = tuple(
    (safe_cell(name), name)
    for name in (
        "id", "session_id", "state", "provider", "transport", "model", "effort"
    )
)


class FilterScreen(_DismissableModal):
    """Field filter overlay with transient masked raw-session input."""

    def __init__(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        super().__init__()
        self._callback = callback
        self._selected_field = "id"

    def compose(self) -> ComposeResult:
        with Vertical(classes="overlay-card"):
            yield Label(safe_cell("Add filter · same field OR, different fields AND"))
            yield Select(
                _FILTER_OPTIONS,
                value="id",
                allow_blank=False,
                id="filter-field",
            )
            yield Input(placeholder="value", id="filter-value")
            yield Static(
                safe_cell("Enter applies · Escape cancels"),
                markup=False,
                classes="overlay-hint",
            )

    def on_mount(self) -> None:
        self.query_one("#filter-value", Input).focus()

    def select_field(self, field: str) -> None:
        if field not in {value for _, value in _FILTER_OPTIONS}:
            return
        self._selected_field = field
        selector = self.query_one("#filter-field", Select)
        selector.value = field
        self.query_one("#filter-value", Input).password = field == "session_id"

    @on(Select.Changed, "#filter-field")
    def field_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, str):
            self._selected_field = event.value
            self.query_one("#filter-value", Input).password = (
                event.value == "session_id"
            )

    @on(Input.Submitted, "#filter-value")
    def input_submitted(self) -> None:
        self.submit_filter()

    def submit_filter(self) -> None:
        field = self.query_one("#filter-value", Input)
        value = field.value
        field.clear()
        if not value:
            return
        self._callback(self._selected_field, value)
        self.dismiss()


_SORT_OPTIONS = tuple((safe_cell(item.value), item.value) for item in SortField)
_DIRECTION_OPTIONS = tuple(
    (safe_cell(item.value), item.value) for item in SortDirection
)


class SortScreen(_DismissableModal):
    """Sort field and direction overlay."""

    def __init__(
        self,
        field: SortField,
        direction: SortDirection,
        callback: Callable[[str, str], None],
    ) -> None:
        super().__init__()
        self._field = field
        self._direction = direction
        self._callback = callback

    def compose(self) -> ComposeResult:
        with Vertical(classes="overlay-card"):
            yield Label(safe_cell("Sort sessions"))
            yield Select(
                _SORT_OPTIONS,
                value=self._field.value,
                allow_blank=False,
                id="sort-field",
            )
            yield Select(
                _DIRECTION_OPTIONS,
                value=self._direction.value,
                allow_blank=False,
                id="sort-direction",
            )
            yield Button("Apply", id="sort-apply", variant="primary")
            yield Static(
                safe_cell("Escape cancels"),
                markup=False,
                classes="overlay-hint",
            )

    @on(Button.Pressed, "#sort-apply")
    def apply_pressed(self) -> None:
        self.submit_sort()

    def submit_sort(self) -> None:
        field = self.query_one("#sort-field", Select).value
        direction = self.query_one("#sort-direction", Select).value
        if isinstance(field, str) and isinstance(direction, str):
            self._callback(field, direction)
            self.dismiss()


_HELP = (
    "Keys\n"
    "  ↑/↓ or j/k  select\n"
    "  Tab/Shift+Tab  switch panes\n"
    "  Enter  details\n"
    "  Escape  back/close\n"
    "  /  quick search\n"
    "  f  filter\n"
    "  s  sort\n"
    "  c  clear search and filters\n"
    "  ?  help\n"
    "  q or Ctrl+C  quit"
)


class HelpScreen(_DismissableModal):
    """Keyboard reference using literal content."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="overlay-card help-card"):
            yield Static(safe_cell(_HELP), markup=False)


class SessionDetailScreen(_DismissableScreen):
    """Full-width narrow-mode session view."""

    def __init__(self, state: TuiState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield SessionDetails(id="screen-session-details")
        yield RequestTable(id="screen-request-table")
        yield RequestDetails(id="screen-request-details")

    def on_mount(self) -> None:
        self.query_one(SessionDetails).sync_state(self._state)
        self.query_one(RequestTable).sync_state(self._state)
        self.query_one(RequestDetails).sync_state(self._state)
        self.query_one(RequestTable).focus()


class RequestDetailScreen(_DismissableModal):
    """Full request metric and safe failure detail overlay."""

    def __init__(self, state: TuiState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        with Vertical(classes="overlay-card request-detail-card"):
            details = RequestDetails()
            details.sync_state(self._state)
            yield details
