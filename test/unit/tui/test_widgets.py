from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rich.text import Text
from textual.app import App, ComposeResult

from claude_code_proxy.tui.app import ConnectionPhase, ConnectionStatus
from claude_code_proxy.tui.details import SessionDetails
from claude_code_proxy.tui.state import StateDelta, TuiState, apply_stream_event
from claude_code_proxy.tui.widgets import (
    ConnectionHeader,
    RequestTable,
    SessionTable,
    WidthMode,
)
from test.unit.tui.support import reset, view, with_agent, with_requests


class WidgetHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield ConnectionHeader()
        yield SessionTable()
        yield SessionDetails()
        yield RequestTable()


def state_with(*identifiers: str) -> TuiState:
    state, _ = apply_stream_event(
        TuiState.empty(), reset(*(view(item) for item in identifiers))
    )
    return state


@pytest.mark.parametrize(
    ("width", "mode", "columns"),
    [
        (
            140,
            WidthMode.WIDE,
            {
                "session", "model", "state", "effort", "context",
                "requests", "active", "elapsed", "ttft", "tokens",
                "cache", "tools", "retries", "result",
            },
        ),
        (
            105,
            WidthMode.MEDIUM,
            {
                "session", "model", "state", "requests", "active",
                "elapsed", "ttft", "tokens", "tools", "result",
            },
        ),
        (
            80,
            WidthMode.NARROW,
            {"session", "state", "elapsed", "ttft", "result"},
        ),
    ],
)
async def test_session_table_uses_responsive_whole_columns(
    width: int, mode: WidthMode, columns: set[str]
) -> None:
    app = WidgetHarness()
    async with app.run_test(size=(width, 32)) as pilot:
        table = app.query_one(SessionTable)
        table.sync_state(
            state_with("safe-a"), StateDelta(replace_all=True), mode=mode
        )
        await pilot.pause()

        assert table.mode is mode
        assert {key.value for key in table.columns} == columns
        assert {key.value for key in table.rows} == {"safe-a"}


async def test_session_table_has_one_stable_key_per_session_and_preserves_selection() -> None:
    app = WidgetHarness()
    initial, _ = apply_stream_event(
        TuiState.empty(),
        reset(
            view("safe-a", first_seen="2026-01-02T03:04:04Z"),
            view("safe-b", first_seen="2026-01-02T03:04:05Z"),
        ),
    )
    async with app.run_test(size=(140, 40)) as pilot:
        table = app.query_one(SessionTable)
        table.sync_state(
            initial, StateDelta(replace_all=True), mode=WidthMode.WIDE
        )
        table.select_key("safe-b")
        updated, _ = apply_stream_event(
            initial,
            reset(
                view("safe-b", first_seen="2026-01-02T03:04:03Z"),
                view("safe-a", first_seen="2026-01-02T03:04:04Z"),
                sequence=8,
            ),
        )
        table.sync_state(
            updated, StateDelta(order_changed=True), mode=WidthMode.WIDE
        )
        await pilot.pause()

        assert tuple(key.value for key in table.rows) == ("safe-b", "safe-a")
        assert table.selected_key == "safe-b"


async def test_session_table_updates_reset_cells_without_rebuilding_columns() -> None:
    app = WidgetHarness()
    initial = state_with("safe-a")
    async with app.run_test(size=(140, 40)):
        table = app.query_one(SessionTable)
        table.sync_state(
            initial, StateDelta(replace_all=True), mode=WidthMode.WIDE
        )
        original_columns = tuple(table.columns.values())
        updated, delta = apply_stream_event(
            initial,
            reset(view("safe-a", model="changed-model"), sequence=8),
        )
        table.sync_state(updated, delta, mode=WidthMode.WIDE)

        assert all(
            current is original
            for current, original in zip(
                table.columns.values(), original_columns, strict=True
            )
        )
        assert table.get_cell("safe-a", "model").plain == "changed-model"


async def test_session_table_cells_are_literal_safe_rich_text() -> None:
    base = view("safe-[bold]id")
    poisoned = base.model_copy(
        update={
            "session": base.session.model_copy(
                update={"model": "[bold]model[/bold]"}
            )
        }
    )
    state, _ = apply_stream_event(TuiState.empty(), reset(poisoned))
    app = WidgetHarness()
    async with app.run_test(size=(140, 40)) as pilot:
        table = app.query_one(SessionTable)
        table.sync_state(
            state, StateDelta(replace_all=True), mode=WidthMode.WIDE
        )
        await pilot.pause()
        model = table.get_cell("safe-[bold]id", "model")

        assert isinstance(model, Text)
        assert model.plain == "[bold]model[/bold]"
        assert all(span.style != "bold" for span in model.spans)
        assert model.no_wrap


async def test_request_table_orders_active_before_recent_and_uses_safe_keys() -> None:
    item = with_requests(
        view("safe-session"),
        active_ids=("active-b", "active-a"),
        recent_ids=("recent-a", "recent-b"),
    )
    state, _ = apply_stream_event(TuiState.empty(), reset(item))
    app = WidgetHarness()
    async with app.run_test(size=(140, 40)) as pilot:
        table = app.query_one(RequestTable)
        table.sync_state(state)
        await pilot.pause()

        assert tuple(key.value for key in table.rows) == (
            "active-b", "active-a", "recent-a", "recent-b"
        )
        assert {key.value for key in table.columns} == {
            "operation", "outcome", "elapsed", "ttft", "tokens",
            "tools", "retries",
        }


async def test_header_shows_connection_retry_process_visibility_and_sort() -> None:
    state = state_with("safe-a", "safe-b")
    app = WidgetHarness()
    async with app.run_test(size=(140, 40)) as pilot:
        header = app.query_one(ConnectionHeader)
        header.update_status(
            state,
            ConnectionStatus(ConnectionPhase.RECONNECTING, 3),
            visible_count=1,
            now=datetime(2026, 1, 2, 3, 5, tzinfo=UTC),
        )
        await pilot.pause()

        rendered = str(header.render())
        for expected in (
            "RECONNECTING", "attempt 3", "PID 42", "cursor 7",
            "1/2 sessions", "sort baseline ascending",
        ):
            assert expected in rendered


async def test_session_details_include_safe_agent_hierarchy_but_table_does_not() -> None:
    item = with_agent(view("safe-session"), "safe-agent")
    state, _ = apply_stream_event(TuiState.empty(), reset(item))
    app = WidgetHarness()
    async with app.run_test(size=(140, 40)) as pilot:
        table = app.query_one(SessionTable)
        details = app.query_one(SessionDetails)
        table.sync_state(
            state, StateDelta(replace_all=True), mode=WidthMode.WIDE
        )
        details.sync_state(state)
        await pilot.pause()

        assert {key.value for key in table.rows} == {"safe-session"}
        rendered = str(details.render())
        assert "safe-agent" in rendered
        assert "parent" in rendered
        assert "reasoning" in rendered.casefold()
