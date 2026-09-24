from __future__ import annotations

from concurrent.futures import CancelledError
from datetime import UTC, datetime, timedelta
from pathlib import Path
import threading

import pytest
from textual.widgets import Select

from claude_code_proxy.tui.app import (
    AppResult,
    ConnectionPhase,
    ConnectionStatus,
    TuiApp,
)
from claude_code_proxy.tui.details import RequestDetails, SessionDetails
from claude_code_proxy.tui.screens import (
    FilterScreen,
    HelpScreen,
    RequestDetailScreen,
    SearchScreen,
    SessionDetailScreen,
    SortScreen,
)
from claude_code_proxy.tui.state import SortField
from claude_code_proxy.tui.widgets import RequestTable, SessionTable, WidthMode
from test.unit.tui.support import reset, view, with_requests


class InertPump:
    def __init__(self) -> None:
        self.stops = 0

    def run(self, on_event, on_status) -> AppResult:
        return AppResult(0)

    def stop(self) -> None:
        self.stops += 1


def app_with_data(*identifiers: str, connected: bool = True) -> TuiApp:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    if identifiers:
        app.pending.offer(reset(*(view(item) for item in identifiers)))
    if not connected:
        app.connection_status = ConnectionStatus(ConnectionPhase.DISCONNECTED)
    return app


@pytest.mark.parametrize(
    ("size", "mode", "details_visible"),
    [
        ((140, 40), WidthMode.WIDE, True),
        ((105, 32), WidthMode.MEDIUM, True),
        ((80, 24), WidthMode.NARROW, False),
        ((140, 18), WidthMode.WIDE, False),
    ],
)
async def test_responsive_layout(size, mode, details_visible) -> None:
    app = app_with_data("safe-a")
    async with app.run_test(size=size) as pilot:
        app.drain_pending()
        await pilot.pause()

        assert app.width_mode is mode
        assert app.query_one(SessionDetails).display is details_visible
        assert app.query_one(RequestTable).display is details_visible


async def test_connected_only_after_reset_and_disconnect_keeps_dimmed_rows() -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    async with app.run_test(size=(140, 40)) as pilot:
        assert app.connection_status.phase is ConnectionPhase.CONNECTING
        app.pending.offer(reset(view("safe-a")))
        app.drain_pending()
        assert app.connection_status.phase is ConnectionPhase.CONNECTED
        app.accept_status(ConnectionStatus(ConnectionPhase.DISCONNECTED, 1))
        await pilot.pause()

        assert {
            key.value for key in app.query_one(SessionTable).rows
        } == {"safe-a"}
        assert app.screen.has_class("disconnected")


async def test_active_elapsed_refreshes_once_per_second_without_events() -> None:
    item = view("safe-active", state="active")
    app = app_with_data()
    app.pending.offer(reset(item))
    async with app.run_test(size=(140, 40)) as pilot:
        app.drain_pending()
        table = app.query_one(SessionTable)
        before = table.get_cell("safe-active", "elapsed").plain
        app.refresh_clock(now=datetime.now(UTC) + timedelta(seconds=2))
        await pilot.pause()
        after = table.get_cell("safe-active", "elapsed").plain

        assert before != after


async def test_search_filters_locally_while_typing_and_clear_restores() -> None:
    app = app_with_data("alpha", "beta")
    async with app.run_test(size=(140, 40)) as pilot:
        app.drain_pending()
        await pilot.press("/")
        assert isinstance(app.screen, SearchScreen)
        await pilot.press("b", "e")
        assert tuple(
            key.value for key in app.query_one(SessionTable).rows
        ) == ("beta",)
        await pilot.press("enter")
        await pilot.press("c")
        assert tuple(
            key.value for key in app.query_one(SessionTable).rows
        ) == ("alpha", "beta")


@pytest.mark.parametrize(
    ("key", "screen_type"),
    [("f", FilterScreen), ("s", SortScreen), ("?", HelpScreen)],
)
async def test_overlay_bindings(key: str, screen_type: type) -> None:
    app = app_with_data("safe-a")
    async with app.run_test(size=(140, 40)) as pilot:
        app.drain_pending()
        await pilot.press(key)
        assert isinstance(app.screen, screen_type)
        await pilot.press("escape")
        assert not isinstance(app.screen, screen_type)


async def test_sort_overlay_applies_selected_field_and_direction() -> None:
    app = app_with_data("safe-a")
    async with app.run_test(size=(140, 40)) as pilot:
        app.drain_pending()
        await pilot.press("s")
        screen = app.screen
        assert isinstance(screen, SortScreen)
        screen.query_one("#sort-field", Select).value = "model"
        screen.query_one("#sort-direction", Select).value = "descending"
        await pilot.click("#sort-apply")

        assert app.state.sort.field is SortField.MODEL
        assert app.state.sort.direction.value == "descending"
        assert not isinstance(app.screen, SortScreen)


async def test_sort_overlay_exposes_all_sort_fields_and_applies_direction() -> None:
    app = app_with_data("safe-b", "safe-a")
    async with app.run_test(size=(140, 40)) as pilot:
        app.drain_pending()
        app.apply_sort(SortField.SESSION_ID.value, "descending")
        await pilot.pause()

        assert tuple(
            key.value for key in app.query_one(SessionTable).rows
        ) == ("safe-b", "safe-a")
        assert "session_id descending" in str(
            app.query_one("#connection-header").render()
        )


async def test_arrow_jk_tab_shift_tab_enter_and_escape_contract() -> None:
    app = app_with_data("safe-a", "safe-b")
    async with app.run_test(size=(140, 40)) as pilot:
        app.drain_pending()
        table = app.query_one(SessionTable)
        assert table.selected_key == "safe-a"
        await pilot.press("j")
        assert table.selected_key == "safe-b"
        await pilot.press("k")
        assert table.selected_key == "safe-a"
        await pilot.press("tab")
        assert isinstance(app.focused, RequestTable)
        await pilot.press("shift+tab")
        assert isinstance(app.focused, SessionTable)
        await pilot.press("enter")
        assert isinstance(app.focused, RequestTable)
        await pilot.press("escape")
        assert isinstance(app.focused, SessionTable)


async def test_narrow_enter_opens_full_width_session_screen() -> None:
    app = app_with_data("safe-a")
    async with app.run_test(size=(80, 24)) as pilot:
        app.drain_pending()
        await pilot.press("enter")
        assert isinstance(app.screen, SessionDetailScreen)
        await pilot.press("escape")


async def test_short_terminal_enter_opens_full_width_session_screen() -> None:
    app = app_with_data("safe-a")
    async with app.run_test(size=(140, 18)) as pilot:
        app.drain_pending()
        await pilot.press("enter")

        assert isinstance(app.screen, SessionDetailScreen)


async def test_open_session_detail_receives_authoritative_reset() -> None:
    app = app_with_data("safe-a")
    async with app.run_test(size=(80, 24)) as pilot:
        app.drain_pending()
        await pilot.press("enter")
        screen = app.screen
        assert isinstance(screen, SessionDetailScreen)

        app.pending.offer(
            reset(view("safe-a", model="replacement-model"), sequence=8)
        )
        app.drain_pending()
        await pilot.pause()

        rendered = str(screen.query_one(SessionDetails).render())
        assert "replacement-model" in rendered
        assert "provider-model" not in rendered


async def test_open_session_detail_refreshes_active_elapsed() -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    app.pending.offer(reset(view("safe-active", state="active")))
    async with app.run_test(size=(80, 24)) as pilot:
        app.drain_pending()
        await pilot.press("enter")
        screen = app.screen
        assert isinstance(screen, SessionDetailScreen)
        table = screen.query_one(RequestTable)
        details = screen.query_one(RequestDetails)
        before_row = table.get_cell("request-safe-active", "elapsed").plain
        before_detail = str(details.render())

        app.refresh_clock(now=datetime.now(UTC) + timedelta(seconds=2))
        await pilot.pause()

        assert table.get_cell("request-safe-active", "elapsed").plain != before_row
        assert str(details.render()) != before_detail


async def test_session_detail_request_selection_updates_visible_detail() -> None:
    item = with_requests(
        view("safe-a"), recent_ids=("recent-a", "recent-b")
    )
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    app.pending.offer(reset(item))
    async with app.run_test(size=(80, 24)) as pilot:
        app.drain_pending()
        await pilot.press("enter")
        screen = app.screen
        assert isinstance(screen, SessionDetailScreen)
        table = screen.query_one(RequestTable)
        details = screen.query_one(RequestDetails)
        assert "recent-a" in str(details.render())

        table.action_cursor_down()
        await pilot.pause()

        assert app.state.selected_request_id == "recent-b"
        assert "recent-b" in str(details.render())


async def test_open_request_detail_receives_reset_and_clock_refresh() -> None:
    active = with_requests(view("safe-a"), active_ids=("request-a",))
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    app.pending.offer(reset(active))
    async with app.run_test(size=(140, 40)) as pilot:
        app.drain_pending()
        screen = RequestDetailScreen(app.state)
        app.push_screen(screen)
        await pilot.pause()
        details = screen.query_one(RequestDetails)
        before = str(details.render())

        app.refresh_clock(now=datetime.now(UTC) + timedelta(seconds=2))
        await pilot.pause()
        assert str(details.render()) != before

        completed = with_requests(
            view("safe-a"), recent_ids=("request-a",)
        )
        app.pending.offer(reset(completed, sequence=8))
        app.drain_pending()
        await pilot.pause()

        assert "completed" in str(details.render())


@pytest.mark.parametrize("key", ["q", "ctrl+c"])
async def test_quit_keys_stop_stream_and_exit_zero(key: str) -> None:
    pump = InertPump()
    app = TuiApp(
        Path("/safe/control.sock"), pump=pump, start_stream=False
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(key)
        await pilot.pause()

    assert pump.stops >= 1
    assert app.return_value == AppResult(0)


def test_stream_callback_race_drops_delivery_after_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    barrier = threading.Barrier(2)
    delivered: list[str] = []
    failures: list[BaseException] = []

    def pause_then_schedule(callback, *args) -> None:
        barrier.wait(timeout=1)
        barrier.wait(timeout=1)
        callback(*args)

    monkeypatch.setattr(app, "call_from_thread", pause_then_schedule)

    def invoke_from_stream() -> None:
        try:
            app._call_from_stream(delivered.append, "late")
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=invoke_from_stream)
    worker.start()
    barrier.wait(timeout=1)
    app.stop_stream()
    barrier.wait(timeout=1)
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert failures == []
    assert delivered == []


def test_queued_stream_callback_rechecks_stop_before_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    scheduled: list[tuple[object, tuple[object, ...]]] = []
    delivered: list[str] = []
    monkeypatch.setattr(
        app,
        "call_from_thread",
        lambda callback, *args: scheduled.append((callback, args)),
    )

    app._call_from_stream(delivered.append, "queued")
    app.stop_stream()
    callback, args = scheduled.pop()
    callback(*args)

    assert delivered == []


def test_stream_callback_delivers_before_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    delivered: list[str] = []
    monkeypatch.setattr(
        app,
        "call_from_thread",
        lambda callback, *args: callback(*args),
    )

    app._call_from_stream(delivered.append, "live")

    assert delivered == ["live"]


@pytest.mark.parametrize("error_type", [CancelledError, RuntimeError])
def test_stream_callback_ignores_scheduler_shutdown_error_after_stop(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )

    def stop_then_fail(callback, *args) -> None:
        app.stop_stream()
        raise error_type("app loop stopped")

    monkeypatch.setattr(app, "call_from_thread", stop_then_fail)

    app._call_from_stream(lambda: None)


@pytest.mark.parametrize("error_type", [CancelledError, RuntimeError])
def test_stream_callback_surfaces_scheduler_error_before_stop(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )

    def fail(callback, *args) -> None:
        raise error_type("unexpected scheduler failure")

    monkeypatch.setattr(app, "call_from_thread", fail)

    with pytest.raises(error_type, match="unexpected scheduler failure"):
        app._call_from_stream(lambda: None)
