from __future__ import annotations

from pathlib import Path

from textual.widgets import Input

from claude_code_proxy.tui.app import AppResult, TuiApp
from claude_code_proxy.tui.screens import FilterScreen
from claude_code_proxy.tui.state import FilterTerm, TuiState
from test.unit.tui.support import reset, view

RAW = "raw-session-marker-[bold]-\x1b\n"
SAFE = "safe-public-id"


class InertPump:
    def run(self, on_event, on_status) -> AppResult:
        return AppResult(0)

    def stop(self) -> None:
        pass


async def test_session_id_filter_is_masked_and_cleared_before_dismiss() -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    observed: list[str] = []
    async with app.run_test(size=(100, 30)) as pilot:
        screen = FilterScreen(
            lambda field, value: observed.append(value)
        )
        app.push_screen(screen)
        await pilot.pause()
        screen.select_field("session_id")
        field = screen.query_one("#filter-value", Input)
        field.value = RAW
        assert field.password is True
        assert RAW not in str(field.render())
        screen.submit_filter()
        assert field.value == ""
        await pilot.pause()

    assert observed == [RAW]
    assert RAW not in repr(screen)


async def test_switching_from_session_id_scrubs_before_unmasking() -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    callbacks: list[tuple[str, str]] = []
    async with app.run_test(size=(100, 30)) as pilot:
        screen = FilterScreen(
            lambda field, value: (
                callbacks.append((field, value)),
                app.apply_filter(field, value),
            )
        )
        app.push_screen(screen)
        await pilot.pause()
        screen.select_field("session_id")
        field = screen.query_one("#filter-value", Input)
        field.value = RAW

        screen.select_field("id")
        await pilot.pause()

        assert field.value == ""
        assert field.password is False
        assert RAW not in str(field.render())
        screen.submit_filter()
        assert callbacks == []
        assert app.state.filters == ()
        assert RAW not in repr(app.state)
        assert RAW not in repr(app)


async def test_switching_to_session_id_clears_prior_unmasked_value() -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    callbacks: list[tuple[str, str]] = []
    async with app.run_test(size=(100, 30)) as pilot:
        screen = FilterScreen(
            lambda field, value: callbacks.append((field, value))
        )
        app.push_screen(screen)
        await pilot.pause()
        field = screen.query_one("#filter-value", Input)
        field.value = "ordinary-filter-marker"

        screen.select_field("session_id")
        await pilot.pause()

        assert field.value == ""
        assert field.password is True
        screen.submit_filter()
        assert callbacks == []
        assert app.state.filters == ()


async def test_escape_scrubs_masked_session_id_before_unmount() -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    callbacks: list[tuple[str, str]] = []
    async with app.run_test(size=(100, 30)) as pilot:
        screen = FilterScreen(
            lambda field, value: callbacks.append((field, value))
        )
        app.push_screen(screen)
        await pilot.pause()
        screen.select_field("session_id")
        field = screen.query_one("#filter-value", Input)
        field.value = RAW

        await pilot.press("escape")
        await pilot.pause()

        assert field.value == ""
        assert callbacks == []
        assert app.state.filters == ()
        assert RAW not in repr(screen)
        assert RAW not in repr(app.state)
        assert RAW not in repr(app)


async def test_programmatic_dismiss_scrubs_before_returning() -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    async with app.run_test(size=(100, 30)) as pilot:
        screen = FilterScreen(lambda field, value: None)
        app.push_screen(screen)
        await pilot.pause()
        screen.select_field("session_id")
        field = screen.query_one("#filter-value", Input)
        field.value = RAW

        screen.dismiss()

        assert field.value == ""
        assert RAW not in repr(screen)


async def test_raw_session_filter_retains_only_returned_safe_ids(
    monkeypatch,
) -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    app.pending.offer(reset(view(SAFE)))

    class Client:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def performance(self, filters):
            assert filters == (f"session_id={RAW}",)
            return reset(view(SAFE)).snapshot

    monkeypatch.setattr("claude_code_proxy.tui.app.ControlClient", Client)
    async with app.run_test(size=(100, 30)) as pilot:
        app.drain_pending()
        await app.resolve_session_filter(RAW)
        await pilot.pause()

        assert app.state.filters == (
            FilterTerm("id", SAFE, exact_id=True, source="session_id"),
        )
        for value in (repr(app.state), repr(app), str(app.screen)):
            assert RAW not in value


async def test_session_lookup_failure_never_exposes_raw_value(
    monkeypatch,
) -> None:
    app = TuiApp(
        Path("/safe/control.sock"), pump=InertPump(), start_stream=False
    )
    notices: list[str] = []

    class Client:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def performance(self, filters):
            raise RuntimeError(RAW)

    monkeypatch.setattr("claude_code_proxy.tui.app.ControlClient", Client)
    async with app.run_test(size=(100, 30)):
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, **kwargs: notices.append(message),
        )
        await app.resolve_session_filter(RAW)

    assert notices == ["Session lookup unavailable"]
    assert RAW not in repr(app)
    assert all(RAW not in message for message in notices)


def test_filter_term_repr_contains_only_safe_exact_id() -> None:
    state = TuiState.empty()
    term = FilterTerm("id", SAFE, exact_id=True, source="session_id")

    assert RAW not in repr(term)
    assert RAW not in repr(state)
