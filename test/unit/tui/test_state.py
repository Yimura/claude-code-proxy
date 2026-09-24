import pytest

from claude_code_proxy.tui.state import TuiState, apply_stream_event
from test.unit.tui.support import (
    cursor,
    event,
    reset,
    view,
    with_agent,
    with_requests,
)


def test_reset_establishes_first_seen_baseline_and_selection() -> None:
    newer = view("newer", first_seen="2026-01-02T03:04:05Z")
    older = view("older", first_seen="2026-01-02T02:04:05Z")

    state, delta = apply_stream_event(TuiState.empty(), reset(newer, older))

    assert state.baseline_order == ("older", "newer")
    assert state.selected_session_id == "older"
    assert state.cursor == 7
    assert delta.replace_all is True


def test_reset_is_authoritative_and_preserves_surviving_keyed_selection() -> None:
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(view("first"), view("second")),
    )
    state = state.select_session("second")

    state, _ = apply_stream_event(
        state,
        reset(view("replacement"), view("second"), sequence=20),
    )

    assert tuple(state.sessions) == ("replacement", "second")
    assert state.selected_session_id == "second"
    assert state.phases == {}


def test_event_replaces_complete_session_without_reordering() -> None:
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(view("first"), view("second")),
    )
    active = view("second", state="active", model="updated-model")

    state, delta = apply_stream_event(
        state,
        event(active, event_type="tool_use", sequence=10),
    )

    assert state.baseline_order == ("first", "second")
    assert state.sessions["second"] == active
    assert state.phase_for("second") == "tool-use"
    assert state.cursor == 10
    assert delta.changed_session_ids == frozenset({"second"})


def test_event_appends_genuinely_new_session() -> None:
    state, _ = apply_stream_event(TuiState.empty(), reset(view("first")))
    active = view("new", state="active")

    state, delta = apply_stream_event(
        state,
        event(active, event_type="request_started", sequence=8),
    )

    assert state.baseline_order == ("first", "new")
    assert delta.order_changed is True


def test_cursor_advances_without_row_change() -> None:
    state, _ = apply_stream_event(TuiState.empty(), reset(view("session")))
    sessions = state.sessions

    state, delta = apply_stream_event(state, cursor(8))

    assert state.cursor == 8
    assert state.sessions is sessions
    assert delta.changed_session_ids == frozenset()
    assert delta.replace_all is False


@pytest.mark.parametrize("sequence", [7, 6])
def test_non_advancing_incremental_sequence_is_rejected(sequence: int) -> None:
    state, _ = apply_stream_event(TuiState.empty(), reset(view("session")))

    with pytest.raises(ValueError, match="sequence must advance"):
        apply_stream_event(state, cursor(sequence))


def test_fresh_reset_discards_stale_rows_and_inferred_phases() -> None:
    active = view("active", state="active")
    state, _ = apply_stream_event(TuiState.empty(), reset(active))
    state, _ = apply_stream_event(
        state,
        event(active, event_type="progress", sequence=8),
    )

    state, _ = apply_stream_event(
        state,
        reset(view("replacement"), sequence=20),
    )

    assert tuple(state.sessions) == ("replacement",)
    assert state.phase_for("replacement") == "idle"
    assert "active" not in state.phases


@pytest.mark.parametrize(
    ("event_type", "phase"),
    [
        ("request_started", "active"),
        ("first_output", "streaming"),
        ("progress", "streaming"),
        ("tool_use", "tool-use"),
        ("retry", "retrying"),
    ],
)
def test_active_events_infer_best_effort_phase(
    event_type: str,
    phase: str,
) -> None:
    active = view("session", state="active")
    state, _ = apply_stream_event(TuiState.empty(), reset(active))

    state, _ = apply_stream_event(
        state,
        event(active, event_type=event_type, sequence=8),
    )

    assert state.sessions["session"].session.state == "active"
    assert state.phase_for("session") == phase


def test_terminal_event_clears_inferred_phase() -> None:
    active = view("session", state="active")
    state, _ = apply_stream_event(TuiState.empty(), reset(active))
    state, _ = apply_stream_event(
        state,
        event(active, event_type="retry", sequence=8),
    )
    completed = view("session")

    state, _ = apply_stream_event(
        state,
        event(completed, event_type="completed", sequence=9),
    )

    assert state.phase_for("session") == "idle"
    assert "session" not in state.phases


def test_event_preserves_agents_from_reset_identity() -> None:
    current = view("session")
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(with_agent(current)),
    )

    state, _ = apply_stream_event(
        state,
        event(current, event_type="completed", sequence=8),
    )

    assert tuple(agent.id for agent in state.sessions["session"].session.agents) == (
        "safe-agent",
    )


def test_selection_tracks_active_and_recent_request_keys() -> None:
    item = with_requests(
        view("session"),
        active_ids=("active-one", "active-two"),
        recent_ids=("recent-one",),
    )

    state, _ = apply_stream_event(TuiState.empty(), reset(item))
    state = state.select_request("active-two")

    assert state.selected_request_id == "active-two"
    assert state.select_request("not-present") is state


def test_state_is_copy_on_write_and_mapping_is_immutable() -> None:
    original, _ = apply_stream_event(TuiState.empty(), reset(view("session")))
    updated, _ = apply_stream_event(original, cursor(8))

    assert updated is not original
    assert original.cursor == 7
    with pytest.raises(TypeError):
        original.sessions["other"] = view("other")


def test_state_repr_retains_only_safe_identifiers() -> None:
    state, _ = apply_stream_event(TuiState.empty(), reset(view("safe-session")))

    assert "safe-session" in repr(state)
    assert "raw-session" not in repr(state)
