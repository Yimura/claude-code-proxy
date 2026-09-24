import pytest

from claude_code_proxy.tui.state import (
    FilterTerm,
    SortDirection,
    SortField,
    SortSpec,
    TuiState,
    apply_stream_event,
    clear_query,
    select_session,
    visible_session_ids,
    with_filter,
    with_search,
    with_sort,
)
from test.unit.tui.support import (
    cursor,
    event,
    reset,
    view,
    with_agent,
    with_aggregates,
    with_latest_metrics,
    with_requests,
    without_requests,
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
    process = state.process
    captured_at = state.captured_at

    state, delta = apply_stream_event(state, cursor(8))

    assert state.cursor == 8
    assert state.sessions is sessions
    assert state.process is process
    assert state.captured_at is captured_at
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "SAFE-SECOND"),
        ("client_model", "CLIENT-SPECIAL"),
        ("model", "MODEL-SPECIAL"),
        ("provider", "PROVIDER-SPECIAL"),
        ("transport", "TRANSPORT-SPECIAL"),
        ("state", "FAILED"),
        ("effort", "MAX-SPECIAL"),
        ("last_result", "FAILED"),
    ],
)
def test_quick_search_casefolds_safe_visible_metadata(
    field: str,
    value: str,
) -> None:
    first = view("safe-first")
    updates = {
        "client_model": "client-special",
        "model": "model-special",
        "provider": "provider-special",
        "transport": "transport-special",
        "effort": "max-special",
    }
    if field == "id":
        second = view("safe-second")
    elif field == "state":
        second = view("safe-second", state="failed")
    else:
        second = view("safe-second")
        session = second.session.model_copy(update={field: updates.get(field, "failed")})
        second = second.model_copy(update={"session": session})
    state, _ = apply_stream_event(TuiState.empty(), reset(first, second))

    searched = with_search(state, value)

    assert visible_session_ids(searched) == ("safe-second",)


def test_quick_search_matches_derived_phase_and_can_be_cleared() -> None:
    active = view("active", state="active")
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(view("idle"), active),
    )
    state, _ = apply_stream_event(
        state,
        event(active, event_type="retry", sequence=8),
    )

    searched = with_search(state, "RETRYING")

    assert visible_session_ids(searched) == ("active",)
    assert visible_session_ids(clear_query(searched)) == ("active", "idle")


def test_filter_values_or_within_field_and_between_fields() -> None:
    first = view("first", provider="one")
    second = view("second", provider="one")
    third = view("third", provider="two")
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(first, second, third),
    )
    state = with_filter(state, FilterTerm("id", "fir"))
    state = with_filter(state, FilterTerm("id", "sec"))
    state = with_filter(state, FilterTerm("provider", "ONE"))

    assert visible_session_ids(state) == ("first", "second")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", "idle"),
        ("provider", "provider"),
        ("transport", "transport"),
        ("model", "provider-model"),
        ("effort", "high"),
    ],
)
def test_each_non_id_filter_is_case_insensitive_exact(
    field: str,
    value: str,
) -> None:
    state, _ = apply_stream_event(TuiState.empty(), reset(view("session")))

    matching = with_filter(state, FilterTerm(field, value.upper()))
    nonmatching = with_filter(state, FilterTerm(field, value[:3]))

    assert visible_session_ids(matching) == ("session",)
    assert visible_session_ids(nonmatching) == ()


def test_normal_id_filter_is_prefix_and_resolved_filter_is_exact_safe_id() -> None:
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(view("abc"), view("abcdef")),
    )

    prefixed = with_filter(state, FilterTerm("id", "ABC"))
    exact = with_filter(
        state,
        FilterTerm("id", "abc", exact_id=True, source="session_id"),
    )

    assert visible_session_ids(prefixed) == ("abc", "abcdef")
    assert visible_session_ids(exact) == ("abc",)
    assert "raw-session" not in repr(exact)


def test_filter_term_rejects_invalid_fields_and_exact_non_id() -> None:
    with pytest.raises(ValueError, match="filter field"):
        FilterTerm("unknown", "value")
    with pytest.raises(ValueError, match="only ID filters"):
        FilterTerm("model", "value", exact_id=True)


def _sortable_views():
    lower = with_latest_metrics(
        view(
            "a",
            state="failed",
            last_seen="2026-01-02T02:04:06Z",
            model="aaa-model",
        ),
        duration=1,
        ttft=1,
    )
    lower = with_aggregates(
        lower,
        input_tokens=(1, 1, 0, 0),
        output_tokens=(1, 1, 0, 0),
        cache_read_tokens=(1, 1, 0, 0),
        cache_creation_tokens=(8, 1, 0, 0),
        tool_calls=(1, 1, 0, 0),
    )
    higher = with_latest_metrics(
        view(
            "b",
            state="idle",
            last_seen="2026-01-02T04:04:06Z",
            model="zzz-model",
        ),
        duration=2,
        ttft=2,
    )
    higher = with_aggregates(
        higher,
        input_tokens=(2, 1, 0, 0),
        output_tokens=(2, 1, 0, 0),
        cache_read_tokens=(18, 1, 0, 0),
        cache_creation_tokens=(0, 1, 0, 0),
        tool_calls=(2, 1, 0, 0),
    )
    return lower, higher


@pytest.mark.parametrize("direction", tuple(SortDirection))
def test_baseline_sort_keeps_stable_order_in_both_directions(
    direction: SortDirection,
) -> None:
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(
            view("later", first_seen="2026-01-02T04:04:05Z"),
            view("earlier", first_seen="2026-01-02T02:04:05Z"),
        ),
    )

    state = with_sort(state, SortSpec(SortField.BASELINE, direction))

    assert visible_session_ids(state) == ("earlier", "later")


@pytest.mark.parametrize("field", tuple(SortField)[1:])
@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (SortDirection.ASCENDING, ("a", "b")),
        (SortDirection.DESCENDING, ("b", "a")),
    ],
)
def test_every_dynamic_sort_obeys_direction_and_safe_id_tie_breaker(
    field: SortField,
    direction: SortDirection,
    expected: tuple[str, str],
) -> None:
    lower, higher = _sortable_views()
    state, _ = apply_stream_event(TuiState.empty(), reset(higher, lower))

    sorted_state = with_sort(state, SortSpec(field, direction))

    assert visible_session_ids(sorted_state) == expected
    assert visible_session_ids(sorted_state) == expected


@pytest.mark.parametrize(
    "field",
    [
        SortField.ELAPSED,
        SortField.TTFT,
        SortField.INPUT_TOKENS,
        SortField.OUTPUT_TOKENS,
        SortField.CACHE_RATIO,
        SortField.TOOL_CALLS,
    ],
)
@pytest.mark.parametrize("direction", tuple(SortDirection))
def test_missing_metrics_sort_last_in_both_directions(
    field: SortField,
    direction: SortDirection,
) -> None:
    observed = view("observed")
    missing = without_requests(view("missing"))
    if field in {
        SortField.INPUT_TOKENS,
        SortField.OUTPUT_TOKENS,
        SortField.CACHE_RATIO,
        SortField.TOOL_CALLS,
    }:
        missing = with_aggregates(
            missing,
            input_tokens=(0, 0, 1, 0),
            output_tokens=(0, 0, 1, 0),
            cache_read_tokens=(0, 0, 1, 0),
            cache_creation_tokens=(0, 0, 1, 0),
            tool_calls=(0, 0, 1, 0),
        )
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(missing, observed),
    )

    state = with_sort(state, SortSpec(field, direction))

    assert visible_session_ids(state)[-1] == "missing"


def test_partial_metric_is_observed_for_sorting() -> None:
    partial = with_aggregates(
        view("partial"),
        input_tokens=(1, 1, 1, 0),
    )
    missing = with_aggregates(
        view("missing"),
        input_tokens=(0, 0, 1, 0),
    )
    state, _ = apply_stream_event(TuiState.empty(), reset(missing, partial))

    state = with_sort(
        state,
        SortSpec(SortField.INPUT_TOKENS, SortDirection.DESCENDING),
    )

    assert visible_session_ids(state) == ("partial", "missing")


def test_baseline_order_does_not_move_on_metric_update() -> None:
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(view("first"), view("second")),
    )
    updated = with_latest_metrics(view("second"), duration=999, ttft=999)

    state, _ = apply_stream_event(
        state,
        event(updated, event_type="completed", sequence=8),
    )

    assert visible_session_ids(state) == ("first", "second")


def test_keyed_selection_survives_sort_and_request_survives_reset() -> None:
    selected = with_requests(
        view("second"),
        active_ids=("active-one", "active-two"),
    )
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(view("first"), selected),
    )
    state = select_session(state, "second").select_request("active-two")
    state = with_sort(
        state,
        SortSpec(SortField.SESSION_ID, SortDirection.DESCENDING),
    )

    state, _ = apply_stream_event(state, reset(selected, view("first"), sequence=20))

    assert state.selected_session_id == "second"
    assert state.selected_request_id == "active-two"


def test_hidden_selection_moves_to_nearest_baseline_row() -> None:
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(view("a"), view("b"), view("c"), view("d")),
    )
    state = select_session(state, "c")
    state = with_filter(state, FilterTerm("id", "b"))
    state = with_filter(state, FilterTerm("id", "d"))

    assert visible_session_ids(state) == ("b", "d")
    assert state.selected_session_id == "b"


def test_removed_selection_moves_to_nearest_surviving_baseline_row() -> None:
    state, _ = apply_stream_event(
        TuiState.empty(),
        reset(view("a"), view("b"), view("c"), view("d")),
    )
    state = select_session(state, "c")

    state, _ = apply_stream_event(
        state,
        reset(view("a"), view("b"), view("d"), sequence=20),
    )

    assert state.selected_session_id == "b"


def test_no_visible_session_clears_session_and_request_selection() -> None:
    item = with_requests(view("session"), active_ids=("request",))
    state, _ = apply_stream_event(TuiState.empty(), reset(item))

    state = with_filter(state, FilterTerm("id", "absent"))

    assert state.selected_session_id is None
    assert state.selected_request_id is None


def test_reset_preserves_search_filters_and_sort() -> None:
    state, _ = apply_stream_event(TuiState.empty(), reset(view("first")))
    state = with_search(state, "first")
    state = with_filter(state, FilterTerm("state", "idle"))
    sort = SortSpec(SortField.SESSION_ID, SortDirection.DESCENDING)
    state = with_sort(state, sort)

    state, _ = apply_stream_event(
        state,
        reset(view("first"), view("second"), sequence=20),
    )

    assert state.search == "first"
    assert state.filters == (FilterTerm("state", "idle"),)
    assert state.sort == sort
    assert visible_session_ids(state) == ("first",)
