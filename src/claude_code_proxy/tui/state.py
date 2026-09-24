"""Pure presentation state for the live analytics TUI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping

from ..control.schemas import (
    MetricAggregateResponse,
    MetricResponse,
    PerformanceCursorResponse,
    PerformanceEventResponse,
    PerformanceResetResponse,
    PerformanceSessionIdentityResponse,
    PerformanceStreamEvent,
    ProcessIdentityResponse,
    SessionPerformanceViewResponse,
)
from .formatting import cache_ratio_value, finite_non_negative

DisplayPhase = Literal[
    "active",
    "streaming",
    "tool-use",
    "retrying",
    "idle",
    "failed",
]
_EVENT_PHASES: Mapping[str, DisplayPhase] = MappingProxyType(
    {
        "request_started": "active",
        "first_output": "streaming",
        "progress": "streaming",
        "tool_use": "tool-use",
        "retry": "retrying",
    }
)

FilterField = Literal[
    "id",
    "state",
    "provider",
    "transport",
    "model",
    "effort",
]
FilterSource = Literal["field", "session_id"]
_FILTER_FIELDS = frozenset({"id", "state", "provider", "transport", "model", "effort"})
_FILTER_SOURCES = frozenset({"field", "session_id"})


@dataclass(frozen=True, slots=True)
class FilterTerm:
    """One retained filter containing safe display values only."""

    field: FilterField
    value: str
    exact_id: bool = False
    source: FilterSource = "field"

    def __post_init__(self) -> None:
        if self.field not in _FILTER_FIELDS:
            raise ValueError("unsupported filter field")
        if self.source not in _FILTER_SOURCES:
            raise ValueError("unsupported filter source")
        if not self.value:
            raise ValueError("filter value must not be empty")
        if self.exact_id and self.field != "id":
            raise ValueError("only ID filters may be exact")
        if self.source == "session_id" and not self.exact_id:
            raise ValueError("resolved session ID filters must be exact")


class SortField(str, Enum):
    BASELINE = "baseline"
    SESSION_ID = "session_id"
    STATE = "state"
    RECENCY = "recency"
    MODEL = "model"
    ELAPSED = "elapsed"
    TTFT = "ttft"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    CACHE_RATIO = "cache_ratio"
    TOOL_CALLS = "tool_calls"


class SortDirection(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True, slots=True)
class SortSpec:
    field: SortField = SortField.BASELINE
    direction: SortDirection = SortDirection.ASCENDING


@dataclass(frozen=True, slots=True)
class PhaseObservation:
    """One transient phase inferred from an ordinary stream event."""

    sequence: int
    request_id: str
    phase: DisplayPhase


@dataclass(frozen=True, slots=True)
class StateDelta:
    """Minimal presentation changes caused by one reduced event."""

    changed_session_ids: frozenset[str] = frozenset()
    order_changed: bool = False
    selection_changed: bool = False
    replace_all: bool = False


@dataclass(frozen=True, slots=True)
class TuiState:
    """Immutable keyed presentation state."""

    process: ProcessIdentityResponse | None = None
    captured_at: datetime | None = None
    cursor: int = 0
    sessions: Mapping[str, SessionPerformanceViewResponse] = field(
        default_factory=lambda: MappingProxyType({})
    )
    baseline_order: tuple[str, ...] = ()
    phases: Mapping[str, PhaseObservation] = field(
        default_factory=lambda: MappingProxyType({})
    )
    selected_session_id: str | None = None
    selected_request_id: str | None = None
    search: str = ""
    filters: tuple[FilterTerm, ...] = ()
    sort: SortSpec = SortSpec()

    @classmethod
    def empty(cls) -> TuiState:
        """Return empty initial state."""
        return cls()

    def phase_for(self, session_id: str) -> DisplayPhase:
        """Return inferred phase when available, else protocol state."""
        observed = self.phases.get(session_id)
        if observed is not None:
            return observed.phase
        return self.sessions[session_id].session.state

    def select_session(self, session_id: str | None) -> TuiState:
        """Select a visible session and its first retained request."""
        return select_session(self, session_id)

    def select_request(self, request_id: str | None) -> TuiState:
        """Select a retained request in the selected session."""
        return select_request(self, request_id)


def apply_stream_event(
    state: TuiState,
    event: PerformanceStreamEvent,
) -> tuple[TuiState, StateDelta]:
    """Apply one validated stream event with copy-on-write semantics."""
    if isinstance(event, PerformanceResetResponse):
        return _apply_reset(state, event)
    _require_advancing_sequence(state, event)
    if isinstance(event, PerformanceEventResponse):
        return _apply_event(state, event)
    if isinstance(event, PerformanceCursorResponse):
        updated = replace(state, cursor=event.sequence)
        return updated, StateDelta()
    raise ValueError("unsupported performance stream event")


def _require_advancing_sequence(
    state: TuiState,
    event: PerformanceEventResponse | PerformanceCursorResponse,
) -> None:
    if event.sequence <= state.cursor:
        raise ValueError("event sequence must advance TUI state")


def _apply_reset(
    previous: TuiState,
    event: PerformanceResetResponse,
) -> tuple[TuiState, StateDelta]:
    ordered = sorted(
        event.snapshot.sessions,
        key=lambda item: (item.session.first_seen, item.session.id),
    )
    sessions = MappingProxyType({item.session.id: item for item in ordered})
    baseline = tuple(sessions)
    selected_session = _nearest_surviving_session(
        previous.selected_session_id,
        previous.baseline_order,
        baseline,
    )
    selected_request = _selected_request(
        sessions,
        selected_session,
        previous.selected_request_id,
    )
    state = TuiState(
        process=event.process,
        captured_at=event.snapshot.captured_at,
        cursor=event.sequence,
        sessions=sessions,
        baseline_order=baseline,
        phases=MappingProxyType({}),
        selected_session_id=selected_session,
        selected_request_id=selected_request,
        search=previous.search,
        filters=previous.filters,
        sort=previous.sort,
    )
    state = _reconcile_visible_selection(state)
    selection_changed = (
        state.selected_session_id != previous.selected_session_id
        or state.selected_request_id != previous.selected_request_id
    )
    delta = StateDelta(
        changed_session_ids=frozenset(sessions),
        order_changed=True,
        selection_changed=selection_changed,
        replace_all=True,
    )
    return state, delta


def _apply_event(
    previous: TuiState,
    event: PerformanceEventResponse,
) -> tuple[TuiState, StateDelta]:
    sessions = dict(previous.sessions)
    is_new = event.session_id not in sessions
    prior_view = sessions.get(event.session_id)
    sessions[event.session_id] = _event_view(event, prior_view)
    baseline = previous.baseline_order
    if is_new:
        baseline = (*baseline, event.session_id)
    phases = _updated_phases(previous.phases, event)
    selected_session = previous.selected_session_id or event.session_id
    selected_request = _selected_request(
        sessions,
        selected_session,
        previous.selected_request_id,
    )
    state = replace(
        previous,
        process=event.process,
        captured_at=event.occurred_at,
        cursor=event.sequence,
        sessions=MappingProxyType(sessions),
        baseline_order=baseline,
        phases=phases,
        selected_session_id=selected_session,
        selected_request_id=selected_request,
    )
    state = _reconcile_visible_selection(state)
    selection_changed = (
        state.selected_session_id != previous.selected_session_id
        or state.selected_request_id != previous.selected_request_id
    )
    delta = StateDelta(
        changed_session_ids=frozenset({event.session_id}),
        order_changed=is_new,
        selection_changed=selection_changed,
    )
    return state, delta


def _event_view(
    event: PerformanceEventResponse,
    prior_view: SessionPerformanceViewResponse | None,
) -> SessionPerformanceViewResponse:
    agents = () if prior_view is None else prior_view.session.agents
    identity = PerformanceSessionIdentityResponse.model_validate(
        {
            **event.activity.model_dump(mode="python"),
            "agents": tuple(
                agent.model_dump(mode="python") for agent in agents
            ),
        }
    )
    return SessionPerformanceViewResponse(
        session=identity,
        performance=event.session,
    )


def _updated_phases(
    current: Mapping[str, PhaseObservation],
    event: PerformanceEventResponse,
) -> Mapping[str, PhaseObservation]:
    phases = dict(current)
    phase = _EVENT_PHASES.get(event.type)
    if phase is None:
        phases.pop(event.session_id, None)
    else:
        phases[event.session_id] = PhaseObservation(
            event.sequence,
            event.request.id,
            phase,
        )
    return MappingProxyType(phases)


def with_search(state: TuiState, query: str) -> TuiState:
    """Replace case-insensitive quick search text."""
    return _reconcile_visible_selection(replace(state, search=query.strip()))


def with_filter(state: TuiState, term: FilterTerm) -> TuiState:
    """Append one immutable filter term."""
    return _reconcile_visible_selection(
        replace(state, filters=(*state.filters, term))
    )


def clear_query(state: TuiState) -> TuiState:
    """Clear search and filters while retaining the active sort."""
    return _reconcile_visible_selection(
        replace(state, search="", filters=())
    )


def with_sort(state: TuiState, sort: SortSpec) -> TuiState:
    """Replace the visible-row sort without changing keyed selection."""
    return _reconcile_visible_selection(replace(state, sort=sort))


def visible_session_ids(state: TuiState) -> tuple[str, ...]:
    """Return safe IDs passing search and filters in selected order."""
    visible = [
        identifier
        for identifier in state.baseline_order
        if _matches_search(state, state.sessions[identifier])
        and _matches_filters(state, state.sessions[identifier])
    ]
    if state.sort.field is SortField.BASELINE:
        return tuple(visible)
    return _sort_sessions(state, visible)


def _matches_search(
    state: TuiState,
    view: SessionPerformanceViewResponse,
) -> bool:
    query = state.search.casefold()
    if not query:
        return True
    session = view.session
    values = (
        session.id,
        session.client_model,
        session.model,
        session.provider,
        session.transport,
        session.state,
        state.phase_for(session.id),
        session.effort,
        session.last_result or "",
    )
    return any(query in value.casefold() for value in values)


def _matches_filters(
    state: TuiState,
    view: SessionPerformanceViewResponse,
) -> bool:
    grouped: dict[FilterField, list[FilterTerm]] = {}
    for term in state.filters:
        grouped.setdefault(term.field, []).append(term)
    for field, terms in grouped.items():
        actual = _filter_value(view, field).casefold()
        if not any(_matches_filter(actual, field, term) for term in terms):
            return False
    return True


def _filter_value(
    view: SessionPerformanceViewResponse,
    field: FilterField,
) -> str:
    session = view.session
    values = {
        "id": session.id,
        "state": session.state,
        "provider": session.provider,
        "transport": session.transport,
        "model": session.model,
        "effort": session.effort,
    }
    return values[field]


def _matches_filter(
    actual: str,
    field: FilterField,
    term: FilterTerm,
) -> bool:
    expected = term.value.casefold()
    if term.exact_id:
        return actual == expected
    if field == "id":
        return actual.startswith(expected)
    return actual == expected


def _sort_sessions(
    state: TuiState,
    identifiers: list[str],
) -> tuple[str, ...]:
    observed: list[tuple[str, object]] = []
    missing: list[str] = []
    for identifier in identifiers:
        value = _sort_value(state, state.sessions[identifier])
        if value is None:
            missing.append(identifier)
        else:
            observed.append((identifier, value))
    observed.sort(key=lambda item: (item[0].casefold(), item[0]))
    observed.sort(
        key=lambda item: item[1],
        reverse=state.sort.direction is SortDirection.DESCENDING,
    )
    missing.sort(key=lambda item: (item.casefold(), item))
    return tuple(identifier for identifier, _ in observed) + tuple(missing)


def _sort_value(
    state: TuiState,
    view: SessionPerformanceViewResponse,
) -> object | None:
    field = state.sort.field
    session = view.session
    performance = view.performance
    latest = performance.latest_request
    if field is SortField.SESSION_ID:
        return session.id.casefold()
    if field is SortField.STATE:
        return state.phase_for(session.id)
    if field is SortField.RECENCY:
        return session.last_seen
    if field is SortField.MODEL:
        return session.model.casefold()
    if field is SortField.ELAPSED:
        return _observed_metric(None if latest is None else latest.duration)
    if field is SortField.TTFT:
        return _observed_metric(None if latest is None else latest.ttft)
    if field is SortField.INPUT_TOKENS:
        return _observed_aggregate(performance.input_tokens)
    if field is SortField.OUTPUT_TOKENS:
        return _observed_aggregate(performance.output_tokens)
    if field is SortField.CACHE_RATIO:
        return cache_ratio_value(performance)
    if field is SortField.TOOL_CALLS:
        return _observed_aggregate(performance.tool_calls)
    raise ValueError(f"unsupported dynamic sort {field.value}")


def _observed_metric(metric: MetricResponse | None) -> int | float | None:
    if metric is None or metric.status != "observed":
        return None
    return finite_non_negative(metric.value)


def _observed_aggregate(
    metric: MetricAggregateResponse,
) -> int | float | None:
    if metric.observed_samples == 0:
        return None
    return finite_non_negative(metric.value)


def select_session(state: TuiState, session_id: str | None) -> TuiState:
    """Select one currently visible safe session ID."""
    if session_id is None or session_id not in visible_session_ids(state):
        return state
    return replace(
        state,
        selected_session_id=session_id,
        selected_request_id=_selected_request(
            state.sessions,
            session_id,
            None,
        ),
    )


def select_request(state: TuiState, request_id: str | None) -> TuiState:
    """Select one request retained under the selected session."""
    selected = state.selected_session_id
    if selected is None or request_id is None:
        return state
    if request_id not in _request_ids(state.sessions[selected]):
        return state
    return replace(state, selected_request_id=request_id)


def _reconcile_visible_selection(state: TuiState) -> TuiState:
    visible = visible_session_ids(state)
    current = state.selected_session_id
    if current in visible:
        request_id = _selected_request(
            state.sessions,
            current,
            state.selected_request_id,
        )
        return replace(state, selected_request_id=request_id)
    selected = _nearest_in_order(current, state.baseline_order, visible)
    request_id = _selected_request(
        state.sessions,
        selected,
        state.selected_request_id,
    )
    return replace(
        state,
        selected_session_id=selected,
        selected_request_id=request_id,
    )


def _nearest_surviving_session(
    selected: str | None,
    previous_order: tuple[str, ...],
    current_order: tuple[str, ...],
) -> str | None:
    if selected in current_order:
        return selected
    return _nearest_in_order(selected, previous_order, current_order)


def _nearest_in_order(
    selected: str | None,
    reference_order: tuple[str, ...],
    candidates: tuple[str, ...],
) -> str | None:
    if not candidates:
        return None
    if selected not in reference_order:
        return candidates[0]
    positions = {identifier: index for index, identifier in enumerate(reference_order)}
    selected_index = positions[selected]
    ranked = (
        (abs(positions[identifier] - selected_index), positions[identifier], identifier)
        for identifier in candidates
        if identifier in positions
    )
    nearest = min(ranked, default=None)
    if nearest is not None:
        return nearest[2]
    return candidates[0]


def _selected_request(
    sessions: Mapping[str, SessionPerformanceViewResponse],
    selected_session_id: str | None,
    selected_request_id: str | None,
) -> str | None:
    if selected_session_id is None:
        return None
    available = _request_ids(sessions[selected_session_id])
    if selected_request_id in available:
        return selected_request_id
    if available:
        return available[0]
    return None


def _request_ids(view: SessionPerformanceViewResponse) -> tuple[str, ...]:
    performance = view.performance
    requests = performance.active_requests + performance.recent_requests
    return tuple(request.id for request in requests)
