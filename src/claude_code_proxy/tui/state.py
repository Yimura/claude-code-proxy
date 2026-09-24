"""Pure presentation state for the live analytics TUI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Literal, Mapping

from ..control.schemas import (
    PerformanceCursorResponse,
    PerformanceEventResponse,
    PerformanceResetResponse,
    PerformanceSessionIdentityResponse,
    PerformanceStreamEvent,
    ProcessIdentityResponse,
    SessionPerformanceViewResponse,
)

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
        """Select an existing session and its first retained request."""
        if session_id is None or session_id not in self.sessions:
            return self
        request_id = _selected_request(self.sessions, session_id, None)
        return replace(
            self,
            selected_session_id=session_id,
            selected_request_id=request_id,
        )

    def select_request(self, request_id: str | None) -> TuiState:
        """Select a retained request in the selected session."""
        if self.selected_session_id is None or request_id is None:
            return self
        available = _request_ids(
            self.sessions[self.selected_session_id]
        )
        if request_id not in available:
            return self
        return replace(self, selected_request_id=request_id)


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
        updated = replace(
            state,
            process=event.process,
            captured_at=event.occurred_at,
            cursor=event.sequence,
        )
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
    selected_session = _surviving_session(
        previous.selected_session_id,
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
    )
    selection_changed = (
        selected_session != previous.selected_session_id
        or selected_request != previous.selected_request_id
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
    selection_changed = (
        selected_session != previous.selected_session_id
        or selected_request != previous.selected_request_id
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


def _surviving_session(
    selected: str | None,
    ordered_ids: tuple[str, ...],
) -> str | None:
    if selected in ordered_ids:
        return selected
    if ordered_ids:
        return ordered_ids[0]
    return None


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
