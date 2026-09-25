"""Literal read-only session and request detail widgets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

from rich.text import Text
from textual.widgets import Static

from ..control.schemas import (
    MetricAggregateResponse,
    PerformanceAgentIdentityResponse,
    RequestPerformanceResponse,
)
from .formatting import (
    format_aggregate,
    format_request_elapsed,
    metric_detail,
    safe_cell,
)
from .state import TuiState


class SessionDetails(Static):
    """Read-only full session and safe agent hierarchy."""

    def __init__(self, *, id: str = "session-details") -> None:
        super().__init__(id=id, markup=False)

    def sync_state(self, state: TuiState) -> None:
        selected = state.selected_session_id
        if selected is None:
            self.update(safe_cell("No session selected"))
            return
        self.update(session_detail_text(state, selected))


class RequestDetails(Static):
    """Read-only request metrics and structured failure fields."""

    def __init__(self, *, id: str = "request-details") -> None:
        super().__init__(id=id, markup=False)

    def sync_state(
        self,
        state: TuiState,
        *,
        now: datetime | None = None,
    ) -> None:
        request = selected_request(state)
        if request is None:
            self.update(safe_cell("No request selected"))
            return
        self.update(
            request_detail_text(
                request,
                outcome=state.phase_for_request(request),
                now=now,
            )
        )


def session_detail_text(state: TuiState, identifier: str) -> Text:
    """Build complete literal session detail content."""
    view = state.sessions[identifier]
    session = view.session
    performance = view.performance
    lines: list[tuple[str, object]] = [
        ("Session", session.id),
        ("Models", f"client {session.client_model} · resolved {session.model}"),
        ("Route", f"{session.provider} · {session.transport}"),
        ("State / phase", f"{session.state} / {state.phase_for(identifier)}"),
        ("Effort / context", f"{session.effort} / {session.context_window or '—'}"),
        (
            "First / last / elapsed",
            f"{session.first_seen.isoformat()} / {session.last_seen.isoformat()} "
            f"/ {session.elapsed_seconds:g}s",
        ),
        (
            "Requests",
            f"{session.requests} · active {performance.current_concurrency} "
            f"· peak {performance.peak_concurrency}",
        ),
        ("Outcomes", _outcomes(performance.outcomes)),
        ("Input tokens", _aggregate_detail(performance.input_tokens)),
        ("Output tokens", _aggregate_detail(performance.output_tokens)),
        ("Cache read", _aggregate_detail(performance.cache_read_tokens)),
        ("Cache create", _aggregate_detail(performance.cache_creation_tokens)),
        ("Reasoning", _aggregate_detail(performance.reasoning_tokens)),
        (
            "Tools / retries",
            f"{_aggregate_detail(performance.tool_calls)} / "
            f"{_aggregate_detail(performance.retries)}",
        ),
        (
            "Latest result",
            performance.latest_request.outcome
            if performance.latest_request is not None
            else "—",
        ),
    ]
    text = _detail_lines(lines)
    text.append_text(_agent_hierarchy(session.agents))
    return text


def _agent_hierarchy(
    agents: tuple[PerformanceAgentIdentityResponse, ...],
) -> Text:
    text = Text("\nAgents\n")
    if not agents:
        text.append_text(safe_cell("  none"))
    for agent in agents:
        parent = agent.parent_id or "root"
        agent_line = (
            f"  {agent.id} · parent {parent} · model {agent.model} · "
            f"state {agent.state} · effort {agent.effort} · "
            f"requests {agent.requests} · active {agent.active_requests} · "
            f"last seen {agent.last_seen.isoformat()}"
        )
        text.append_text(safe_cell(agent_line))
        text.append("\n")
    return text


def request_detail_text(
    request: RequestPerformanceResponse,
    *,
    outcome: str | None = None,
    now: datetime | None = None,
) -> Text:
    """Build complete literal request detail content."""
    lines: list[tuple[str, object]] = [
        ("Request", request.id),
        (
            "Operation / outcome",
            f"{request.operation} / {outcome or request.outcome}",
        ),
        ("Started", request.started_at.isoformat()),
        (
            "Finished",
            request.finished_at.isoformat() if request.finished_at else "active",
        ),
        ("Duration", format_request_elapsed(request, now=now)),
        ("Upstream", metric_detail(request.upstream_duration)),
        ("TTFT", metric_detail(request.ttft)),
        ("Input tokens", metric_detail(request.input_tokens)),
        ("Output tokens", metric_detail(request.output_tokens)),
        ("Cache read", metric_detail(request.cache_read_tokens)),
        ("Cache create", metric_detail(request.cache_creation_tokens)),
        ("Reasoning", metric_detail(request.reasoning_tokens)),
        ("Tool calls", metric_detail(request.tool_calls)),
        ("Retries", metric_detail(request.retries)),
        ("Peak concurrency", metric_detail(request.peak_concurrency)),
        ("Continuation", request.reasoning_continuation),
    ]
    if request.failure is not None:
        failure = request.failure
        lines.extend(
            (
                (
                    "Failure category / stage",
                    f"{failure.category} / {failure.stage}",
                ),
                ("Failure code", failure.code),
                ("Provider code", failure.provider_code or "—"),
                ("Exception type", failure.exception_type or "—"),
                ("Location", failure.location or "—"),
            )
        )
    return _detail_lines(lines)


def selected_request(state: TuiState) -> RequestPerformanceResponse | None:
    """Return the selected retained request when present."""
    session_id = state.selected_session_id
    request_id = state.selected_request_id
    if session_id is None or request_id is None:
        return None
    performance = state.sessions[session_id].performance
    for request in performance.active_requests + performance.recent_requests:
        if request.id == request_id:
            return request
    return None


def _detail_lines(lines: Iterable[tuple[str, object]]) -> Text:
    text = Text(no_wrap=False, overflow="fold")
    for label, value in lines:
        text.append(label + ": ", style="bold")
        text.append_text(safe_cell(value))
        text.append("\n")
    return text


def _aggregate_detail(metric: MetricAggregateResponse) -> str:
    return (
        f"{format_aggregate(metric)} "
        f"(observed {metric.observed_samples}, unavailable "
        f"{metric.unavailable_samples}, not applicable "
        f"{metric.not_applicable_samples})"
    )


def _outcomes(outcomes: Mapping[str, int]) -> str:
    if not outcomes:
        return "none"
    return ", ".join(
        f"{key} {value}" for key, value in sorted(outcomes.items())
    )
