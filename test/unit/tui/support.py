from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

from claude_code_proxy.control.schemas import (
    PerformanceAgentIdentityResponse,
    PerformanceCursorResponse,
    PerformanceEventResponse,
    PerformanceResetResponse,
    RequestPerformanceResponse,
    SessionPerformanceViewResponse,
)
from test.unit.test_performance_cli import (
    aggregate,
    metric,
    performance_response,
    request_payload,
    view_payload,
)

_CAPTURED_AT = "2026-01-02T03:04:06Z"
_PROCESS = {"pid": 42, "started_at": "2026-01-02T03:00:00Z"}


def view(
    identifier: str,
    *,
    state: str = "idle",
    first_seen: str = "2026-01-02T03:04:05Z",
    last_seen: str = _CAPTURED_AT,
    model: str = "provider-model",
    client_model: str = "client-model",
    provider: str = "provider",
    transport: str = "transport",
    effort: str = "high",
) -> SessionPerformanceViewResponse:
    payload = view_payload(identifier=identifier, model=model)
    session = payload["session"]
    session.update(
        {
            "state": state,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "client_model": client_model,
            "provider": provider,
            "transport": transport,
            "effort": effort,
        }
    )
    if state == "active":
        _make_view_active(payload, identifier)
    return SessionPerformanceViewResponse.model_validate(payload)


def _make_view_active(payload: dict[str, object], identifier: str) -> None:
    request = request_payload(outcome="active")
    request["id"] = f"request-{identifier}"
    request["session_id"] = identifier
    request["finished_at"] = None
    session = payload["session"]
    session["active_requests"] = 1
    session["last_result"] = None
    performance = payload["performance"]
    performance["active_requests"] = [request]
    performance["recent_requests"] = []
    performance["latest_request"] = request
    performance["current_concurrency"] = 1
    performance["outcomes"] = {}


def with_requests(
    item: SessionPerformanceViewResponse,
    *,
    active_ids: tuple[str, ...] = (),
    recent_ids: tuple[str, ...] = (),
) -> SessionPerformanceViewResponse:
    payload = item.model_dump(mode="json")
    active = tuple(_request(item.session.id, request_id, active=True) for request_id in active_ids)
    recent = tuple(_request(item.session.id, request_id, active=False) for request_id in recent_ids)
    performance = payload["performance"]
    performance["active_requests"] = [entry.model_dump(mode="json") for entry in active]
    performance["recent_requests"] = [entry.model_dump(mode="json") for entry in recent]
    performance["latest_request"] = (
        recent[0].model_dump(mode="json")
        if recent
        else active[0].model_dump(mode="json")
        if active
        else None
    )
    performance["current_concurrency"] = len(active)
    performance["peak_concurrency"] = max(1, len(active))
    performance["outcomes"] = {"completed": len(recent)} if recent else {}
    performance["requests"] = len(active) + len(recent)
    session = payload["session"]
    session["state"] = "active" if active else "idle"
    session["active_requests"] = len(active)
    session["requests"] = len(active) + len(recent)
    session["last_result"] = None if active else "completed" if recent else None
    return SessionPerformanceViewResponse.model_validate(payload)


def _request(
    session_id: str,
    request_id: str,
    *,
    active: bool,
) -> RequestPerformanceResponse:
    payload = request_payload(outcome="active" if active else "completed")
    payload["id"] = request_id
    payload["session_id"] = session_id
    payload["finished_at"] = None if active else _CAPTURED_AT
    return RequestPerformanceResponse.model_validate(payload)


def with_agent(
    item: SessionPerformanceViewResponse,
    identifier: str = "safe-agent",
) -> SessionPerformanceViewResponse:
    observed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    agent = PerformanceAgentIdentityResponse(
        id=identifier,
        parent_id=None,
        state="idle",
        active_requests=0,
        requests=1,
        client_model="client-model",
        model="provider-model",
        provider="provider",
        transport="transport",
        effort="high",
        context_window=1_000,
        first_seen=observed,
        last_seen=observed,
        elapsed_seconds=1,
        last_result="completed",
    )
    identity = item.session.model_copy(update={"agents": (agent,)})
    return item.model_copy(update={"session": identity})


def with_latest_metrics(
    item: SessionPerformanceViewResponse,
    *,
    duration: int | float | None = None,
    ttft: int | float | None = None,
) -> SessionPerformanceViewResponse:
    payload = item.model_dump(mode="json")
    latest = payload["performance"]["latest_request"]
    assert latest is not None
    if duration is not None:
        latest["duration"] = metric(value=duration)
    if ttft is not None:
        latest["ttft"] = metric(value=ttft)
    payload["performance"]["recent_requests"][0] = latest
    return SessionPerformanceViewResponse.model_validate(payload)


def without_requests(
    item: SessionPerformanceViewResponse,
) -> SessionPerformanceViewResponse:
    payload = item.model_dump(mode="json")
    performance = payload["performance"]
    performance["requests"] = 0
    performance["active_requests"] = []
    performance["recent_requests"] = []
    performance["outcomes"] = {}
    performance["current_concurrency"] = 0
    performance["latest_request"] = None
    session = payload["session"]
    session["requests"] = 0
    session["active_requests"] = 0
    session["last_result"] = None
    return SessionPerformanceViewResponse.model_validate(payload)


def with_aggregates(
    item: SessionPerformanceViewResponse,
    **values: tuple[int | float, int, int, int],
) -> SessionPerformanceViewResponse:
    payload = item.model_dump(mode="json")
    for name, (value, observed, unavailable, not_applicable) in values.items():
        payload["performance"][name] = aggregate(
            value,
            observed=observed,
            unavailable=unavailable,
            not_applicable=not_applicable,
        )
    return SessionPerformanceViewResponse.model_validate(payload)


def reset(
    *views: SessionPerformanceViewResponse,
    sequence: int = 7,
) -> PerformanceResetResponse:
    snapshot = performance_response().model_dump(mode="json")
    snapshot["cursor"] = sequence
    snapshot["sessions"] = [item.model_dump(mode="json") for item in views]
    return PerformanceResetResponse.model_validate(
        {
            "process": snapshot["process"],
            "sequence": sequence,
            "occurred_at": snapshot["captured_at"],
            "type": "reset",
            "snapshot": snapshot,
        }
    )


def event(
    item: SessionPerformanceViewResponse,
    *,
    event_type: str,
    sequence: int,
) -> PerformanceEventResponse:
    request = item.performance.latest_request
    assert request is not None
    return PerformanceEventResponse.model_validate(
        {
            "process": deepcopy(_PROCESS),
            "sequence": sequence,
            "occurred_at": _CAPTURED_AT,
            "type": event_type,
            "session_id": item.session.id,
            "activity": item.session.model_dump(mode="json", exclude={"agents"}),
            "request": request.model_dump(mode="json"),
            "session": item.performance.model_dump(mode="json"),
        }
    )


def cursor(sequence: int) -> PerformanceCursorResponse:
    return PerformanceCursorResponse.model_validate(
        {
            "process": deepcopy(_PROCESS),
            "sequence": sequence,
            "occurred_at": _CAPTURED_AT,
            "type": "cursor",
        }
    )
