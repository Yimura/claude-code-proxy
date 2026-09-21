"""Validated response schemas for the local control API."""

from collections.abc import Mapping
from datetime import UTC, datetime
import math
from types import MappingProxyType
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from ..failures import FailureCategory, FailureStage
from ..limits import MAX_CONTROL_INTEGER


NonNegativeControlInteger = Annotated[
    int, Field(strict=True, ge=0, le=MAX_CONTROL_INTEGER)
]
PositiveControlInteger = Annotated[
    int, Field(strict=True, ge=1, le=MAX_CONTROL_INTEGER)
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


def _require_wire_duration(value: object) -> object:
    if type(value) not in (int, float):
        raise ValueError("duration must be a JSON number")
    return value


class SessionCounts(_FrozenModel):
    active: NonNegativeControlInteger
    retained: NonNegativeControlInteger


class HealthResponse(_FrozenModel):
    protocol_version: Literal[1] = 1
    application_version: str
    pid: PositiveControlInteger
    started_at: datetime
    uptime_seconds: float
    capabilities: tuple[str, ...] = ("sessions", "agents")
    sessions: SessionCounts
    inactive_limit: NonNegativeControlInteger

    @field_validator("uptime_seconds", mode="before")
    @classmethod
    def validate_uptime_wire_type(cls, value: object) -> object:
        return _require_wire_duration(value)


class _ActivityResponse(_FrozenModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: str
    state: Literal["active", "idle", "failed"]
    active_requests: NonNegativeControlInteger
    requests: NonNegativeControlInteger
    client_model: str
    model: str
    provider: str
    transport: str
    effort: str
    context_window: PositiveControlInteger | None
    first_seen: datetime
    last_seen: datetime
    elapsed_seconds: float
    last_result: Literal["completed", "failed"] | None

    @field_validator("elapsed_seconds", mode="before")
    @classmethod
    def validate_elapsed_wire_type(cls, value: object) -> object:
        return _require_wire_duration(value)


class AgentResponse(_ActivityResponse):
    parent_id: str | None


class SessionResponse(_ActivityResponse):
    agents: tuple[AgentResponse, ...] = ()


class SessionListResponse(_FrozenModel):
    captured_at: datetime
    sessions: tuple[SessionResponse, ...]


def _require_wire_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("datetime must be an ISO string or datetime")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("datetime must be an ISO string or datetime") from error


def _normalize_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    try:
        return value.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("datetime is outside the supported UTC range") from error


def _require_safe_string(value: str) -> str:
    if not value.strip() or not value.isprintable():
        raise ValueError("value must be a nonblank printable string")
    return value


UTCDateTime = Annotated[
    datetime,
    BeforeValidator(_require_wire_datetime),
    AfterValidator(_normalize_utc_datetime),
]
SafeString = Annotated[
    str,
    Field(strict=True),
    AfterValidator(_require_safe_string),
]
StrictNumber: TypeAlias = StrictInt | StrictFloat
RequestOutcome = Literal[
    "active",
    "completed",
    "failed",
    "cancelled",
    "client_disconnected",
]
TerminalRequestOutcome = Literal[
    "completed",
    "failed",
    "cancelled",
    "client_disconnected",
]


def _require_finite_duration(value: int | float) -> int | float:
    if value < 0 or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("duration must be finite and non-negative")
    if isinstance(value, int) and value > MAX_CONTROL_INTEGER:
        raise ValueError("duration integer exceeds the control limit")
    return value


FiniteDuration = Annotated[
    StrictNumber,
    AfterValidator(_require_finite_duration),
]


class _TelemetryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        frozen=True,
    )


class _PerformanceActivityResponse(_TelemetryModel):
    id: SafeString
    state: Literal["active", "idle", "failed"]
    active_requests: NonNegativeControlInteger
    requests: NonNegativeControlInteger
    client_model: SafeString
    model: SafeString
    provider: SafeString
    transport: SafeString
    effort: SafeString
    context_window: PositiveControlInteger | None
    first_seen: UTCDateTime
    last_seen: UTCDateTime
    elapsed_seconds: FiniteDuration
    last_result: Literal["completed", "failed"] | None


class PerformanceAgentIdentityResponse(_PerformanceActivityResponse):
    parent_id: SafeString | None


class PerformanceActivityResponse(_PerformanceActivityResponse):
    pass


class PerformanceSessionIdentityResponse(_PerformanceActivityResponse):
    agents: tuple[PerformanceAgentIdentityResponse, ...] = ()


class ProcessIdentityResponse(_TelemetryModel):
    pid: PositiveControlInteger
    started_at: UTCDateTime


class MetricResponse(_TelemetryModel):
    status: Literal["observed", "unavailable", "not_applicable"]
    value: StrictNumber | None

    @model_validator(mode="after")
    def validate_status_value(self) -> Self:
        if self.status != "observed":
            if self.value is not None:
                raise ValueError("non-observed metric must not have a value")
            return self
        if self.value is None:
            raise ValueError("observed metric requires a value")
        _validate_metric_number(self.value, bound_integer=True)
        return self


class MetricAggregateResponse(_TelemetryModel):
    value: StrictNumber
    observed_samples: NonNegativeControlInteger
    unavailable_samples: NonNegativeControlInteger
    not_applicable_samples: NonNegativeControlInteger

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        _validate_metric_number(self.value, bound_integer=False)
        if self.observed_samples == 0 and self.value != 0:
            raise ValueError("aggregate value requires an observed sample")
        return self

    @property
    def partial(self) -> bool:
        return self.unavailable_samples > 0


class FailureDiagnosticResponse(_TelemetryModel):
    category: FailureCategory
    stage: FailureStage
    code: SafeString
    provider_code: SafeString | None = None
    exception_type: SafeString | None = None
    location: SafeString | None = None


class RequestPerformanceResponse(_TelemetryModel):
    id: SafeString
    session_id: SafeString
    operation: Literal["messages", "count_tokens"]
    outcome: RequestOutcome
    started_at: UTCDateTime
    finished_at: UTCDateTime | None
    duration: MetricResponse
    upstream_duration: MetricResponse
    ttft: MetricResponse
    input_tokens: MetricResponse
    output_tokens: MetricResponse
    cache_read_tokens: MetricResponse
    cache_creation_tokens: MetricResponse
    reasoning_tokens: MetricResponse
    tool_calls: MetricResponse
    retries: MetricResponse
    peak_concurrency: MetricResponse
    reasoning_continuation: Literal[
        "expected",
        "restored",
        "missing",
        "not_applicable",
        "unavailable",
    ]
    failure: FailureDiagnosticResponse | None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.outcome == "active" and self.finished_at is not None:
            raise ValueError("active request must not be finished")
        if self.outcome != "active" and self.finished_at is None:
            raise ValueError("terminal request must be finished")
        if self.outcome != "failed" and self.failure is not None:
            raise ValueError("only failed requests may include failure details")
        return self


def _validate_request_collection(
    requests: tuple[RequestPerformanceResponse, ...],
    session_id: str,
    *,
    active: bool,
) -> None:
    expected = "active" if active else "terminal"
    for request in requests:
        if request.session_id != session_id:
            raise ValueError(f"{expected} request must match session identity")
        if (request.outcome == "active") != active:
            raise ValueError(f"{expected} request has invalid lifecycle state")


def _latest_request(
    recent: tuple[RequestPerformanceResponse, ...],
    active: tuple[RequestPerformanceResponse, ...],
) -> RequestPerformanceResponse | None:
    if recent:
        return recent[0]
    if active:
        return active[0]
    return None


class SessionPerformanceResponse(_TelemetryModel):
    session_id: SafeString
    requests: NonNegativeControlInteger
    active_requests: tuple[RequestPerformanceResponse, ...]
    recent_requests: Annotated[
        tuple[RequestPerformanceResponse, ...],
        Field(max_length=20),
    ]
    outcomes: Mapping[TerminalRequestOutcome, NonNegativeControlInteger]
    input_tokens: MetricAggregateResponse
    output_tokens: MetricAggregateResponse
    cache_read_tokens: MetricAggregateResponse
    cache_creation_tokens: MetricAggregateResponse
    reasoning_tokens: MetricAggregateResponse
    tool_calls: MetricAggregateResponse
    retries: MetricAggregateResponse
    current_concurrency: NonNegativeControlInteger
    peak_concurrency: NonNegativeControlInteger
    latest_request: RequestPerformanceResponse | None

    @field_validator("outcomes", mode="after")
    @classmethod
    def freeze_outcomes(
        cls, value: Mapping[TerminalRequestOutcome, int]
    ) -> Mapping[TerminalRequestOutcome, int]:
        return MappingProxyType(dict(value))

    @field_serializer("outcomes")
    def serialize_outcomes(
        self, value: Mapping[TerminalRequestOutcome, int]
    ) -> dict[TerminalRequestOutcome, int]:
        return dict(value)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        _validate_request_collection(
            self.active_requests, self.session_id, active=True
        )
        _validate_request_collection(
            self.recent_requests, self.session_id, active=False
        )
        requests = self.active_requests + self.recent_requests
        if len({request.id for request in requests}) != len(requests):
            raise ValueError("request IDs must be unique within the session snapshot")
        if self.current_concurrency != len(self.active_requests):
            raise ValueError("current concurrency must equal active requests")
        if self.peak_concurrency < self.current_concurrency:
            raise ValueError("peak concurrency must cover current concurrency")
        completed = sum(self.outcomes.values())
        if self.requests != len(self.active_requests) + completed:
            raise ValueError("request count must match active and terminal requests")
        if self.latest_request != _latest_request(
            self.recent_requests, self.active_requests
        ):
            raise ValueError("latest request must match snapshot ordering")
        return self


def _validate_activity_performance(
    activity: _PerformanceActivityResponse,
    performance: SessionPerformanceResponse,
) -> None:
    if activity.id != performance.session_id:
        raise ValueError("activity identity must match performance identity")
    if activity.requests != performance.requests:
        raise ValueError("activity request counts must match")
    if activity.active_requests != performance.current_concurrency:
        raise ValueError("activity active request counts must match")
    activity_is_active = activity.state == "active"
    if activity_is_active != (performance.current_concurrency > 0):
        raise ValueError("activity state must match current concurrency")


class SessionPerformanceViewResponse(_TelemetryModel):
    session: PerformanceSessionIdentityResponse
    performance: SessionPerformanceResponse

    @model_validator(mode="after")
    def validate_session_identity(self) -> Self:
        _validate_activity_performance(self.session, self.performance)
        return self


class PerformanceListResponse(_TelemetryModel):
    process: ProcessIdentityResponse
    captured_at: UTCDateTime
    cursor: NonNegativeControlInteger
    sessions: tuple[SessionPerformanceViewResponse, ...]


OrdinaryPerformanceEventType = Literal[
    "request_started",
    "first_output",
    "progress",
    "tool_use",
    "retry",
    "completed",
    "failed",
    "cancelled",
    "client_disconnected",
]
_EVENT_OUTCOMES: dict[OrdinaryPerformanceEventType, RequestOutcome] = {
    "request_started": "active",
    "first_output": "active",
    "progress": "active",
    "tool_use": "active",
    "retry": "active",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "client_disconnected": "client_disconnected",
}


def _validate_event_request_position(
    request: RequestPerformanceResponse,
    session: SessionPerformanceResponse,
) -> None:
    if request.outcome == "active":
        matches = tuple(
            item for item in session.active_requests if item.id == request.id
        )
        if len(matches) != 1 or matches[0] != request:
            raise ValueError("active event must match active session request")
        return
    if not session.recent_requests:
        raise ValueError("terminal event requires a recent request")
    if session.recent_requests[0] != request:
        raise ValueError("terminal event must match latest finalized request")


def _validate_event_payload(
    event_type: OrdinaryPerformanceEventType,
    session_id: str,
    request: RequestPerformanceResponse,
    session: SessionPerformanceResponse,
) -> None:
    if request.session_id != session_id:
        raise ValueError("event identity must match request identity")
    if session.session_id != session_id:
        raise ValueError("event identity must match session identity")
    if request.outcome != _EVENT_OUTCOMES[event_type]:
        raise ValueError("event type must match request outcome")
    _validate_event_request_position(request, session)


class PerformanceEventResponse(_TelemetryModel):
    process: ProcessIdentityResponse
    sequence: PositiveControlInteger
    occurred_at: UTCDateTime
    type: OrdinaryPerformanceEventType
    session_id: SafeString
    activity: PerformanceActivityResponse
    request: RequestPerformanceResponse
    session: SessionPerformanceResponse

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        _validate_activity_performance(self.activity, self.session)
        _validate_event_payload(
            self.type,
            self.session_id,
            self.request,
            self.session,
        )
        return self


class PerformanceCursorResponse(_TelemetryModel):
    process: ProcessIdentityResponse
    sequence: PositiveControlInteger
    occurred_at: UTCDateTime
    type: Literal["cursor"]


class PerformanceResetResponse(_TelemetryModel):
    process: ProcessIdentityResponse
    sequence: NonNegativeControlInteger
    occurred_at: UTCDateTime
    type: Literal["reset"]
    snapshot: PerformanceListResponse

    @model_validator(mode="after")
    def validate_snapshot_identity(self) -> Self:
        if self.process != self.snapshot.process:
            raise ValueError("reset process must match snapshot process")
        if self.sequence != self.snapshot.cursor:
            raise ValueError("reset sequence must match snapshot cursor")
        return self


PerformanceStreamEvent: TypeAlias = Annotated[
    PerformanceEventResponse | PerformanceCursorResponse | PerformanceResetResponse,
    Field(discriminator="type"),
]


def _validate_metric_number(value: int | float, *, bound_integer: bool) -> None:
    if value < 0:
        raise ValueError("metric value must be non-negative")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metric value must be finite")
    if bound_integer and isinstance(value, int) and value > MAX_CONTROL_INTEGER:
        raise ValueError("metric integer exceeds the control limit")
