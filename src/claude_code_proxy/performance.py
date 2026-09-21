"""Provider-neutral request performance measurements and reduction."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import math
import sys
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias

from .domain.models import (
    CompletionResponse,
    RedactedThinking,
    RedactedThinkingBlock,
    StreamComplete,
    StreamError,
    StreamEvent,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolUseBlock,
    ToolUseStart,
    UsageField,
)
from .failures import FailureDiagnostic
from .limits import MAX_CONTROL_INTEGER

MetricStatus: TypeAlias = Literal["observed", "unavailable", "not_applicable"]
OperationKind: TypeAlias = Literal["messages", "count_tokens"]
RequestOutcome: TypeAlias = Literal[
    "active",
    "completed",
    "failed",
    "cancelled",
    "client_disconnected",
]
ReasoningContinuation: TypeAlias = Literal[
    "expected",
    "restored",
    "missing",
    "not_applicable",
    "unavailable",
]

logger = logging.getLogger(__name__)


class ProviderTelemetry(Protocol):
    """Provider-facing telemetry controls without request content fields."""

    def mark_retries_supported(self) -> None: ...

    def record_retry(self) -> None: ...

    def set_reasoning_continuation(
        self, value: ReasoningContinuation
    ) -> None: ...


class RequestTelemetry(ProviderTelemetry, Protocol):
    """Observe a provider request lifecycle without retaining its content."""

    def upstream_started(self) -> None: ...

    def upstream_finished(self) -> None: ...

    def stream_event(self, event: StreamEvent) -> None: ...

    def response(self, response: CompletionResponse) -> None: ...

    def count_tokens(self, value: int) -> None: ...


def notify_telemetry(
    telemetry: object | None, method_name: str, *args: object
) -> None:
    """Invoke one telemetry callback without affecting provider behavior."""
    if telemetry is None:
        return
    try:
        callback = getattr(telemetry, method_name)
        callback(*args)
    except Exception as error:
        logger.warning(
            "telemetry callback failed exception=%s",
            type(error).__name__,
        )


class _TelemetryRegistry(Protocol):
    def upstream_started(self, handle: object) -> None: ...

    def upstream_finished(self, handle: object) -> None: ...

    def stream_event(self, handle: object, event: StreamEvent) -> None: ...

    def response(self, handle: object, response: CompletionResponse) -> None: ...

    def count_tokens(self, handle: object, value: int) -> None: ...

    def mark_retries_supported(self, handle: object) -> None: ...

    def record_retry(self, handle: object) -> None: ...

    def set_reasoning_continuation(
        self, handle: object, value: ReasoningContinuation
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class RequestTelemetryObserver:
    """Delegate provider observations to one registry-owned request."""

    _registry: _TelemetryRegistry
    _handle: object

    def upstream_started(self) -> None:
        self._registry.upstream_started(self._handle)

    def upstream_finished(self) -> None:
        self._registry.upstream_finished(self._handle)

    def stream_event(self, event: StreamEvent) -> None:
        self._registry.stream_event(self._handle, event)

    def response(self, response: CompletionResponse) -> None:
        self._registry.response(self._handle, response)

    def count_tokens(self, value: int) -> None:
        self._registry.count_tokens(self._handle, value)

    def mark_retries_supported(self) -> None:
        self._registry.mark_retries_supported(self._handle)

    def record_retry(self) -> None:
        self._registry.record_retry(self._handle)

    def set_reasoning_continuation(
        self, value: ReasoningContinuation
    ) -> None:
        self._registry.set_reasoning_continuation(self._handle, value)


_METRIC_STATUSES = frozenset({"observed", "unavailable", "not_applicable"})
_OPERATIONS = frozenset({"messages", "count_tokens"})
_TERMINAL_OUTCOMES = frozenset({
    "completed",
    "failed",
    "cancelled",
    "client_disconnected",
})
_REASONING_CONTINUATIONS = frozenset({
    "expected",
    "restored",
    "missing",
    "not_applicable",
    "unavailable",
})
_MAX_TIME_MAGNITUDE = sys.float_info.max / 2
_AGGREGATE_METRICS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "tool_calls",
    "retries",
)


@dataclass(frozen=True, slots=True)
class Measurement:
    """Represent an observed value or why no value is available."""

    status: MetricStatus
    value: int | float | None

    def __post_init__(self) -> None:
        if self.status not in _METRIC_STATUSES:
            raise ValueError("invalid measurement status")
        if self.status != "observed":
            if self.value is not None:
                raise ValueError(f"{self.status} measurement must not have a value")
            return
        _validate_observed_value(self.value)

    @classmethod
    def observed(cls, value: int | float) -> "Measurement":
        return cls("observed", value)

    @classmethod
    def unavailable(cls) -> "Measurement":
        return cls("unavailable", None)

    @classmethod
    def not_applicable(cls) -> "Measurement":
        return cls("not_applicable", None)

    def model_value(self) -> dict[str, str | int | float | None]:
        return {"status": self.status, "value": self.value}


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """Summarize lifetime coverage and total value for one metric."""

    value: int | float
    observed_samples: int
    unavailable_samples: int
    not_applicable_samples: int

    def __post_init__(self) -> None:
        _validate_aggregate_value(self.value)
        for name in (
            "observed_samples",
            "unavailable_samples",
            "not_applicable_samples",
        ):
            _require_control_integer(name, getattr(self, name), minimum=0)

    @property
    def partial(self) -> bool:
        return self.unavailable_samples > 0


class _Aggregate:
    """Accumulate measurements without exposing mutable state."""

    __slots__ = (
        "_value",
        "_observed_samples",
        "_unavailable_samples",
        "_not_applicable_samples",
    )

    def __init__(self, initial: MetricAggregate | None = None) -> None:
        source = initial or MetricAggregate(0, 0, 0, 0)
        self._value = source.value
        self._observed_samples = source.observed_samples
        self._unavailable_samples = source.unavailable_samples
        self._not_applicable_samples = source.not_applicable_samples

    def add(self, measurement: Measurement) -> None:
        value = self._value
        observed = self._observed_samples
        unavailable = self._unavailable_samples
        not_applicable = self._not_applicable_samples
        if measurement.status == "observed":
            _validate_observed_value(measurement.value)
            assert measurement.value is not None
            value += measurement.value
            observed = _increment_control_integer("observed samples", observed)
        elif measurement.status == "unavailable":
            unavailable = _increment_control_integer(
                "unavailable samples", unavailable
            )
        elif measurement.status == "not_applicable":
            not_applicable = _increment_control_integer(
                "not applicable samples", not_applicable
            )
        else:
            raise ValueError("invalid measurement status")
        updated = MetricAggregate(value, observed, unavailable, not_applicable)
        self._value = updated.value
        self._observed_samples = updated.observed_samples
        self._unavailable_samples = updated.unavailable_samples
        self._not_applicable_samples = updated.not_applicable_samples

    def snapshot(self) -> MetricAggregate:
        return MetricAggregate(
            self._value,
            self._observed_samples,
            self._unavailable_samples,
            self._not_applicable_samples,
        )


def _validate_aggregate_value(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("aggregate value requires a finite non-negative number")
    if value < 0:
        raise ValueError("aggregate value requires a finite non-negative number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("aggregate value requires a finite non-negative number")


def _validate_observed_value(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("observed measurement requires a finite non-negative number")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("observed measurement requires a non-negative integer")
        if value > MAX_CONTROL_INTEGER:
            raise ValueError("observed measurement integer exceeds the control limit")
        return
    if value < 0 or not math.isfinite(value):
        raise ValueError("observed measurement requires a finite non-negative number")


@dataclass(frozen=True, slots=True)
class RequestPerformanceSnapshot:
    """Immutable safe summary of one request's performance."""

    id: str
    session_id: str
    operation: OperationKind
    outcome: RequestOutcome
    started_at: datetime
    finished_at: datetime | None
    duration: Measurement
    upstream_duration: Measurement
    ttft: Measurement
    input_tokens: Measurement
    output_tokens: Measurement
    cache_read_tokens: Measurement
    cache_creation_tokens: Measurement
    reasoning_tokens: Measurement
    tool_calls: Measurement
    retries: Measurement
    peak_concurrency: Measurement
    reasoning_continuation: ReasoningContinuation
    failure: FailureDiagnostic | None


@dataclass(frozen=True, slots=True)
class SessionPerformanceSnapshot:
    """Immutable summary of active, recent, and lifetime session metrics."""

    session_id: str
    requests: int
    active_requests: tuple[RequestPerformanceSnapshot, ...]
    recent_requests: tuple[RequestPerformanceSnapshot, ...]
    outcomes: Mapping[str, int]
    input_tokens: MetricAggregate
    output_tokens: MetricAggregate
    cache_read_tokens: MetricAggregate
    cache_creation_tokens: MetricAggregate
    reasoning_tokens: MetricAggregate
    tool_calls: MetricAggregate
    retries: MetricAggregate
    current_concurrency: int
    peak_concurrency: int
    latest_request: RequestPerformanceSnapshot | None


class RequestPerformance:
    """Reduce request lifecycle observations into a safe immutable snapshot."""

    __slots__ = (
        "_request_id",
        "_session_id",
        "_operation",
        "_started_at",
        "_started_monotonic",
        "_outcome",
        "_finished_at",
        "_finished_monotonic",
        "_upstream_started",
        "_upstream_finished",
        "_first_output",
        "_input_tokens",
        "_output_tokens",
        "_cache_read_tokens",
        "_cache_creation_tokens",
        "_reasoning_tokens",
        "_tool_calls",
        "_retries",
        "_peak_concurrency",
        "_reasoning_continuation",
        "_failure",
    )

    def __init__(
        self,
        request_id: str,
        session_id: str,
        operation: OperationKind,
        started_at: datetime,
        started_monotonic: float,
        initial_concurrency: int,
    ) -> None:
        _require_operation(operation)
        _require_utc_datetime("started_at", started_at)
        started = _require_finite_time("started_monotonic", started_monotonic)
        _require_positive_control_integer("initial_concurrency", initial_concurrency)

        self._request_id = request_id
        self._session_id = session_id
        self._operation = operation
        self._started_at = started_at
        self._started_monotonic = started
        self._outcome: RequestOutcome = "active"
        self._finished_at: datetime | None = None
        self._finished_monotonic: float | None = None
        self._upstream_started: float | None = None
        self._upstream_finished: float | None = None
        self._first_output: float | None = None
        self._initialize_metrics(operation)
        self._retries: int | None = None
        self._peak_concurrency = initial_concurrency
        self._failure: FailureDiagnostic | None = None

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def operation(self) -> OperationKind:
        return self._operation

    @property
    def started_monotonic(self) -> float:
        return self._started_monotonic

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    def final_snapshot(self) -> RequestPerformanceSnapshot:
        if not self._is_terminal:
            raise ValueError("final snapshot requires a terminal request")
        assert self._finished_monotonic is not None
        return self.snapshot(self._finished_monotonic)

    def _initialize_metrics(self, operation: OperationKind) -> None:
        self._input_tokens = Measurement.unavailable()
        if operation == "count_tokens":
            self._output_tokens = Measurement.not_applicable()
            self._cache_read_tokens = Measurement.not_applicable()
            self._cache_creation_tokens = Measurement.not_applicable()
            self._reasoning_tokens = Measurement.not_applicable()
            self._tool_calls = Measurement.not_applicable()
            self._reasoning_continuation: ReasoningContinuation = "not_applicable"
            return
        self._output_tokens = Measurement.unavailable()
        self._cache_read_tokens = Measurement.unavailable()
        self._cache_creation_tokens = Measurement.unavailable()
        self._reasoning_tokens = Measurement.unavailable()
        self._tool_calls = Measurement.observed(0)
        self._reasoning_continuation = "unavailable"

    def would_mark_upstream_started(self) -> bool:
        return not self._is_terminal and self._upstream_started is None

    def would_mark_upstream_finished(self) -> bool:
        return (
            not self._is_terminal
            and self._upstream_started is not None
            and self._upstream_finished is None
        )

    def would_mark_stream_output(self, event: StreamEvent) -> bool:
        return self._would_mark_first_output(_is_semantic_event(event))

    def would_mark_response_output(self, response: CompletionResponse) -> bool:
        meaningful = any(
            _is_meaningful_block(block) for block in response.content
        )
        return self._would_mark_first_output(meaningful)

    def mark_upstream_started(self, now: float) -> bool:
        sampled = _require_finite_time("upstream start", now)
        if self._is_terminal or self._upstream_started is not None:
            return False
        self._upstream_started = sampled
        return True

    def mark_upstream_finished(self, now: float) -> bool:
        sampled = _require_finite_time("upstream finish", now)
        if self._is_terminal or self._upstream_finished is not None:
            return False
        if self._upstream_started is None:
            raise ValueError("cannot finish upstream before upstream start")
        self._upstream_finished = sampled
        return True

    def observe_stream_event(self, event: StreamEvent, now: float) -> bool:
        sampled = _require_finite_time("stream event time", now)
        if self._is_terminal:
            return False
        if isinstance(event, ToolUseStart):
            self._increment_tool_calls()
        if isinstance(event, StreamComplete):
            self.record_usage(event.usage)
        if isinstance(event, StreamError):
            self._failure = event.diagnostic
        return self._mark_first_output(sampled, _is_semantic_event(event))

    def observe_response(self, response: CompletionResponse, now: float) -> bool:
        sampled = _require_finite_time("response time", now)
        if self._is_terminal:
            return False
        meaningful = any(_is_meaningful_block(block) for block in response.content)
        tool_count = sum(isinstance(block, ToolUseBlock) for block in response.content)
        self.record_usage(response.usage)
        if self._operation == "messages":
            self._tool_calls = Measurement.observed(tool_count)
        self._mark_first_output(sampled, meaningful)
        return meaningful

    def record_usage(self, usage: TokenUsage) -> None:
        if self._is_terminal:
            return
        input_tokens = _usage_measurement(usage, "input_tokens")
        if self._operation == "count_tokens":
            self._input_tokens = input_tokens
            return
        output_tokens = _usage_measurement(usage, "output_tokens")
        cache_read_tokens = _usage_measurement(
            usage, "cache_read_input_tokens"
        )
        cache_creation_tokens = _usage_measurement(
            usage, "cache_creation_input_tokens"
        )
        reasoning_tokens = _usage_measurement(usage, "thinking_tokens")
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cache_read_tokens = cache_read_tokens
        self._cache_creation_tokens = cache_creation_tokens
        self._reasoning_tokens = reasoning_tokens

    def mark_retries_supported(self) -> None:
        if not self._is_terminal and self._retries is None:
            self._retries = 0

    def record_retry(self) -> None:
        if self._is_terminal:
            return
        self.mark_retries_supported()
        assert self._retries is not None
        self._retries = _increment_control_integer("retries", self._retries)

    def set_concurrency(self, concurrency: int) -> None:
        _require_positive_control_integer("concurrency", concurrency)
        if not self._is_terminal:
            self._peak_concurrency = max(self._peak_concurrency, concurrency)

    def set_reasoning_continuation(
        self,
        continuation: ReasoningContinuation,
    ) -> None:
        if continuation not in _REASONING_CONTINUATIONS:
            raise ValueError("invalid reasoning continuation")
        if not self._is_terminal and self._operation == "messages":
            self._reasoning_continuation = continuation

    def finish(
        self,
        outcome: RequestOutcome,
        finished_at: datetime,
        finished_monotonic: float,
        failure: FailureDiagnostic | None = None,
    ) -> bool:
        if outcome not in _TERMINAL_OUTCOMES:
            raise ValueError("finish requires a terminal outcome")
        if self._is_terminal:
            return False
        _require_utc_datetime("finished_at", finished_at)
        finished = _require_finite_time(
            "finished_monotonic", finished_monotonic
        )
        effective_failure = failure if failure is not None else self._failure
        if outcome != "failed" and effective_failure is not None:
            raise ValueError("non-failed outcome cannot retain a failure diagnostic")
        self._outcome = outcome
        self._finished_at = finished_at
        self._finished_monotonic = finished
        if self._upstream_started is not None and self._upstream_finished is None:
            self._upstream_finished = finished
        self._failure = effective_failure
        return True

    def snapshot(self, now: float) -> RequestPerformanceSnapshot:
        sampled = _require_finite_time("snapshot time", now)
        total_end = self._finished_monotonic
        if total_end is None:
            total_end = sampled
        return RequestPerformanceSnapshot(
            id=self._request_id,
            session_id=self._session_id,
            operation=self._operation,
            outcome=self._outcome,
            started_at=self._started_at,
            finished_at=self._finished_at,
            duration=_duration(self._started_monotonic, total_end),
            upstream_duration=self._upstream_duration(sampled),
            ttft=self._ttft(),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cache_read_tokens=self._cache_read_tokens,
            cache_creation_tokens=self._cache_creation_tokens,
            reasoning_tokens=self._reasoning_tokens,
            tool_calls=self._tool_calls,
            retries=self._retry_measurement(),
            peak_concurrency=Measurement.observed(self._peak_concurrency),
            reasoning_continuation=self._reasoning_continuation,
            failure=self._failure,
        )

    @property
    def _is_terminal(self) -> bool:
        return self._outcome != "active"

    def _increment_tool_calls(self) -> None:
        if self._operation != "messages":
            return
        count = self._tool_calls.value
        assert isinstance(count, int)
        self._tool_calls = Measurement.observed(
            _increment_control_integer("tool calls", count)
        )

    def _would_mark_first_output(self, meaningful: bool) -> bool:
        return (
            meaningful
            and self._operation == "messages"
            and self._first_output is None
            and not self._is_terminal
        )

    def _mark_first_output(self, now: float, meaningful: bool) -> bool:
        if not meaningful or self._operation != "messages":
            return False
        if self._first_output is not None:
            return False
        self._first_output = now
        return True

    def _upstream_duration(self, now: float) -> Measurement:
        if self._upstream_started is None:
            return Measurement.unavailable()
        end = self._upstream_finished
        if end is None:
            end = now
        return _duration(self._upstream_started, end)

    def _ttft(self) -> Measurement:
        if self._operation == "count_tokens":
            return Measurement.not_applicable()
        if self._first_output is None:
            return Measurement.unavailable()
        return _duration(self._started_monotonic, self._first_output)

    def _retry_measurement(self) -> Measurement:
        if self._retries is None:
            return Measurement.unavailable()
        return Measurement.observed(self._retries)


class SessionPerformance:
    """Reduce request snapshots into bounded history and lifetime metrics."""

    __slots__ = (
        "_session_id",
        "_pending",
        "_recent",
        "_requests",
        "_outcomes",
        "_aggregates",
        "_peak_concurrency",
    )

    def __init__(self, session_id: str, history_limit: int = 20) -> None:
        self._session_id = _require_safe_identifier("session_id", session_id)
        _require_positive_control_integer("history_limit", history_limit)
        self._pending: dict[str, RequestPerformance] = {}
        self._recent: deque[RequestPerformanceSnapshot] = deque(
            maxlen=history_limit
        )
        self._requests = 0
        self._outcomes: dict[str, int] = {}
        self._aggregates = {name: _Aggregate() for name in _AGGREGATE_METRICS}
        self._peak_concurrency = 0

    def start(self, request: RequestPerformance) -> int:
        if request.session_id != self._session_id:
            raise ValueError("request session does not match session")
        if request.is_terminal:
            raise ValueError("request must be active when started")
        request_id = _require_safe_identifier("request_id", request.request_id)
        if request_id in self._pending:
            raise ValueError("duplicate pending request ID")
        requests = _increment_control_integer("requests", self._requests)
        self._pending[request_id] = request
        live = self._live_requests()
        concurrency = len(live)
        for active in live:
            active.set_concurrency(concurrency)
        self._requests = requests
        self._peak_concurrency = max(self._peak_concurrency, concurrency)
        return concurrency

    def request(self, request_id: str) -> RequestPerformance | None:
        """Return one exact reducer for registry coordination only."""
        return self._pending.get(request_id)

    def add_finalized(
        self, request: RequestPerformance
    ) -> RequestPerformanceSnapshot | None:
        if request.session_id != self._session_id:
            return None
        pending = self._pending.get(request.request_id)
        if pending is not request:
            return None
        if not request.is_terminal:
            raise ValueError("request must be terminal before finalization")
        snapshot = request.final_snapshot()
        outcome = snapshot.outcome
        outcome_count = _increment_control_integer(
            "outcome count", self._outcomes.get(outcome, 0)
        )
        aggregates = self._updated_aggregates(snapshot)
        del self._pending[request.request_id]
        self._recent.appendleft(snapshot)
        self._outcomes[outcome] = outcome_count
        self._aggregates = aggregates
        return snapshot

    def _live_requests(self) -> tuple[RequestPerformance, ...]:
        return tuple(
            request
            for request in self._pending.values()
            if not request.is_terminal
        )

    def _updated_aggregates(
        self, snapshot: RequestPerformanceSnapshot
    ) -> dict[str, _Aggregate]:
        updated: dict[str, _Aggregate] = {}
        for name in _AGGREGATE_METRICS:
            aggregate = _Aggregate(self._aggregates[name].snapshot())
            aggregate.add(getattr(snapshot, name))
            updated[name] = aggregate
        return updated

    def snapshot(self, now: float) -> SessionPerformanceSnapshot:
        sampled = _require_finite_time("snapshot time", now)
        active = tuple(
            sorted(
                (request.snapshot(sampled) for request in self._live_requests()),
                key=lambda item: (item.started_at, item.id),
                reverse=True,
            )
        )
        recent = tuple(self._recent)
        latest = recent[0] if recent else (active[0] if active else None)
        aggregates = {
            name: aggregate.snapshot()
            for name, aggregate in self._aggregates.items()
        }
        return SessionPerformanceSnapshot(
            session_id=self._session_id,
            requests=self._requests,
            active_requests=active,
            recent_requests=recent,
            outcomes=MappingProxyType(dict(self._outcomes)),
            input_tokens=aggregates["input_tokens"],
            output_tokens=aggregates["output_tokens"],
            cache_read_tokens=aggregates["cache_read_tokens"],
            cache_creation_tokens=aggregates["cache_creation_tokens"],
            reasoning_tokens=aggregates["reasoning_tokens"],
            tool_calls=aggregates["tool_calls"],
            retries=aggregates["retries"],
            current_concurrency=len(active),
            peak_concurrency=self._peak_concurrency,
            latest_request=latest,
        )


def validate_clock_sample(
    wall_time: datetime, monotonic_time: object
) -> float:
    """Validate one wall/monotonic pair before coordinated mutation."""
    _require_utc_datetime("wall clock", wall_time)
    return _require_finite_time("monotonic clock", monotonic_time)


def _is_semantic_event(event: StreamEvent) -> bool:
    if isinstance(event, TextDelta):
        return bool(event.text)
    return isinstance(event, (RedactedThinking, ToolUseStart))


def _is_meaningful_block(block: object) -> bool:
    if isinstance(block, TextBlock):
        return bool(block.text)
    return isinstance(block, (RedactedThinkingBlock, ToolUseBlock))


def _usage_measurement(usage: TokenUsage, field: UsageField) -> Measurement:
    observed_fields = usage.observed_fields or frozenset()
    if field not in observed_fields:
        return Measurement.unavailable()
    return Measurement.observed(getattr(usage, field))


def _duration(start: float, end: float) -> Measurement:
    return Measurement.observed(max(0.0, end - start))


def _require_operation(operation: object) -> None:
    if operation not in _OPERATIONS:
        raise ValueError("invalid operation")


def _require_utc_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be an aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an aware UTC datetime")


def _require_finite_time(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    try:
        sampled = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(sampled) or abs(sampled) > _MAX_TIME_MAGNITUDE:
        raise ValueError(f"{name} must be safely subtractable")
    return sampled


def _require_safe_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or not value.isprintable():
        raise ValueError(f"{name} must be a nonblank printable string")
    return value


def _require_control_integer(
    name: str, value: object, *, minimum: int
) -> None:
    if type(value) is not int or not minimum <= value <= MAX_CONTROL_INTEGER:
        raise ValueError(
            f"{name} must be an integer between {minimum} and "
            f"{MAX_CONTROL_INTEGER}"
        )


def _require_positive_control_integer(name: str, value: object) -> None:
    _require_control_integer(name, value, minimum=1)


def _increment_control_integer(name: str, value: int) -> int:
    if value >= MAX_CONTROL_INTEGER:
        raise ValueError(f"{name} exceeds the control limit")
    return value + 1
