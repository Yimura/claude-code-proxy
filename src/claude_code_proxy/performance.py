"""Provider-neutral request performance measurements and reduction."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Literal, TypeAlias

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


def _validate_observed_value(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("observed measurement requires a finite non-negative number")
    if value < 0 or not math.isfinite(value):
        raise ValueError("observed measurement requires a finite non-negative number")
    if isinstance(value, int) and value > MAX_CONTROL_INTEGER:
        raise ValueError("observed measurement integer exceeds the control limit")


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
        self._input_tokens = _usage_measurement(usage, "input_tokens")
        if self._operation == "count_tokens":
            return
        self._output_tokens = _usage_measurement(usage, "output_tokens")
        self._cache_read_tokens = _usage_measurement(
            usage, "cache_read_input_tokens"
        )
        self._cache_creation_tokens = _usage_measurement(
            usage, "cache_creation_input_tokens"
        )
        self._reasoning_tokens = _usage_measurement(usage, "thinking_tokens")

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
        self._outcome = outcome
        self._finished_at = finished_at
        self._finished_monotonic = finished
        if self._upstream_started is not None and self._upstream_finished is None:
            self._upstream_finished = finished
        self._failure = failure if failure is not None else self._failure
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
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_positive_control_integer(name: str, value: object) -> None:
    if type(value) is not int or not 1 <= value <= MAX_CONTROL_INTEGER:
        raise ValueError(
            f"{name} must be an integer between 1 and {MAX_CONTROL_INTEGER}"
        )


def _increment_control_integer(name: str, value: int) -> int:
    if value >= MAX_CONTROL_INTEGER:
        raise ValueError(f"{name} exceeds the control limit")
    return value + 1
