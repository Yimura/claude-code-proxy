"""Mechanics and privacy-safe reporting for opt-in Codex Agent evaluation."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Literal, Self
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from claude_code_proxy.control.client import ControlClient
from claude_code_proxy.control.socket import resolve_socket_path

REPORT_SCHEMA_VERSION = 1
MAX_TRIALS = 100
DEFAULT_TRIALS = 5
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_BASE_URL = "http://127.0.0.1:8082"
MAX_REPORT_BYTES = 1024 * 1024
EVAL_SCENARIO_NAMES = (
    "authorized_recursion",
    "cancellation_without_completion",
    "conflicting_parent",
    "consolidated_reviewer_reuse",
    "depth_bound_denial",
    "discovery_sufficiency",
    "missing_completion_unavailable",
    "push_completion",
    "ten_task_broad_owners",
    "unauthorized_recursion",
    "user_preference_escalation",
    "worker_ownership_and_reuse",
)
MetricStatus = Literal["observed", "unavailable"]
_FAILURE_CODES = frozenset({
    "agent_completion_polled",
    "authorized_recursion_missing",
    "cancellation_misclassified",
    "conflicting_parent_allowed",
    "depth_limit_allowed",
    "duplicate_agent_launch",
    "excessive_discovery",
    "excessive_owner_count",
    "fragmented_ownership",
    "incomplete_outcome",
    "lifecycle_claimed",
    "preference_not_escalated",
    "reviewer_not_reused",
    "scenario_failure",
    "unauthorized_recursion",
    "unexpected_tool",
    "worker_not_reused",
})
_METRIC_KEYS = frozenset({"status", "value"})
_TRIAL_UNAVAILABLE_KEYS = (
    "maximum_observed_depth",
    "input_tokens",
    "output_tokens",
)
_TOOL_COUNT_KEYS = ("agent", "send_message")
EVAL_MAX_DEPTH = 2


@dataclass(frozen=True, slots=True)
class EvalLineageStep:
    agent_id: str
    parent_agent_id: str | None
    agent_requested: bool


AUTHORIZED_RECURSION_STEPS = (
    EvalLineageStep("authorized-parent", None, True),
    EvalLineageStep("authorized-child", "authorized-parent", True),
)
DEPTH_DENIAL_STEPS = (
    EvalLineageStep("depth-parent", None, False),
    EvalLineageStep("depth-child", "depth-parent", True),
)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    enabled: bool
    trials: int
    model: str
    report_path: Path | None
    base_url: str

    @classmethod
    def from_environment(cls) -> Self:
        trials = _positive_integer_environment("CODEX_AGENT_EVAL_TRIALS", DEFAULT_TRIALS)
        model = _nonblank_environment("CODEX_AGENT_EVAL_MODEL", DEFAULT_MODEL)
        report_raw = os.environ.get("CODEX_AGENT_EVAL_REPORT")
        report_path = None if report_raw is None else _report_path(report_raw)
        base_url = _base_url(os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL))
        return cls(
            enabled=os.environ.get("RUN_CODEX_AGENT_EVAL") == "1",
            trials=trials,
            model=model,
            report_path=report_path,
            base_url=base_url,
        )


@dataclass(frozen=True, slots=True)
class MetricValue:
    status: MetricStatus
    value: int | float | None

    def __post_init__(self) -> None:
        if self.status not in ("observed", "unavailable"):
            raise ValueError("metric status must be observed or unavailable")
        if self.status == "unavailable":
            if self.value is not None:
                raise ValueError("unavailable metric value must be null")
            return
        _require_non_negative_number(self.value, "observed metric value")

    @classmethod
    def observed(cls, value: int | float) -> Self:
        return cls("observed", value)

    @classmethod
    def unavailable(cls) -> Self:
        return cls("unavailable", None)

    def to_json_object(self) -> dict[str, object]:
        return {"status": self.status, "value": self.value}

    @classmethod
    def from_json_object(cls, value: object) -> Self:
        row = _strict_object(value, _METRIC_KEYS, "metric")
        return cls(status=row["status"], value=row["value"])


@dataclass(frozen=True, slots=True)
class TrialResult:
    passed: bool
    complete: bool
    agent_calls: int
    send_message_calls: int
    maximum_observed_depth: MetricValue
    maximum_discovery_streak: int
    duplicate_launches: int
    input_tokens: MetricValue
    output_tokens: MetricValue
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or type(self.complete) is not bool:
            raise ValueError("trial outcomes must be booleans")
        for name in (
            "agent_calls",
            "send_message_calls",
            "maximum_discovery_streak",
            "duplicate_launches",
        ):
            _require_non_negative_integer(getattr(self, name), name)
        if self.failure_code is not None and self.failure_code not in _FAILURE_CODES:
            raise ValueError("failure_code is not recognized")
        if self.passed and (not self.complete or self.failure_code is not None):
            raise ValueError("a passing trial must be complete without a failure code")


@dataclass(frozen=True, slots=True)
class ScenarioAggregate:
    name: str
    trial_count: int
    passed_trials: int
    complete_trials: int
    pass_rate: float
    completeness_rate: float
    tool_counts: Mapping[str, int]
    input_tokens: MetricValue
    output_tokens: MetricValue
    maximum_observed_depth: MetricValue
    maximum_discovery_streak: int
    reuse_calls: int
    duplicate_launches: int
    unavailable_counts: Mapping[str, int]
    failure_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_name(self.name, "scenario name")
        _require_positive_integer(self.trial_count, "trial_count", MAX_TRIALS)
        for name in (
            "passed_trials",
            "complete_trials",
            "maximum_discovery_streak",
            "reuse_calls",
            "duplicate_launches",
        ):
            _require_non_negative_integer(getattr(self, name), name)
        if self.passed_trials > self.trial_count or self.complete_trials > self.trial_count:
            raise ValueError("scenario outcome counts exceed trial_count")
        _require_rate(self.pass_rate, "pass_rate")
        _require_rate(self.completeness_rate, "completeness_rate")
        if self.pass_rate != self.passed_trials / self.trial_count:
            raise ValueError("pass_rate does not match passed_trials")
        if self.completeness_rate != self.complete_trials / self.trial_count:
            raise ValueError("completeness_rate does not match complete_trials")
        _require_exact_counts(self.tool_counts, _TOOL_COUNT_KEYS, "tool_counts")
        _require_exact_counts(
            self.unavailable_counts, _TRIAL_UNAVAILABLE_KEYS, "unavailable_counts"
        )
        object.__setattr__(self, "tool_counts", MappingProxyType(dict(self.tool_counts)))
        object.__setattr__(
            self,
            "unavailable_counts",
            MappingProxyType(dict(self.unavailable_counts)),
        )
        if tuple(sorted(set(self.failure_codes))) != self.failure_codes:
            raise ValueError("failure_codes must be unique and sorted")
        if any(code not in _FAILURE_CODES for code in self.failure_codes):
            raise ValueError("failure_codes contains an unrecognized code")

    def to_json_object(self) -> dict[str, object]:
        return {
            "name": self.name,
            "trial_count": self.trial_count,
            "passed_trials": self.passed_trials,
            "complete_trials": self.complete_trials,
            "pass_rate": self.pass_rate,
            "completeness_rate": self.completeness_rate,
            "tool_counts": dict(self.tool_counts),
            "input_tokens": self.input_tokens.to_json_object(),
            "output_tokens": self.output_tokens.to_json_object(),
            "maximum_observed_depth": self.maximum_observed_depth.to_json_object(),
            "maximum_discovery_streak": self.maximum_discovery_streak,
            "reuse_calls": self.reuse_calls,
            "duplicate_launches": self.duplicate_launches,
            "unavailable_counts": dict(self.unavailable_counts),
            "failure_codes": list(self.failure_codes),
        }

    @classmethod
    def from_json_object(cls, value: object) -> Self:
        keys = frozenset({
            "name", "trial_count", "passed_trials", "complete_trials",
            "pass_rate", "completeness_rate", "tool_counts", "input_tokens",
            "output_tokens", "maximum_observed_depth", "maximum_discovery_streak",
            "reuse_calls", "duplicate_launches", "unavailable_counts",
            "failure_codes",
        })
        row = _strict_object(value, keys, "scenario")
        failure_codes = row["failure_codes"]
        if not isinstance(failure_codes, list) or not all(
            isinstance(code, str) for code in failure_codes
        ):
            raise ValueError("scenario failure_codes must be an array of strings")
        return cls(
            name=row["name"],
            trial_count=row["trial_count"],
            passed_trials=row["passed_trials"],
            complete_trials=row["complete_trials"],
            pass_rate=row["pass_rate"],
            completeness_rate=row["completeness_rate"],
            tool_counts=_object_counts(row["tool_counts"], "tool_counts"),
            input_tokens=MetricValue.from_json_object(row["input_tokens"]),
            output_tokens=MetricValue.from_json_object(row["output_tokens"]),
            maximum_observed_depth=MetricValue.from_json_object(
                row["maximum_observed_depth"]
            ),
            maximum_discovery_streak=row["maximum_discovery_streak"],
            reuse_calls=row["reuse_calls"],
            duplicate_launches=row["duplicate_launches"],
            unavailable_counts=_object_counts(
                row["unavailable_counts"], "unavailable_counts"
            ),
            failure_codes=tuple(failure_codes),
        )


@dataclass(frozen=True, slots=True)
class EvalReport:
    schema_version: int
    mode: str
    model: str
    trial_count: int
    pass_rate: float
    completeness_rate: float
    scenarios: tuple[ScenarioAggregate, ...]
    active_workers: MetricValue
    revision_deduplication: MetricValue

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if self.mode not in ("off", "advisory", "enforce"):
            raise ValueError("mode must be off, advisory, or enforce")
        _require_name(self.model, "model")
        _require_positive_integer(self.trial_count, "trial_count", MAX_TRIALS)
        _require_rate(self.pass_rate, "pass_rate")
        _require_rate(self.completeness_rate, "completeness_rate")
        names = tuple(item.name for item in self.scenarios)
        if names != EVAL_SCENARIO_NAMES:
            raise ValueError("scenario names must match the evaluation allowlist")
        if any(item.trial_count != self.trial_count for item in self.scenarios):
            raise ValueError("scenario trial counts must match report trial_count")
        expected_pass_rate = sum(item.passed_trials for item in self.scenarios) / (
            self.trial_count * len(self.scenarios)
        )
        expected_completeness_rate = sum(
            item.complete_trials for item in self.scenarios
        ) / (self.trial_count * len(self.scenarios))
        if self.pass_rate != expected_pass_rate:
            raise ValueError("pass_rate does not match scenario outcomes")
        if self.completeness_rate != expected_completeness_rate:
            raise ValueError("completeness_rate does not match scenario outcomes")
        if self.active_workers != MetricValue.unavailable():
            raise ValueError("active_workers must be unavailable")
        if self.revision_deduplication != MetricValue.unavailable():
            raise ValueError("revision_deduplication must be unavailable")

    def to_json_object(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "model": self.model,
            "trial_count": self.trial_count,
            "pass_rate": self.pass_rate,
            "completeness_rate": self.completeness_rate,
            "scenarios": [item.to_json_object() for item in self.scenarios],
            "active_workers": self.active_workers.to_json_object(),
            "revision_deduplication": self.revision_deduplication.to_json_object(),
        }

    @classmethod
    def from_json_object(cls, value: object) -> Self:
        keys = frozenset({
            "schema_version", "mode", "model", "trial_count", "pass_rate",
            "completeness_rate", "scenarios", "active_workers",
            "revision_deduplication",
        })
        row = _strict_object(value, keys, "report")
        scenarios = row["scenarios"]
        if not isinstance(scenarios, list):
            raise ValueError("report scenarios must be an array")
        return cls(
            schema_version=row["schema_version"],
            mode=row["mode"],
            model=row["model"],
            trial_count=row["trial_count"],
            pass_rate=row["pass_rate"],
            completeness_rate=row["completeness_rate"],
            scenarios=tuple(
                ScenarioAggregate.from_json_object(item) for item in scenarios
            ),
            active_workers=MetricValue.from_json_object(row["active_workers"]),
            revision_deduplication=MetricValue.from_json_object(
                row["revision_deduplication"]
            ),
        )


def aggregate_report(
    *, mode: str, model: str, results: Mapping[str, Sequence[TrialResult]]
) -> EvalReport:
    if not results:
        raise ValueError("results must not be empty")
    trial_counts = {len(rows) for rows in results.values()}
    if len(trial_counts) != 1:
        raise ValueError("every scenario must have the same trial count")
    trial_count = next(iter(trial_counts))
    _require_positive_integer(trial_count, "trial_count", MAX_TRIALS)
    scenarios = tuple(
        _aggregate_scenario(name, tuple(results[name])) for name in sorted(results)
    )
    total_trials = trial_count * len(scenarios)
    return EvalReport(
        schema_version=REPORT_SCHEMA_VERSION,
        mode=mode,
        model=model,
        trial_count=trial_count,
        pass_rate=sum(row.passed_trials for row in scenarios) / total_trials,
        completeness_rate=(
            sum(row.complete_trials for row in scenarios) / total_trials
        ),
        scenarios=scenarios,
        active_workers=MetricValue.unavailable(),
        revision_deduplication=MetricValue.unavailable(),
    )


def _aggregate_scenario(name: str, rows: tuple[TrialResult, ...]) -> ScenarioAggregate:
    return ScenarioAggregate(
        name=name,
        trial_count=len(rows),
        passed_trials=sum(row.passed for row in rows),
        complete_trials=sum(row.complete for row in rows),
        pass_rate=sum(row.passed for row in rows) / len(rows),
        completeness_rate=sum(row.complete for row in rows) / len(rows),
        tool_counts={
            "agent": sum(row.agent_calls for row in rows),
            "send_message": sum(row.send_message_calls for row in rows),
        },
        input_tokens=_sum_metric(rows, "input_tokens"),
        output_tokens=_sum_metric(rows, "output_tokens"),
        maximum_observed_depth=_maximum_metric(rows, "maximum_observed_depth"),
        maximum_discovery_streak=max(row.maximum_discovery_streak for row in rows),
        reuse_calls=sum(row.send_message_calls for row in rows),
        duplicate_launches=sum(row.duplicate_launches for row in rows),
        unavailable_counts={
            key: sum(getattr(row, key).status == "unavailable" for row in rows)
            for key in _TRIAL_UNAVAILABLE_KEYS
        },
        failure_codes=tuple(sorted({
            row.failure_code for row in rows if row.failure_code is not None
        })),
    )


def write_report_atomic(path: Path, report: EvalReport) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report.to_json_object(), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def private_orchestration_mode() -> str:
    socket_path = resolve_socket_path(None)
    with ControlClient(socket_path) as client:
        return client.health().orchestration_mode


def authorize_nesting(raw_session_id: str, *, max_depth: int) -> None:
    socket_path = resolve_socket_path(None)
    with ControlClient(socket_path) as client:
        client.allow_nesting(raw_session_id, max_depth=max_depth, duration_seconds=3600)


def revoke_nesting(raw_session_id: str) -> None:
    socket_path = resolve_socket_path(None)
    with ControlClient(socket_path) as client:
        client.revoke_nesting(raw_session_id)


def unique_raw_session(trial_number: int) -> str:
    return f"codex-agent-eval-{trial_number}-{uuid4().hex}"


def send_message(
    config: EvalConfig,
    messages: list[dict[str, object]],
    tools: Sequence[Mapping[str, object]],
    raw_session_id: str,
    *,
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
) -> dict[str, object]:
    headers = {
        "content-type": "application/json",
        "x-claude-code-session-id": raw_session_id,
    }
    if agent_id is not None:
        headers["x-claude-code-agent-id"] = agent_id
    if parent_agent_id is not None:
        headers["x-claude-code-parent-agent-id"] = parent_agent_id
    response = httpx.post(
        f"{config.base_url}/v1/messages",
        headers=headers,
        json={
            "model": config.model,
            "max_tokens": 2_048,
            "messages": messages,
            "tools": list(tools),
        },
        timeout=180.0,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("live evaluation response must be an object")
    return payload


def tool_calls(response: Mapping[str, object]) -> list[dict[str, object]]:
    content = response.get("content")
    if not isinstance(content, list):
        raise ValueError("live evaluation response content must be an array")
    return [
        block for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def tool_result(call: Mapping[str, object], content: str) -> dict[str, object]:
    return {"type": "tool_result", "tool_use_id": call["id"], "content": content}


def response_usage(response: Mapping[str, object]) -> tuple[MetricValue, MetricValue]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return MetricValue.unavailable(), MetricValue.unavailable()
    return _usage_metric(usage.get("input_tokens")), _usage_metric(
        usage.get("output_tokens")
    )


def classify_duplicate_agent_inputs(
    calls: Sequence[Mapping[str, object]],
) -> int:
    """Compare canonical Agent inputs ephemerally; retain only the duplicate count."""
    seen: set[str] = set()
    duplicates = 0
    for call in calls:
        if call.get("name") != "Agent":
            continue
        raw_input = call.get("input")
        canonical = json.dumps(raw_input, sort_keys=True, separators=(",", ":"))
        if canonical in seen:
            duplicates += 1
        seen.add(canonical)
    return duplicates


def build_trial_result(
    *,
    responses: list[dict[str, object]],
    calls: list[dict[str, object]],
    passed: bool,
    complete: bool,
    failure_code: str | None,
    depth: MetricValue | None = None,
    discovery: int = 0,
) -> TrialResult:
    """Reduce runtime state to allowlisted metrics, then discard that state."""
    input_tokens, output_tokens = _aggregate_response_usage(responses)
    agent_calls = sum(call.get("name") == "Agent" for call in calls)
    send_message_calls = sum(call.get("name") == "SendMessage" for call in calls)
    duplicate_launches = classify_duplicate_agent_inputs(calls)
    responses.clear()
    calls.clear()
    return TrialResult(
        passed=passed,
        complete=complete,
        agent_calls=agent_calls,
        send_message_calls=send_message_calls,
        maximum_observed_depth=depth or MetricValue.unavailable(),
        maximum_discovery_streak=discovery,
        duplicate_launches=duplicate_launches,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        failure_code=failure_code,
    )


def _aggregate_response_usage(
    responses: Sequence[Mapping[str, object]],
) -> tuple[MetricValue, MetricValue]:
    inputs = []
    outputs = []
    for response in responses:
        input_value, output_value = response_usage(response)
        inputs.append(input_value)
        outputs.append(output_value)
    return _sum_usage(inputs), _sum_usage(outputs)


def _sum_usage(values: Sequence[MetricValue]) -> MetricValue:
    if not values or any(value.status == "unavailable" for value in values):
        return MetricValue.unavailable()
    return MetricValue.observed(sum(value.value for value in values))


def _sum_metric(rows: Sequence[TrialResult], field: str) -> MetricValue:
    values = [getattr(row, field) for row in rows]
    if any(value.status == "unavailable" for value in values):
        return MetricValue.unavailable()
    return MetricValue.observed(sum(value.value for value in values))


def _maximum_metric(rows: Sequence[TrialResult], field: str) -> MetricValue:
    observed = [
        getattr(row, field).value
        for row in rows
        if getattr(row, field).status == "observed"
    ]
    if not observed:
        return MetricValue.unavailable()
    return MetricValue.observed(max(observed))


def _usage_metric(value: object) -> MetricValue:
    if type(value) is not int or value < 0:
        return MetricValue.unavailable()
    return MetricValue.observed(value)


def _positive_integer_environment(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} must be an exact positive integer")
    value = int(raw)
    if not 1 <= value <= MAX_TRIALS:
        raise ValueError(f"{name} must be between 1 and {MAX_TRIALS}")
    return value


def _nonblank_environment(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    _require_name(value, name)
    return value


def _report_path(value: str) -> Path:
    if not value or value != value.strip() or "\0" in value:
        raise ValueError("CODEX_AGENT_EVAL_REPORT must be a nonblank path")
    return Path(value)


def _base_url(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("ANTHROPIC_BASE_URL must be a nonblank HTTP URL")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("ANTHROPIC_BASE_URL must be an HTTP URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("ANTHROPIC_BASE_URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("ANTHROPIC_BASE_URL must not contain query or fragment")
    return value.rstrip("/")


def _strict_object(
    value: object, expected_keys: frozenset[str], label: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected_keys:
        unexpected = sorted(actual - expected_keys)
        missing = sorted(expected_keys - actual)
        raise ValueError(
            f"{label} fields invalid: unexpected={unexpected}, missing={missing}"
        )
    return value


def _object_counts(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _require_exact_counts(
    value: Mapping[str, int], expected: Sequence[str], label: str
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label} fields are invalid")
    for count in value.values():
        _require_non_negative_integer(count, label)


def _require_name(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonblank string without surrounding whitespace")
    if len(value) > 256 or not value.isprintable():
        raise ValueError(f"{label} must be printable and at most 256 characters")


def _require_non_negative_integer(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _require_positive_integer(value: object, label: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be between 1 and {maximum}")


def _require_non_negative_number(value: object, label: str) -> None:
    if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite non-negative number")


def _require_rate(value: object, label: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be between zero and one")
