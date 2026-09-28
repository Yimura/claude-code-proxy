"""Live mechanics for opt-in Codex Agent evaluation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from claude_code_proxy.codex_agent_eval_report import (
    MAX_TRIALS,
    MetricValue,
    TrialResult,
)
from claude_code_proxy.control.client import ControlClient
from claude_code_proxy.control.socket import resolve_socket_path

DEFAULT_TRIALS = 5
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_BASE_URL = "http://127.0.0.1:8082"

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


def _usage_metric(value: object) -> MetricValue:
    if type(value) is not int or value < 0:
        return MetricValue.unavailable()
    return MetricValue.observed(value)


def _require_name(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{label} must be a nonblank string without surrounding whitespace"
        )
    if len(value) > 256 or not value.isprintable():
        raise ValueError(f"{label} must be printable and at most 256 characters")
