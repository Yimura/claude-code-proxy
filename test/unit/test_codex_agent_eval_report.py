import json
from pathlib import Path

import pytest

from claude_code_proxy.config import CodexOrchestrationMode
from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionRequest,
    Message,
    TextBlock,
    ToolDefinition,
)
from claude_code_proxy.public_identity import PublicIdentity
from claude_code_proxy.providers.codex.orchestration_policy import (
    OrchestrationDecisionCode,
    OrchestrationPolicyCoordinator,
)
from claude_code_proxy.providers.codex.orchestration_registry import (
    AuthorizationStatus,
    OrchestrationRegistry,
)
from claude_code_proxy.reasoning import ReasoningPolicy
from test.integration import test_codex_agent_polling as live_eval
from test.integration.codex_agent_eval_support import (
    AUTHORIZED_RECURSION_STEPS,
    DEPTH_DENIAL_STEPS,
    EVAL_MAX_DEPTH,
    EVAL_SCENARIO_NAMES,
    EvalConfig,
    EvalReport,
    MetricValue,
    ScenarioAggregate,
    TrialResult,
    aggregate_report,
    build_trial_result,
    tool_calls,
    write_report_atomic,
)


def test_eval_config_defaults_to_five_trials(monkeypatch):
    monkeypatch.setenv("RUN_CODEX_AGENT_EVAL", "1")
    monkeypatch.delenv("CODEX_AGENT_EVAL_TRIALS", raising=False)
    monkeypatch.delenv("CODEX_AGENT_EVAL_MODEL", raising=False)
    monkeypatch.delenv("CODEX_AGENT_EVAL_REPORT", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    config = EvalConfig.from_environment()

    assert config.enabled is True
    assert config.trials == 5
    assert config.model == "claude-opus-5"
    assert config.report_path is None
    assert config.base_url == "http://127.0.0.1:8082"


def test_eval_config_parses_every_operator_setting(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    monkeypatch.setenv("RUN_CODEX_AGENT_EVAL", "1")
    monkeypatch.setenv("CODEX_AGENT_EVAL_TRIALS", "100")
    monkeypatch.setenv("CODEX_AGENT_EVAL_MODEL", "custom-model")
    monkeypatch.setenv("CODEX_AGENT_EVAL_REPORT", str(report))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9000/")

    config = EvalConfig.from_environment()

    assert config == EvalConfig(
        enabled=True,
        trials=100,
        model="custom-model",
        report_path=report,
        base_url="http://127.0.0.1:9000",
    )


@pytest.mark.parametrize("value", ["", "0", "101", "+1", "1.0", " 1", "1 ", "x"])
def test_eval_config_rejects_non_exact_or_out_of_range_trial_count(
    monkeypatch, value
):
    monkeypatch.setenv("CODEX_AGENT_EVAL_TRIALS", value)

    with pytest.raises(ValueError, match="CODEX_AGENT_EVAL_TRIALS"):
        EvalConfig.from_environment()


def test_eval_is_disabled_unless_flag_is_exactly_one(monkeypatch):
    monkeypatch.setenv("RUN_CODEX_AGENT_EVAL", "true")

    assert EvalConfig.from_environment().enabled is False


def test_trial_builder_discards_runtime_transcript_before_returning_result():
    marker = "PRIVATE_AGENT_INPUT_MARKER"
    responses = [{
        "usage": {"input_tokens": 3, "output_tokens": 2},
        "content": [{
            "type": "tool_use",
            "id": "tool-1",
            "name": "Agent",
            "input": {"prompt": marker},
        }],
    }]
    calls = [responses[0]["content"][0]]

    result = build_trial_result(
        responses=responses,
        calls=calls,
        passed=True,
        complete=True,
        failure_code=None,
    )

    assert responses == []
    assert calls == []
    assert result.agent_calls == 1
    assert marker not in repr(result)


def test_report_serialization_is_allowlisted_and_discards_private_markers():
    marker = "PRIVATE_PROMPT_AND_SESSION_MARKER"
    raw_agent_input = {"prompt": marker}
    result = TrialResult(
        passed=False,
        complete=True,
        agent_calls=2,
        send_message_calls=1,
        maximum_observed_depth=MetricValue.observed(2),
        maximum_discovery_streak=3,
        duplicate_launches=int(raw_agent_input == {"prompt": marker}),
        input_tokens=MetricValue.observed(100),
        output_tokens=MetricValue.unavailable(),
        failure_code="duplicate_agent_launch",
    )
    del raw_agent_input

    report = aggregate_report(
        mode="enforce",
        model="model-safe",
        results={name: (result,) for name in EVAL_SCENARIO_NAMES},
    )
    encoded = json.dumps(report.to_json_object(), sort_keys=True)

    assert marker not in encoded
    assert "transcript" not in encoded
    assert report.active_workers == MetricValue.unavailable()
    assert report.revision_deduplication == MetricValue.unavailable()
    assert report.scenarios[0].failure_codes == ("duplicate_agent_launch",)


def test_report_round_trip_is_strict_and_frozen():
    report = _report()

    restored = EvalReport.from_json_object(report.to_json_object())

    assert restored == report
    with pytest.raises((AttributeError, TypeError)):
        restored.model = "changed"
    with pytest.raises(TypeError):
        restored.scenarios[0].tool_counts["agent"] = 9
    invalid = report.to_json_object() | {"unexpected": "field"}
    with pytest.raises(ValueError, match="unexpected"):
        EvalReport.from_json_object(invalid)


def test_report_rejects_boolean_schema_version_and_inconsistent_rates():
    payload = _report().to_json_object()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        EvalReport.from_json_object(payload)

    payload = _report().to_json_object()
    payload["scenarios"][0]["pass_rate"] = 0.5
    with pytest.raises(ValueError, match="pass_rate"):
        EvalReport.from_json_object(payload)


def test_metric_unavailable_cannot_carry_value():
    with pytest.raises(ValueError, match="unavailable"):
        MetricValue(status="unavailable", value=1)


def test_atomic_write_replaces_target_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("old", encoding="utf-8")

    write_report_atomic(path, _report())

    assert EvalReport.from_json_object(json.loads(path.read_text())) == _report()
    assert list(tmp_path.iterdir()) == [path]


def test_aggregate_scores_completeness_before_efficiency():
    complete_expensive = TrialResult(
        passed=True,
        complete=True,
        agent_calls=2,
        send_message_calls=1,
        maximum_observed_depth=MetricValue.observed(1),
        maximum_discovery_streak=2,
        duplicate_launches=0,
        input_tokens=MetricValue.observed(100),
        output_tokens=MetricValue.observed(50),
    )
    incomplete_cheap = TrialResult(
        passed=False,
        complete=False,
        agent_calls=0,
        send_message_calls=0,
        maximum_observed_depth=MetricValue.unavailable(),
        maximum_discovery_streak=0,
        duplicate_launches=0,
        input_tokens=MetricValue.observed(1),
        output_tokens=MetricValue.observed(1),
        failure_code="incomplete_outcome",
    )

    report = aggregate_report(
        mode="advisory",
        model="model-safe",
        results={
            name: (complete_expensive, incomplete_cheap)
            for name in EVAL_SCENARIO_NAMES
        },
    )

    scenario = report.scenarios[0]
    assert report.completeness_rate == 0.5
    assert scenario.completeness_rate == 0.5
    assert scenario.input_tokens == MetricValue.observed(101)
    assert scenario.unavailable_counts == {
        "maximum_observed_depth": 1,
        "input_tokens": 0,
        "output_tokens": 0,
    }


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_report_requires_exact_scenario_allowlist(mutation):
    payload = _report().to_json_object()
    if mutation == "missing":
        payload["scenarios"].pop()
    elif mutation == "extra":
        extra = dict(payload["scenarios"][0])
        extra["name"] = "unexpected_scenario"
        payload["scenarios"].append(extra)
    else:
        payload["scenarios"].append(dict(payload["scenarios"][0]))

    with pytest.raises(ValueError, match="scenario names"):
        EvalReport.from_json_object(payload)


def test_reviewer_reuse_rejects_different_worker():
    calls = [{
        "name": "SendMessage",
        "input": {"to": "review-2", "message": "recheck"},
    }]

    assert not live_eval._reviewer_reused(calls)
    calls[0]["input"]["to"] = "review-1"
    assert live_eval._reviewer_reused(calls)


def test_user_escalation_rejects_self_selected_preference_text():
    response = {
        "content": [{
            "type": "text",
            "text": "I prefer JSON and selected it without asking.",
        }]
    }

    assert not live_eval._requires_user_input(tool_calls(response))
    assert live_eval._requires_user_input([{
        "name": "RequestUserInput",
        "input": {"requires_user_input": True},
    }])


def test_completion_unavailable_rejects_noted_as_successful():
    false_positive = [{
        "name": "RecordCompletionState",
        "input": {"completion_state": "noted as successful"},
    }]

    assert not live_eval._completion_unavailable(false_positive)
    assert live_eval._completion_unavailable([{
        "name": "RecordCompletionState",
        "input": {"completion_state": "unavailable"},
    }])


def test_authorized_recursion_steps_allow_one_delegation_then_deny_deeper():
    results, registry, public_session = _run_depth_steps(AUTHORIZED_RECURSION_STEPS)
    decisions = [result.decision for result in results]

    assert [decision.code for decision in decisions] == [
        OrchestrationDecisionCode.NESTED_ALLOWED,
        OrchestrationDecisionCode.DEPTH_LIMIT_REACHED,
    ]
    assert [decision.depth.value for decision in decisions] == [1, 2]
    assert _has_agent(results[0])
    assert not _has_agent(results[1])
    assert registry.authorization(public_session).status is AuthorizationStatus.ABSENT


def test_depth_denial_steps_observe_parent_then_deny_agent_bearing_child():
    results, registry, public_session = _run_depth_steps(DEPTH_DENIAL_STEPS)
    decisions = [result.decision for result in results]

    assert [decision.code for decision in decisions] == [
        OrchestrationDecisionCode.NESTED_ALLOWED,
        OrchestrationDecisionCode.DEPTH_LIMIT_REACHED,
    ]
    assert [decision.depth.value for decision in decisions] == [1, 2]
    assert not _has_agent(results[0])
    assert not _has_agent(results[1])
    assert registry.authorization(public_session).status is AuthorizationStatus.ABSENT


def _run_depth_steps(steps):
    identity = PublicIdentity(secret=b"eval-depth-test")
    registry = OrchestrationRegistry(identity=identity)
    policy = OrchestrationPolicyCoordinator(
        CodexOrchestrationMode.ENFORCE,
        identity,
        registry,
    )
    raw_session = "eval-depth-session"
    authorization = registry.authorize_raw_session(
        raw_session,
        max_depth=EVAL_MAX_DEPTH,
        duration_seconds=60,
    )
    assert authorization.status is AuthorizationStatus.ACTIVE
    try:
        results = [
            policy.reconcile(_depth_request(raw_session, step))
            for step in steps
        ]
    finally:
        registry.revoke_raw_session(raw_session)
    return results, registry, identity.public_id(raw_session)


def _has_agent(result):
    return any(tool.name == "Agent" for tool in result.request.tools)


def _depth_request(raw_session, step):
    tools = (ToolDefinition("RecordDecision"),)
    if step.agent_requested:
        tools = (ToolDefinition("Agent", "Delegate."), *tools)
    return CompletionRequest(
        original_model="claude",
        model="openai/gpt",
        response_model="claude",
        max_tokens=100,
        messages=(Message("user", (TextBlock("work"),)),),
        reasoning=ReasoningPolicy(None, None),
        client_identity=ClientIdentity(
            raw_session,
            step.agent_id,
            step.parent_agent_id,
        ),
        tools=tools,
    )


def _report() -> EvalReport:
    scenarios = tuple(
        ScenarioAggregate(
            name=name,
            trial_count=1,
            passed_trials=1,
            complete_trials=1,
            pass_rate=1.0,
            completeness_rate=1.0,
            tool_counts={"agent": 1, "send_message": 0},
            input_tokens=MetricValue.observed(10),
            output_tokens=MetricValue.observed(5),
            maximum_observed_depth=MetricValue.observed(1),
            maximum_discovery_streak=2,
            reuse_calls=0,
            duplicate_launches=0,
            unavailable_counts={
                "maximum_observed_depth": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            failure_codes=(),
        )
        for name in EVAL_SCENARIO_NAMES
    )
    return EvalReport(
        schema_version=1,
        mode="advisory",
        model="model-safe",
        trial_count=1,
        pass_rate=1.0,
        completeness_rate=1.0,
        scenarios=scenarios,
        active_workers=MetricValue.unavailable(),
        revision_deduplication=MetricValue.unavailable(),
    )
