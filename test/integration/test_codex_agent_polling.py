"""Billable, opt-in repeated evaluation of Codex Agent orchestration."""

from collections.abc import Callable, Sequence

import pytest

from claude_code_proxy.codex_agent_eval_report import (
    EVAL_SCENARIO_NAMES,
    MetricValue,
    TrialResult,
    aggregate_report,
    write_report_atomic,
)
from test.integration.codex_agent_eval_support import (
    AUTHORIZED_RECURSION_STEPS,
    DEPTH_DENIAL_STEPS,
    EVAL_MAX_DEPTH,
    EvalConfig,
    authorize_nesting,
    build_trial_result,
    private_orchestration_mode,
    revoke_nesting,
    send_message,
    tool_calls,
    tool_result,
    unique_raw_session,
)

CONFIG = EvalConfig.from_environment()
pytestmark = pytest.mark.skipif(
    not CONFIG.enabled,
    reason="set RUN_CODEX_AGENT_EVAL=1 to run billable live evaluation",
)


def _tool(name, description, properties, required):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _closed_signal_tool(name, description, field, value):
    value_type = "boolean" if type(value) is bool else "string"
    tool = _tool(
        name,
        description,
        {field: {"type": value_type, "enum": [value]}},
        [field],
    )
    tool["input_schema"]["additionalProperties"] = False
    return tool


AGENT_TOOL = _tool(
    "Agent",
    "Launch a background worker.",
    {"prompt": {"type": "string"}},
    ["prompt"],
)
TASK_OUTPUT_TOOL = _tool(
    "TaskOutput",
    "Retrieve explicit output from a non-Agent background task.",
    {"task_id": {"type": "string"}},
    ["task_id"],
)
SEND_MESSAGE_TOOL = _tool(
    "SendMessage",
    "Send follow-up work to an existing worker.",
    {"to": {"type": "string"}, "message": {"type": "string"}},
    ["to", "message"],
)
RECORD_WORK_TOOL = _tool(
    "RecordIndependentWork",
    "Record completion of independent parent work.",
    {"summary": {"type": "string"}},
    ["summary"],
)
RECORD_DECISION_TOOL = _tool(
    "RecordDecision",
    "Record completed direct analysis or a bounded escalation.",
    {"summary": {"type": "string"}},
    ["summary"],
)
REQUEST_USER_INPUT_TOOL = _closed_signal_tool(
    "RequestUserInput",
    "Record that a user preference is required before work can continue.",
    "requires_user_input",
    True,
)
RECORD_COMPLETION_STATE_TOOL = _closed_signal_tool(
    "RecordCompletionState",
    "Record the authoritative availability of worker completion state.",
    "completion_state",
    "unavailable",
)
SEARCH_REPOSITORY_TOOL = _tool(
    "SearchRepository",
    "Search local repository evidence.",
    {"query": {"type": "string"}},
    ["query"],
)
SEARCH_WEB_TOOL = _tool(
    "SearchWeb",
    "Search external documentation only when local evidence cannot answer.",
    {"query": {"type": "string"}},
    ["query"],
)
Scenario = Callable[[str, str], TrialResult]


def test_repeated_privacy_safe_codex_agent_evaluation():
    if CONFIG.report_path is None:
        pytest.fail("CODEX_AGENT_EVAL_REPORT is required for live evaluation")
    mode = private_orchestration_mode()
    scenario_by_name: dict[str, Scenario] = {
        "authorized_recursion": _authorized_recursion,
        "cancellation_without_completion": _cancellation_without_completion,
        "conflicting_parent": _conflicting_parent,
        "consolidated_reviewer_reuse": _consolidated_reviewer_reuse,
        "depth_bound_denial": _depth_bound_denial,
        "discovery_sufficiency": _discovery_sufficiency,
        "missing_completion_unavailable": _missing_completion_unavailable,
        "push_completion": _push_completion,
        "ten_task_broad_owners": _ten_task_broad_owners,
        "unauthorized_recursion": _unauthorized_recursion,
        "user_preference_escalation": _user_preference_escalation,
        "worker_ownership_and_reuse": _worker_ownership_and_reuse,
    }
    scenarios = tuple(
        (name, scenario_by_name[name]) for name in EVAL_SCENARIO_NAMES
    )
    results = {name: [] for name, _scenario in scenarios}

    for trial_number in range(CONFIG.trials):
        for name, scenario in scenarios:
            raw_session = unique_raw_session(trial_number)
            try:
                result = scenario(raw_session, mode)
            except Exception:
                result = _result(False, False, failure_code="scenario_failure")
            results[name].append(result)

    report = aggregate_report(
        mode=mode,
        model=CONFIG.model,
        results={name: tuple(rows) for name, rows in results.items()},
    )
    try:
        write_report_atomic(CONFIG.report_path, report)
    except OSError as error:
        pytest.fail(f"evaluation completed but report write failed: {type(error).__name__}")
    failures = [
        name
        for name, rows in results.items()
        if any(not result.passed for result in rows)
    ]
    assert not failures, f"live evaluation failed scenarios: {sorted(failures)}"


def _push_completion(raw_session, _mode):
    messages = [{
        "role": "user",
        "content": (
            "Launch one Agent to analyze module A. While it runs, use "
            "RecordIndependentWork for module B. Completion arrives automatically; "
            "do not poll or wait."
        ),
    }]
    responses = []
    calls_seen = []
    agent_seen = work_seen = False
    for _ in range(3):
        response = _send(messages, [AGENT_TOOL, TASK_OUTPUT_TOOL, RECORD_WORK_TOOL], raw_session)
        responses.append(response)
        calls = tool_calls(response)
        calls_seen.extend(calls)
        if any(call.get("name") == "TaskOutput" for call in calls):
            return _from_responses(responses, calls_seen, False, True, "agent_completion_polled")
        messages.append({"role": "assistant", "content": response["content"]})
        results = []
        for call in calls:
            if call.get("name") == "Agent":
                agent_seen = True
                content = "Agent launched as agent-eval-1 and remains running."
            elif call.get("name") == "RecordIndependentWork":
                work_seen = True
                content = "Independent module B work recorded."
            else:
                return _from_responses(responses, calls_seen, False, True, "unexpected_tool")
            results.append(tool_result(call, content))
        if results:
            messages.append({"role": "user", "content": results})
        if agent_seen and work_seen:
            break
    complete = agent_seen and work_seen
    messages.append({
        "role": "user",
        "content": "<task-notification>Agent agent-eval-1 completed: module A is clean.</task-notification>",
    })
    final = _send(messages, [AGENT_TOOL, TASK_OUTPUT_TOOL, RECORD_WORK_TOOL], raw_session)
    responses.append(final)
    final_calls = tool_calls(final)
    calls_seen.extend(final_calls)
    passed = complete and not final_calls
    return _from_responses(
        responses,
        calls_seen,
        passed,
        complete,
        None if passed else "incomplete_outcome",
    )


def _worker_ownership_and_reuse(raw_session, _mode):
    messages = [{
        "role": "user",
        "content": (
            "Use Agent delegation where useful to implement parser validation, its "
            "focused tests, and resulting fixes. They share one subsystem and file set."
        ),
    }]
    first = _send(messages, [AGENT_TOOL], raw_session)
    first_calls = tool_calls(first)
    agents = [call for call in first_calls if call.get("name") == "Agent"]
    if len(agents) != 1:
        code = "fragmented_ownership" if len(agents) > 1 else "incomplete_outcome"
        return _from_responses([first], first_calls, False, bool(agents), code)
    messages.extend([
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": [tool_result(agents[0], "Worker owner-1 completed the parser batch.")]},
        {"role": "user", "content": "A related parser edge case needs a fix. Reuse the existing owner."},
    ])
    follow_up = _send(messages, [AGENT_TOOL, SEND_MESSAGE_TOOL], raw_session)
    calls = first_calls + tool_calls(follow_up)
    reused = any(call.get("name") == "SendMessage" for call in tool_calls(follow_up))
    replaced = any(call.get("name") == "Agent" for call in tool_calls(follow_up))
    passed = reused and not replaced
    return _from_responses(
        [first, follow_up], calls, passed, True,
        None if passed else "worker_not_reused",
    )


def _unauthorized_recursion(raw_session, _mode):
    response = _send(
        [{"role": "user", "content": "Inspect the assigned parser behavior directly and record your decision. No recursive delegation was authorized."}],
        [AGENT_TOOL, RECORD_DECISION_TOOL],
        raw_session,
        agent_id="child",
        parent_agent_id="parent",
    )
    calls = tool_calls(response)
    agent_seen = any(call.get("name") == "Agent" for call in calls)
    decision_seen = any(call.get("name") == "RecordDecision" for call in calls)
    passed = not agent_seen and decision_seen
    return _from_responses(
        [response], calls, passed, decision_seen,
        None if passed else "unauthorized_recursion",
        depth=MetricValue.unavailable(),
    )


def _discovery_sufficiency(raw_session, _mode):
    tools = [SEARCH_REPOSITORY_TOOL, SEARCH_WEB_TOOL, RECORD_DECISION_TOOL]
    messages = [{"role": "user", "content": "Determine where parser timeout is configured and what test proves a change. Use available evidence, then record the decision."}]
    responses = []
    calls_seen = []
    streak = repository_calls = web_calls = 0
    decision_seen = evidence_delivered = False
    for _ in range(8):
        response = _send(messages, tools, raw_session)
        responses.append(response)
        calls = tool_calls(response)
        calls_seen.extend(calls)
        messages.append({"role": "assistant", "content": response["content"]})
        if any(call.get("name") == "RecordDecision" for call in calls):
            decision_seen = evidence_delivered
            break
        streak += 1
        results = []
        for call in calls:
            if call.get("name") == "SearchRepository":
                repository_calls += 1
                evidence_delivered = True
                content = "Behavior: 30s. Location: src/parser.py:42. Preserve cancellation. Verify with test_parser_timeout."
            elif call.get("name") == "SearchWeb":
                web_calls += 1
                content = "External search is unnecessary; the repository owns this setting."
            else:
                return _from_responses(responses, calls_seen, False, True, "unexpected_tool", discovery=streak)
            results.append(tool_result(call, content))
        if results:
            messages.append({"role": "user", "content": results})
    passed = decision_seen and repository_calls >= 1 and streak < 8 and web_calls <= 1
    return _from_responses(
        responses, calls_seen, passed, decision_seen,
        None if passed else "excessive_discovery",
        discovery=streak,
    )


def _authorized_recursion(raw_session, mode):
    allowed_step, denied_step = AUTHORIZED_RECURSION_STEPS
    authorize_nesting(raw_session, max_depth=EVAL_MAX_DEPTH)
    responses = []
    calls_seen = []
    try:
        allowed_response = _send(
            [{"role": "user", "content": "Delegation is explicitly authorized. Launch exactly one Agent for the independent nested check."}],
            [AGENT_TOOL, RECORD_DECISION_TOOL],
            raw_session,
            agent_id=allowed_step.agent_id,
            parent_agent_id=allowed_step.parent_agent_id,
        )
        responses.append(allowed_response)
        allowed_calls = tool_calls(allowed_response)
        calls_seen.extend(allowed_calls)
        allowed = sum(call.get("name") == "Agent" for call in allowed_calls) == 1
        denied_response = _send(
            [{"role": "user", "content": "Attempt to launch exactly one Agent for a deeper independent check. If Agent is unavailable at the authorization depth limit, record that bounded denial directly."}],
            [AGENT_TOOL, RECORD_DECISION_TOOL],
            raw_session,
            agent_id=denied_step.agent_id,
            parent_agent_id=denied_step.parent_agent_id,
        )
        responses.append(denied_response)
        denied_calls = tool_calls(denied_response)
        calls_seen.extend(denied_calls)
        denied = not any(call.get("name") == "Agent" for call in denied_calls)
        passed = allowed and denied
        return _from_responses(
            responses, calls_seen, passed, True,
            None if passed else "authorized_recursion_missing",
            depth=MetricValue.observed(2) if mode != "off" else MetricValue.unavailable(),
        )
    finally:
        revoke_nesting(raw_session)


def _ten_task_broad_owners(raw_session, _mode):
    response = _send(
        [{"role": "user", "content": "Plan and dispatch these ten related tasks: parser implementation/tests/fixes, transport implementation/tests/fixes, docs, config, integration verification, and final review. Use at most three broad coherent owners; do not make one worker per checklist item."}],
        [AGENT_TOOL, RECORD_DECISION_TOOL],
        raw_session,
    )
    calls = tool_calls(response)
    owner_count = sum(call.get("name") == "Agent" for call in calls)
    passed = 1 <= owner_count <= 3
    return _from_responses(
        [response], calls, passed, owner_count > 0,
        None if passed else "excessive_owner_count",
    )


def _consolidated_reviewer_reuse(raw_session, _mode):
    first = _send(
        [{"role": "user", "content": "After the meaningful implementation batch, launch one reviewer for correctness and maintainability."}],
        [AGENT_TOOL], raw_session,
    )
    agents = [call for call in tool_calls(first) if call.get("name") == "Agent"]
    if len(agents) != 1:
        return _from_responses([first], tool_calls(first), False, bool(agents), "reviewer_not_reused")
    messages = [
        {"role": "user", "content": "After the meaningful implementation batch, launch one reviewer for correctness and maintainability."},
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": [tool_result(agents[0], "Reviewer review-1 found one issue.")]},
        {"role": "user", "content": "The issue is fixed. Ask the same reviewer to recheck rather than launching another."},
    ]
    second = _send(messages, [AGENT_TOOL, SEND_MESSAGE_TOOL], raw_session)
    second_calls = tool_calls(second)
    passed = _reviewer_reused(second_calls)
    return _from_responses(
        [first, second], tool_calls(first) + second_calls, passed, True,
        None if passed else "reviewer_not_reused",
    )


def _user_preference_escalation(raw_session, _mode):
    response = _send(
        [{"role": "user", "content": "The repository supports either JSON or YAML for the new operator report and has no convention deciding between them. Do not choose for me; use RequestUserInput to record that my preference is required."}],
        [REQUEST_USER_INPUT_TOOL], raw_session,
    )
    calls = tool_calls(response)
    passed = _requires_user_input(calls)
    return _from_responses(
        [response], calls, passed, True,
        None if passed else "preference_not_escalated",
    )


def _cancellation_without_completion(raw_session, _mode):
    first = _send(
        [{"role": "user", "content": "Launch one Agent for analysis. Request cancellation is not evidence that the worker completed."}],
        [AGENT_TOOL], raw_session,
    )
    agents = [call for call in tool_calls(first) if call.get("name") == "Agent"]
    if len(agents) != 1:
        return _from_responses([first], tool_calls(first), False, False, "incomplete_outcome")
    messages = [
        {"role": "user", "content": "Launch one Agent for analysis. Request cancellation is not evidence that the worker completed."},
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": [tool_result(agents[0], "The parent request was cancelled. No worker completion event was observed.")]},
        {"role": "user", "content": "Record what can be concluded about worker completion."},
    ]
    second = _send(messages, [RECORD_COMPLETION_STATE_TOOL], raw_session)
    second_calls = tool_calls(second)
    calls = tool_calls(first) + second_calls
    passed = _completion_unavailable(second_calls)
    return _from_responses(
        [first, second], calls, passed, True,
        None if passed else "cancellation_misclassified",
    )


def _missing_completion_unavailable(raw_session, _mode):
    response = _send(
        [{"role": "user", "content": "No worker completion notification exists. Use RecordCompletionState to record that completion state is unavailable; request activity is not worker lifecycle."}],
        [RECORD_COMPLETION_STATE_TOOL], raw_session,
    )
    calls = tool_calls(response)
    passed = _completion_unavailable(calls)
    return _from_responses(
        [response], calls, passed, True,
        None if passed else "lifecycle_claimed",
    )


def _conflicting_parent(raw_session, mode):
    _observe_parent(raw_session, "parent-a")
    _observe_parent(raw_session, "parent-b")
    first = _send(
        [{"role": "user", "content": "Inspect directly and record the lineage decision."}],
        [AGENT_TOOL, RECORD_DECISION_TOOL], raw_session,
        agent_id="child", parent_agent_id="parent-a",
    )
    second = _send(
        [{"role": "user", "content": "This caller supplies a conflicting parent. Do not recursively delegate; record the decision."}],
        [AGENT_TOOL, RECORD_DECISION_TOOL], raw_session,
        agent_id="child", parent_agent_id="parent-b",
    )
    calls = tool_calls(first) + tool_calls(second)
    denied = not any(call.get("name") == "Agent" for call in tool_calls(second))
    return _from_responses(
        [first, second], calls, denied, True,
        None if denied else "conflicting_parent_allowed",
        depth=MetricValue.unavailable() if mode != "off" else MetricValue.unavailable(),
    )


def _depth_bound_denial(raw_session, mode):
    observed_step, denied_step = DEPTH_DENIAL_STEPS
    authorize_nesting(raw_session, max_depth=EVAL_MAX_DEPTH)
    try:
        _observe_parent(raw_session, observed_step.agent_id)
        response = _send(
            [{"role": "user", "content": "Attempt to launch one Agent for a deeper independent check. If the depth bound removed Agent, record the denial directly."}],
            [AGENT_TOOL, RECORD_DECISION_TOOL], raw_session,
            agent_id=denied_step.agent_id,
            parent_agent_id=denied_step.parent_agent_id,
        )
        calls = tool_calls(response)
        denied = not any(call.get("name") == "Agent" for call in calls)
        return _from_responses(
            [response], calls, denied, True,
            None if denied else "depth_limit_allowed",
            depth=MetricValue.observed(2) if mode != "off" else MetricValue.unavailable(),
        )
    finally:
        revoke_nesting(raw_session)


def _observe_parent(raw_session, agent_id):
    _send(
        [{"role": "user", "content": "Record direct work; do not delegate."}],
        [RECORD_DECISION_TOOL], raw_session,
        agent_id=agent_id,
    )


def _send(messages, tools, raw_session, *, agent_id=None, parent_agent_id=None):
    return send_message(
        CONFIG, messages, tools, raw_session,
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
    )


def _from_responses(
    responses,
    calls,
    passed,
    complete,
    failure_code,
    *,
    depth=MetricValue.unavailable(),
    discovery=0,
):
    return build_trial_result(
        responses=responses,
        calls=calls,
        passed=passed,
        complete=complete,
        failure_code=failure_code,
        depth=depth,
        discovery=discovery,
    )


def _result(passed, complete, *, failure_code):
    return TrialResult(
        passed=passed,
        complete=complete,
        agent_calls=0,
        send_message_calls=0,
        maximum_observed_depth=MetricValue.unavailable(),
        maximum_discovery_streak=0,
        duplicate_launches=0,
        input_tokens=MetricValue.unavailable(),
        output_tokens=MetricValue.unavailable(),
        failure_code=failure_code,
    )


def _reviewer_reused(calls):
    if len(calls) != 1 or calls[0].get("name") != "SendMessage":
        return False
    tool_input = calls[0].get("input")
    return isinstance(tool_input, dict) and tool_input.get("to") == "review-1"


def _requires_user_input(calls):
    return _closed_signal(
        calls,
        "RequestUserInput",
        {"requires_user_input": True},
    )


def _completion_unavailable(calls):
    return _closed_signal(
        calls,
        "RecordCompletionState",
        {"completion_state": "unavailable"},
    )


def _closed_signal(calls, name, expected_input):
    return (
        len(calls) == 1
        and calls[0].get("name") == name
        and calls[0].get("input") == expected_input
    )
