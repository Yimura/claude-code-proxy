import pytest

from claude_code_proxy.config import CodexOrchestrationMode
from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionRequest,
    Message,
    TextBlock,
    ToolDefinition,
)
from claude_code_proxy.providers.codex.orchestration import (
    AGENT_COMPLETION_POLICY,
    AGENT_GUIDANCE,
    AGENT_SCOPE_POLICY,
    AUTHORIZED_NESTING_POLICY,
    DISCOVERY_POLICY,
    ENFORCED_SUBAGENT_POLICY,
    ORCHESTRATION_RATIONALE,
    SEND_MESSAGE_GUIDANCE,
    SUBAGENT_AGENT_GUIDANCE,
    SUBAGENT_SCOPE_POLICY,
    TASK_OUTPUT_GUIDANCE,
    reconcile_codex_orchestration,
    reconcile_codex_request,
)
from claude_code_proxy.reasoning import ReasoningPolicy


def _agent(schema=None):
    return ToolDefinition(
        "Agent",
        "Launch a worker.",
        schema
        or {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
        },
    )


def _send_message():
    return ToolDefinition(
        "SendMessage",
        "Message a worker.",
        {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "message": {"type": "string"},
            },
        },
    )


def _request(*, agent_id=None):
    return CompletionRequest(
        original_model="claude-opus-5",
        model="openai/gpt-5.6-sol",
        response_model="claude-opus-5",
        max_tokens=100,
        messages=(Message("user", (TextBlock("work"),)),),
        reasoning=ReasoningPolicy(None, None),
        client_identity=ClientIdentity(
            session_id="session",
            agent_id=agent_id,
            parent_agent_id="parent" if agent_id else None,
        ),
        system=(TextBlock("base system"),),
        tools=(
            _agent(),
            _send_message(),
            ToolDefinition("TaskOutput", "Retrieve output."),
        ),
    )


def test_root_agent_adds_stable_system_and_tool_policies():
    request = _request()

    reconciled = reconcile_codex_request(request)

    assert reconciled.system == (
        request.system[0],
        TextBlock(AGENT_COMPLETION_POLICY),
        TextBlock(AGENT_SCOPE_POLICY),
        TextBlock(DISCOVERY_POLICY),
        TextBlock(ORCHESTRATION_RATIONALE),
    )
    assert reconciled.tools[0].description == (
        f"Launch a worker.\n\n{AGENT_GUIDANCE}"
    )
    assert reconciled.tools[1].description == (
        f"Message a worker.\n\n{SEND_MESSAGE_GUIDANCE}"
    )
    assert reconciled.tools[2].description == (
        f"Retrieve output.\n\n{TASK_OUTPUT_GUIDANCE}"
    )


def test_subagent_adds_scope_policy_and_agent_warning():
    request = _request(agent_id="worker-1")

    reconciled = reconcile_codex_request(request)

    assert reconciled.system[-1] == TextBlock(SUBAGENT_SCOPE_POLICY)
    assert reconciled.tools[0].description.endswith(
        f"{AGENT_GUIDANCE}\n\n{SUBAGENT_AGENT_GUIDANCE}"
    )


def test_agent_schema_identity_is_preserved():
    schema = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
    }
    agent = _agent(schema)

    _, tools = reconcile_codex_orchestration((), (agent,))

    assert tools[0].input_schema is schema


def test_task_output_guidance_remains_without_agent():
    task_output = ToolDefinition("TaskOutput", "Retrieve output.")

    system, tools = reconcile_codex_orchestration((), (task_output,))

    assert system == ()
    assert tools[0].description == (
        f"Retrieve output.\n\n{TASK_OUTPUT_GUIDANCE}"
    )


def test_empty_descriptions_receive_guidance_without_leading_separator():
    _, tools = reconcile_codex_orchestration(
        (),
        (
            ToolDefinition("Agent"),
            ToolDefinition("SendMessage"),
            ToolDefinition("TaskOutput"),
        ),
    )

    assert tools[0].description == AGENT_GUIDANCE
    assert tools[1].description == SEND_MESSAGE_GUIDANCE
    assert tools[2].description == TASK_OUTPUT_GUIDANCE


def test_unrelated_tools_and_system_remain_unchanged():
    system = (TextBlock("base system"),)
    lookup = ToolDefinition("lookup", "Lookup data.", {"type": "object"})
    tools = (lookup,)

    reconciled_system, reconciled_tools = reconcile_codex_orchestration(
        system, tools
    )

    assert reconciled_system is system
    assert reconciled_tools is tools
    assert reconciled_tools[0] is lookup


@pytest.mark.parametrize("agent_id", [None, "worker-1"])
def test_reconciliation_is_idempotent(agent_id):
    request = _request(agent_id=agent_id)

    first = reconcile_codex_request(request)
    second = reconcile_codex_request(first)

    assert second is first


def test_blank_agent_id_uses_root_policy():
    request = _request(agent_id="   ")

    reconciled = reconcile_codex_request(request)

    assert TextBlock(SUBAGENT_SCOPE_POLICY) not in reconciled.system
    assert SUBAGENT_AGENT_GUIDANCE not in reconciled.tools[0].description


def test_send_message_remains_unchanged_without_agent():
    send_message = _send_message()

    system, tools = reconcile_codex_orchestration((), (send_message,))

    assert system == ()
    assert tools == (send_message,)
    assert tools[0] is send_message


def test_off_returns_original_objects_without_guidance():
    request = _request(agent_id="worker-1")

    reconciled = reconcile_codex_request(
        request,
        mode=CodexOrchestrationMode.OFF,
        agent_allowed=False,
        authorization_present=False,
    )

    assert reconciled is request
    assert reconciled.system is request.system
    assert reconciled.tools is request.tools


def test_advisory_preserves_agent_and_adds_fixed_rationale():
    request = _request(agent_id="worker-1")

    reconciled = reconcile_codex_request(
        request,
        mode=CodexOrchestrationMode.ADVISORY,
        agent_allowed=False,
        authorization_present=False,
    )

    assert reconciled.tools[0].name == "Agent"
    assert reconciled.system.count(TextBlock(ORCHESTRATION_RATIONALE)) == 1
    assert "narrow" in ORCHESTRATION_RATIONALE
    assert "recursive" in ORCHESTRATION_RATIONALE
    assert "duplicate" in ORCHESTRATION_RATIONALE
    assert "discovery" in ORCHESTRATION_RATIONALE
    assert "token" in ORCHESTRATION_RATIONALE


def test_enforce_removes_only_agent_for_unauthorized_subagent():
    request = _request(agent_id="worker-1")

    reconciled = reconcile_codex_request(
        request,
        mode=CodexOrchestrationMode.ENFORCE,
        agent_allowed=False,
        authorization_present=False,
    )

    assert tuple(tool.name for tool in reconciled.tools) == (
        "SendMessage",
        "TaskOutput",
    )
    assert reconciled.tools[0].input_schema is request.tools[1].input_schema
    assert reconciled.tools[1].input_schema is request.tools[2].input_schema
    assert TextBlock(ENFORCED_SUBAGENT_POLICY) in reconciled.system


def test_enforce_retains_agent_for_root_and_authorized_nesting():
    root = reconcile_codex_request(
        _request(),
        mode=CodexOrchestrationMode.ENFORCE,
        agent_allowed=True,
        authorization_present=False,
    )
    nested = reconcile_codex_request(
        _request(agent_id="worker-1"),
        mode=CodexOrchestrationMode.ENFORCE,
        agent_allowed=True,
        authorization_present=True,
    )

    assert any(tool.name == "Agent" for tool in root.tools)
    assert any(tool.name == "Agent" for tool in nested.tools)
    assert TextBlock(AUTHORIZED_NESTING_POLICY) in nested.system


def test_enforced_reconciliation_is_idempotent():
    request = _request(agent_id="worker-1")
    kwargs = {
        "mode": CodexOrchestrationMode.ENFORCE,
        "agent_allowed": False,
        "authorization_present": False,
    }

    first = reconcile_codex_request(request, **kwargs)
    second = reconcile_codex_request(first, **kwargs)

    assert second is first
