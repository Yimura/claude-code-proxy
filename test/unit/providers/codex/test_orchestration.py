from claude_code_proxy.providers.codex.orchestration import (
    AGENT_COMPLETION_POLICY,
    AGENT_GUIDANCE,
    TASK_OUTPUT_GUIDANCE,
    reconcile_codex_orchestration,
)
from claude_code_proxy.domain.models import TextBlock, ToolDefinition


def test_agent_adds_system_policy_and_preserves_schema():
    schema = {"type": "object", "properties": {"prompt": {"type": "string"}}}
    system = (TextBlock("base system"),)
    agent = ToolDefinition("Agent", "Launch a worker.", schema)

    reconciled_system, reconciled_tools = reconcile_codex_orchestration(
        system, (agent,)
    )

    assert reconciled_system == (
        system[0],
        TextBlock(AGENT_COMPLETION_POLICY),
    )
    assert reconciled_tools[0].description == f"Launch a worker.\n\n{AGENT_GUIDANCE}"
    assert reconciled_tools[0].input_schema is schema


def test_task_output_keeps_explicit_non_agent_background_use():
    schema = {"type": "object", "properties": {"task_id": {"type": "string"}}}
    task_output = ToolDefinition("TaskOutput", "Retrieve task output.", schema)

    system, tools = reconcile_codex_orchestration((), (task_output,))

    assert system == ()
    assert tools[0].description == f"Retrieve task output.\n\n{TASK_OUTPUT_GUIDANCE}"
    assert "non-Agent background tasks" in tools[0].description
    assert tools[0].input_schema is schema


def test_empty_descriptions_receive_guidance_without_leading_separator():
    _, tools = reconcile_codex_orchestration(
        (),
        (ToolDefinition("Agent"), ToolDefinition("TaskOutput")),
    )

    assert tools[0].description == AGENT_GUIDANCE
    assert tools[1].description == TASK_OUTPUT_GUIDANCE


def test_unrelated_tools_and_system_are_returned_unchanged():
    system = (TextBlock("base system"),)
    lookup = ToolDefinition("lookup", "Lookup data.", {"type": "object"})
    tools = (lookup,)

    reconciled_system, reconciled_tools = reconcile_codex_orchestration(
        system, tools
    )

    assert reconciled_system is system
    assert reconciled_tools is tools
    assert reconciled_tools[0] is lookup


def test_reconciliation_is_idempotent():
    system = (TextBlock("base system"),)
    tools = (
        ToolDefinition("Agent", "Launch a worker."),
        ToolDefinition("TaskOutput", "Retrieve output."),
    )

    first_system, first_tools = reconcile_codex_orchestration(system, tools)
    second_system, second_tools = reconcile_codex_orchestration(
        first_system, first_tools
    )

    assert second_system is first_system
    assert second_tools is first_tools
    assert sum(block.text == AGENT_COMPLETION_POLICY for block in second_system) == 1
    assert second_tools[0].description.count(AGENT_GUIDANCE) == 1
    assert second_tools[1].description.count(TASK_OUTPUT_GUIDANCE) == 1
