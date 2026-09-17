"""Codex-specific guidance for Claude Code Agent orchestration."""

from dataclasses import replace

from ...domain.models import CompletionRequest, TextBlock, ToolDefinition

AGENT_COMPLETION_POLICY = (
    "Agent completion policy for this Codex-backed session:\n"
    "- Agent tasks complete through one automatic parent notification.\n"
    "- Never use TaskOutput for Agent tasks or poll Agent completion.\n"
    "- Continue independent work after dispatch; if none remains, end the turn "
    "and wait.\n"
    "- TaskOutput remains available only for non-Agent background tasks without "
    "automatic completion."
)
AGENT_GUIDANCE = (
    "Agent completion is push-based: one completion notification arrives "
    "automatically. Never call TaskOutput for Agent tasks. Continue independent "
    "work after dispatch; if none remains, end the turn and wait."
)
TASK_OUTPUT_GUIDANCE = (
    "Never use TaskOutput for Agent tasks or poll Agent completion. Agent results "
    "arrive automatically exactly once. Use TaskOutput only for non-Agent "
    "background tasks that require explicit retrieval and have no automatic "
    "completion notification."
)
_TOOL_GUIDANCE = {
    "Agent": AGENT_GUIDANCE,
    "TaskOutput": TASK_OUTPUT_GUIDANCE,
}


def reconcile_codex_orchestration(
    system: tuple[TextBlock, ...],
    tools: tuple[ToolDefinition, ...],
) -> tuple[tuple[TextBlock, ...], tuple[ToolDefinition, ...]]:
    """Add Codex Agent guidance without changing tool capabilities."""
    reconciled_tools = tuple(_reconcile_tool(tool) for tool in tools)
    if reconciled_tools == tools:
        reconciled_tools = tools

    if not any(tool.name == "Agent" for tool in tools):
        return system, reconciled_tools
    if any(block.text == AGENT_COMPLETION_POLICY for block in system):
        return system, reconciled_tools
    return (*system, TextBlock(AGENT_COMPLETION_POLICY)), reconciled_tools


def reconcile_codex_request(request: CompletionRequest) -> CompletionRequest:
    """Return request with Codex-specific Agent guidance applied once."""
    system, tools = reconcile_codex_orchestration(request.system, request.tools)
    if system is request.system and tools is request.tools:
        return request
    return replace(request, system=system, tools=tools)


def _reconcile_tool(tool: ToolDefinition) -> ToolDefinition:
    guidance = _TOOL_GUIDANCE.get(tool.name)
    if guidance is None or guidance in tool.description:
        return tool
    separator = "\n\n" if tool.description else ""
    return replace(tool, description=f"{tool.description}{separator}{guidance}")
