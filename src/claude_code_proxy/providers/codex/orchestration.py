"""Codex-specific guidance for Claude Code Agent orchestration."""

from dataclasses import replace

from ...config import CodexOrchestrationMode
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
AGENT_SCOPE_POLICY = (
    "Agent orchestration policy for this Codex-backed session:\n"
    "- Prefer a few agents with broad, coherent ownership over many narrow "
    "agents.\n"
    "- Group work sharing files, subsystem state, or reasoning under one owner.\n"
    "- Let one owner handle related implementation, tests, and resulting fixes.\n"
    "- Do not create separate agents for checklist steps or overlapping review "
    "dimensions.\n"
    "- Review after a meaningful batch, then reuse the same reviewer for "
    "rechecks.\n"
    "- Reuse existing workers with SendMessage instead of creating replacements.\n"
    "- Load every applicable skill, but treat skill procedures as required work "
    "rather than automatic agent boundaries.\n"
    "- Use the least capable model that can complete delegated work reliably."
)
DISCOVERY_POLICY = (
    "Discovery policy for this Codex-backed session:\n"
    "- Use tools to settle named decisions, not to investigate for completeness.\n"
    "- Discovery is sufficient when current behavior, change location, "
    "constraints, and verification path are known.\n"
    "- Read required local code before editing it and search for existing "
    "precedent before adding named behavior.\n"
    "- Search locally before using the web. Use external search only for current "
    "facts the repository cannot answer.\n"
    "- Batch independent read-only work. Do not reread unchanged files or repeat "
    "equivalent searches.\n"
    "- If two focused search approaches fail, report their bounds and escalate.\n"
    "- If uncertainty is a preference, priority, scope, or product decision, ask "
    "the user early.\n"
    "- After eight consecutive discovery-only tool turns, explicitly choose to "
    "act, ask, stop, or continue for one named unresolved decision.\n"
    "- Never continue through thirty consecutive discovery-only tool turns "
    "without changing approach or receiving explicit authorization."
)
SUBAGENT_SCOPE_POLICY = (
    "Subagent scope policy for this Codex-backed session:\n"
    "- Work directly within the coherent scope assigned by the parent.\n"
    "- Do not invoke Agent, Workflow, or agent-spawning skills unless the parent "
    "explicitly authorized nested delegation.\n"
    "- If required context is missing, return NEEDS_CONTEXT with the exact "
    "missing facts.\n"
    "- If the task expands materially or cannot be completed safely, return "
    "BLOCKED with evidence.\n"
    "- Do not broaden a focused task into an audit.\n"
    "- Return conclusions with paths, lines, commands, and bounded negative "
    "results instead of raw file dumps."
)
AGENT_GUIDANCE = (
    "Agent completion is push-based: one completion notification arrives "
    "automatically. Never call TaskOutput for Agent tasks. Delegate coherent "
    "problem domains rather than individual steps, avoid overlapping workers, "
    "and provide known paths, symbols, constraints, commands, and expected "
    "evidence in the worker prompt."
)
SUBAGENT_AGENT_GUIDANCE = (
    "This request belongs to a subagent. Do not launch another Agent unless the "
    "parent explicitly authorized nested delegation."
)
ORCHESTRATION_RATIONALE = (
    "Codex orchestration rationale:\n"
    "Prior sessions used excessive narrow forks, recursive delegation, duplicate "
    "review workers, and repeated discovery, causing avoidable token use. Prefer "
    "a few broad owners, direct work for focused tasks, reuse of existing workers "
    "for fixes and rechecks, batched local discovery, and one review after a "
    "meaningful implementation batch."
)
ENFORCED_SUBAGENT_POLICY = (
    "Enforced subagent policy:\n"
    "This nested request is not authorized to delegate further, so Agent is not "
    "available. Complete the assigned scope directly, reuse an existing worker "
    "through SendMessage when applicable, or return the exact missing context to "
    "the parent."
)
AUTHORIZED_NESTING_POLICY = (
    "Authorized nesting policy:\n"
    "Nested delegation is explicitly authorized only within its expiring, "
    "depth-bounded grant. Keep ownership broad, avoid recursive fan-out, and reuse "
    "workers instead of creating duplicate implementers or reviewers."
)
SEND_MESSAGE_GUIDANCE = (
    "Reuse the original implementer for related fixes and the original reviewer "
    "for rechecks. Send missing context to an existing worker instead of creating "
    "a replacement."
)
TASK_OUTPUT_GUIDANCE = (
    "Never use TaskOutput for Agent tasks or poll Agent completion. Agent results "
    "arrive automatically exactly once. Use TaskOutput only for non-Agent "
    "background tasks that require explicit retrieval and have no automatic "
    "completion notification."
)
_ROOT_POLICIES = (
    AGENT_COMPLETION_POLICY,
    AGENT_SCOPE_POLICY,
    DISCOVERY_POLICY,
)


def reconcile_codex_orchestration(
    system: tuple[TextBlock, ...],
    tools: tuple[ToolDefinition, ...],
    *,
    mode: CodexOrchestrationMode = CodexOrchestrationMode.ADVISORY,
    is_subagent: bool = False,
    agent_allowed: bool = True,
    authorization_present: bool = False,
) -> tuple[tuple[TextBlock, ...], tuple[ToolDefinition, ...]]:
    """Apply stable Codex guidance and optionally remove only Agent."""
    if mode is CodexOrchestrationMode.OFF:
        return system, tools

    has_agent = any(tool.name == "Agent" for tool in tools)
    filtered_tools = _filter_agent_tool(
        tools,
        mode=mode,
        is_subagent=is_subagent,
        agent_allowed=agent_allowed,
    )
    reconciled_tools = tuple(
        _reconcile_tool(
            tool,
            is_subagent=is_subagent,
            has_agent=has_agent,
        )
        for tool in filtered_tools
    )
    if reconciled_tools == tools:
        reconciled_tools = tools
    elif reconciled_tools == filtered_tools:
        reconciled_tools = filtered_tools

    if not has_agent:
        return system, reconciled_tools

    policies = _orchestration_policies(
        mode=mode,
        is_subagent=is_subagent,
        agent_allowed=agent_allowed,
        authorization_present=authorization_present,
    )
    return _append_policies(system, policies), reconciled_tools


def reconcile_codex_request(
    request: CompletionRequest,
    *,
    mode: CodexOrchestrationMode = CodexOrchestrationMode.ADVISORY,
    agent_allowed: bool = True,
    authorization_present: bool = False,
) -> CompletionRequest:
    """Return a request with one role-aware orchestration reconciliation."""
    agent_id = (request.client_identity.agent_id or "").strip()
    system, tools = reconcile_codex_orchestration(
        request.system,
        request.tools,
        mode=mode,
        is_subagent=bool(agent_id),
        agent_allowed=agent_allowed,
        authorization_present=authorization_present,
    )
    if system is request.system and tools is request.tools:
        return request
    return replace(request, system=system, tools=tools)


def _filter_agent_tool(
    tools: tuple[ToolDefinition, ...],
    *,
    mode: CodexOrchestrationMode,
    is_subagent: bool,
    agent_allowed: bool,
) -> tuple[ToolDefinition, ...]:
    if mode is not CodexOrchestrationMode.ENFORCE:
        return tools
    if not is_subagent or agent_allowed:
        return tools
    return tuple(tool for tool in tools if tool.name != "Agent")


def _orchestration_policies(
    *,
    mode: CodexOrchestrationMode,
    is_subagent: bool,
    agent_allowed: bool,
    authorization_present: bool,
) -> tuple[str, ...]:
    policies = (*_ROOT_POLICIES, ORCHESTRATION_RATIONALE)
    if not is_subagent:
        return policies
    policies = (*policies, SUBAGENT_SCOPE_POLICY)
    if mode is not CodexOrchestrationMode.ENFORCE:
        return policies
    if agent_allowed and authorization_present:
        return (*policies, AUTHORIZED_NESTING_POLICY)
    if not agent_allowed:
        return (*policies, ENFORCED_SUBAGENT_POLICY)
    return policies


def _append_policies(
    system: tuple[TextBlock, ...], policies: tuple[str, ...]
) -> tuple[TextBlock, ...]:
    existing = {block.text for block in system}
    missing = tuple(
        TextBlock(policy) for policy in policies if policy not in existing
    )
    return system if not missing else (*system, *missing)


def _reconcile_tool(
    tool: ToolDefinition,
    *,
    is_subagent: bool,
    has_agent: bool,
) -> ToolDefinition:
    description = tool.description
    for guidance in _tool_guidance(
        tool.name,
        is_subagent=is_subagent,
        has_agent=has_agent,
    ):
        if guidance in description:
            continue
        separator = "\n\n" if description else ""
        description = f"{description}{separator}{guidance}"
    if description == tool.description:
        return tool
    return replace(tool, description=description)


def _tool_guidance(
    name: str, *, is_subagent: bool, has_agent: bool
) -> tuple[str, ...]:
    if name == "Agent":
        if is_subagent:
            return AGENT_GUIDANCE, SUBAGENT_AGENT_GUIDANCE
        return (AGENT_GUIDANCE,)
    if name == "SendMessage" and has_agent:
        return (SEND_MESSAGE_GUIDANCE,)
    if name == "TaskOutput":
        return (TASK_OUTPUT_GUIDANCE,)
    return ()
