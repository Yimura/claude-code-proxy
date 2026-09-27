from claude_code_proxy.providers.codex.orchestration_registry import (
    OrchestrationRegistry,
    ParentState,
)


class Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value

    def advance(self) -> None:
        self.value += 1.0


def test_no_agent_has_no_lineage() -> None:
    registry = OrchestrationRegistry()

    assert registry.observe_lineage("session", None, None) is None
    assert registry.lineage("session", None) is None


def test_agent_without_parent_is_depth_one_and_absent() -> None:
    clock = Clock()
    registry = OrchestrationRegistry(monotonic_clock=clock)

    lineage = registry.observe_lineage("session", "agent", None)

    assert lineage is not None
    assert lineage.session_id == "session"
    assert lineage.agent_id == "agent"
    assert lineage.parent_agent_id is None
    assert lineage.parent_state is ParentState.ABSENT
    assert lineage.depth == 1
    assert lineage.first_observed_at == 10.0
    assert lineage.last_observed_at == 10.0


def test_unknown_parent_resolves_when_parent_appears() -> None:
    clock = Clock()
    registry = OrchestrationRegistry(monotonic_clock=clock)
    child = registry.observe_lineage("session", "child", "parent")
    assert child is not None
    assert child.parent_state is ParentState.UNKNOWN
    assert child.depth is None

    clock.advance()
    registry.observe_lineage("session", "parent", None)
    resolved = registry.lineage("session", "child")

    assert resolved is not None
    assert resolved.parent_state is ParentState.OBSERVED
    assert resolved.depth == 2
    assert resolved.parent_agent_id == "parent"
    assert resolved.first_observed_at == 10.0
    assert resolved.last_observed_at == 11.0


def test_conflicting_parent_marks_ambiguous_without_overwrite() -> None:
    registry = OrchestrationRegistry()
    registry.observe_lineage("session", "first", None)
    registry.observe_lineage("session", "second", None)
    registry.observe_lineage("session", "child", "first")

    lineage = registry.observe_lineage("session", "child", "second")

    assert lineage is not None
    assert lineage.parent_state is ParentState.AMBIGUOUS
    assert lineage.parent_agent_id == "first"
    assert lineage.depth is None


def test_self_and_multi_agent_cycles_are_closed_states() -> None:
    registry = OrchestrationRegistry()

    self_cycle = registry.observe_lineage("session", "self", "self")
    registry.observe_lineage("session", "a", "b")
    registry.observe_lineage("session", "b", "a")

    assert self_cycle is not None
    assert self_cycle.parent_state is ParentState.CYCLIC
    assert self_cycle.depth is None
    assert registry.lineage("session", "a").parent_state is ParentState.CYCLIC
    assert registry.lineage("session", "b").parent_state is ParentState.CYCLIC


def test_sessions_scope_equal_agent_ids_and_eviction_removes_only_one_session() -> None:
    registry = OrchestrationRegistry()
    registry.observe_lineage("one", "agent", None)
    registry.observe_lineage("two", "agent", None)

    registry.remove_session("one")

    assert registry.lineage("one", "agent") is None
    assert registry.lineage("two", "agent") is not None
