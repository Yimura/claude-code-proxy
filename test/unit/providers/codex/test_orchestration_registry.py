import pytest

from claude_code_proxy.providers.codex.orchestration_registry import (
    AuthorizationStatus,
    MAX_LINEAGE_RECORDS_PER_SESSION,
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


def test_conditional_session_removal_requires_unchanged_generation() -> None:
    registry = OrchestrationRegistry()
    registry.observe_lineage("session", "old", None)
    generation = registry.session_generation("session")
    registry.observe_lineage("session", "new", None)

    assert registry.remove_session_if_generation("session", generation) is False
    assert registry.lineage("session", "new") is not None
    current = registry.session_generation("session")
    assert registry.remove_session_if_generation("session", current) is True
    assert registry.lineage("session", "new") is None


def test_reverse_order_deep_chain_resolves_iteratively() -> None:
    registry = OrchestrationRegistry()
    count = min(900, MAX_LINEAGE_RECORDS_PER_SESSION)

    for index in reversed(range(count)):
        parent = None if index == 0 else f"agent-{index - 1}"
        registry.observe_lineage("session", f"agent-{index}", parent)

    deepest = registry.lineage("session", f"agent-{count - 1}")
    assert deepest is not None
    assert deepest.parent_state is ParentState.OBSERVED
    assert deepest.depth == count


def test_lineage_cardinality_limit_fails_closed_without_retention() -> None:
    registry = OrchestrationRegistry()
    for index in range(MAX_LINEAGE_RECORDS_PER_SESSION):
        registry.observe_lineage("session", f"agent-{index}", None)

    overflow = registry.observe_lineage("session", "overflow-agent", None)

    assert overflow is not None
    assert overflow.parent_state is ParentState.UNKNOWN
    assert overflow.depth is None
    assert registry.lineage("session", "overflow-agent") is None
    assert registry.lineage("session", "agent-0") is not None


def test_authorization_before_lineage_is_supported_and_expires_atomically() -> None:
    clock = Clock()
    registry = OrchestrationRegistry(monotonic_clock=clock)

    authorization = registry.authorize("session", max_depth=4, duration=2.0)

    assert authorization.session_id == "session"
    assert authorization.max_depth == 4
    assert registry.authorization("session").status is AuthorizationStatus.ACTIVE
    clock.value = 12.0
    expired = registry.authorization("session")
    assert expired.status is AuthorizationStatus.EXPIRED
    assert expired.max_depth == 4
    assert registry.authorization("session").status is AuthorizationStatus.EXPIRED


@pytest.mark.parametrize("max_depth", [True, 1, 9, 2.0])
def test_authorization_rejects_invalid_exact_depth(max_depth) -> None:
    with pytest.raises(ValueError, match="max_depth"):
        OrchestrationRegistry().authorize(
            "session", max_depth=max_depth, duration=60
        )


@pytest.mark.parametrize("duration", [True, 0, 86401, float("inf")])
def test_authorization_rejects_invalid_duration(duration) -> None:
    with pytest.raises(ValueError, match="duration"):
        OrchestrationRegistry().authorize(
            "session", max_depth=2, duration=duration
        )


def test_revoke_is_idempotent() -> None:
    registry = OrchestrationRegistry()
    registry.authorize("session", max_depth=2, duration=60)

    assert registry.revoke("session") is True
    assert registry.revoke("session") is False
    assert registry.authorization("session").status is AuthorizationStatus.ABSENT


def test_raw_authorization_is_immediately_keyed_by_public_identity() -> None:
    from claude_code_proxy.public_identity import PublicIdentity

    identity = PublicIdentity(secret=b"control-secret")
    registry = OrchestrationRegistry(identity=identity)
    raw_session = "raw-sensitive-session"

    authorization = registry.authorize_raw_session(
        raw_session, max_depth=3, duration_seconds=60
    )

    public_id = identity.public_id(raw_session)
    assert authorization.session_id == public_id
    assert raw_session not in repr(registry)
    assert registry.authorization(public_id).status is AuthorizationStatus.ACTIVE


def test_authorization_rows_include_only_live_remaining_duration() -> None:
    clock = Clock()
    registry = OrchestrationRegistry(monotonic_clock=clock)
    registry.authorize("second", max_depth=4, duration=5)
    registry.authorize("first", max_depth=2, duration=2)
    clock.value = 12.0

    rows = registry.authorizations()

    assert [(row.session_id, row.max_depth, row.remaining_seconds) for row in rows] == [
        ("second", 4, 3.0)
    ]
    assert registry.authorization("first").status is AuthorizationStatus.EXPIRED


def test_raw_revocation_is_idempotent() -> None:
    from claude_code_proxy.public_identity import PublicIdentity

    identity = PublicIdentity(secret=b"control-secret")
    registry = OrchestrationRegistry(identity=identity)
    registry.authorize_raw_session("raw", max_depth=2, duration_seconds=60)

    assert registry.revoke_raw_session("raw") is True
    assert registry.revoke_raw_session("raw") is False


def test_simultaneous_conflicting_parent_freezes_first_observation() -> None:
    from concurrent.futures import ThreadPoolExecutor
    import threading

    registry = OrchestrationRegistry()
    registry.observe_lineage("session", "one", None)
    registry.observe_lineage("session", "two", None)
    gate = threading.Barrier(3)

    def observe(parent: str):
        gate.wait()
        return registry.observe_lineage("session", "child", parent)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(observe, parent) for parent in ("one", "two")]
        gate.wait()
        [future.result(timeout=5) for future in futures]

    lineage = registry.lineage("session", "child")
    assert lineage is not None
    assert lineage.parent_agent_id in {"one", "two"}
    assert lineage.parent_state is ParentState.AMBIGUOUS


def test_concurrent_authorize_revoke_and_reads_never_expose_partial_state() -> None:
    from concurrent.futures import ThreadPoolExecutor
    import threading

    clock = Clock()
    registry = OrchestrationRegistry(monotonic_clock=clock)
    gate = threading.Barrier(4)
    observed = []

    def authorize():
        gate.wait()
        return registry.authorize("session", max_depth=8, duration=60)

    def revoke():
        gate.wait()
        return registry.revoke("session")

    def read():
        gate.wait()
        for _ in range(100):
            observed.append(registry.authorization("session"))
            registry.authorizations()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(authorize), executor.submit(revoke), executor.submit(read)]
        gate.wait()
        [future.result(timeout=5) for future in futures]

    for authorization in observed:
        if authorization.status is AuthorizationStatus.ABSENT:
            assert authorization.max_depth is None
            assert authorization.authorized_at is None
            assert authorization.expires_at is None
        else:
            assert authorization.max_depth == 8
            assert authorization.authorized_at == 10.0
            assert authorization.expires_at == 70.0


def test_concurrent_expiry_reads_produce_closed_complete_snapshots() -> None:
    from concurrent.futures import ThreadPoolExecutor
    import threading

    clock = Clock()
    registry = OrchestrationRegistry(monotonic_clock=clock)
    registry.authorize("session", max_depth=3, duration=1)
    clock.value = 11.0
    gate = threading.Barrier(9)

    def read():
        gate.wait()
        return registry.authorization("session")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read) for _ in range(8)]
        gate.wait()
        results = [future.result(timeout=5) for future in futures]

    assert all(result.status is AuthorizationStatus.EXPIRED for result in results)
    assert all(result.max_depth == 3 for result in results)
    assert registry.authorizations() == ()
