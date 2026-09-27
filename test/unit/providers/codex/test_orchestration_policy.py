from dataclasses import replace

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
    OrchestrationRegistry,
)
from claude_code_proxy.reasoning import ReasoningPolicy


class Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


def request(agent_id=None, parent_id=None) -> CompletionRequest:
    return CompletionRequest(
        original_model="claude",
        model="openai/gpt",
        response_model="claude",
        max_tokens=100,
        messages=(Message("user", (TextBlock("work"),)),),
        reasoning=ReasoningPolicy(None, None),
        client_identity=ClientIdentity("session", agent_id, parent_id),
        tools=(ToolDefinition("Agent", "Delegate."), ToolDefinition("Read")),
    )


def coordinator(mode, clock=None):
    identity = PublicIdentity(secret=b"secret")
    registry = OrchestrationRegistry(monotonic_clock=clock)
    return OrchestrationPolicyCoordinator(mode, identity, registry), registry, identity


def public_ids(identity, agent, parent=None):
    session = identity.public_id("session")
    public_agent = identity.public_agent_id("session", agent)
    public_parent = (
        identity.public_agent_id("session", parent) if parent else None
    )
    return session, public_agent, public_parent


def has_agent(result) -> bool:
    return any(tool.name == "Agent" for tool in result.request.tools)


def test_off_returns_original_request_without_registry_mutation() -> None:
    policy, registry, identity = coordinator(CodexOrchestrationMode.OFF)
    original = request("child", "parent")

    result = policy.reconcile(original)
    session, agent, _ = public_ids(identity, "child", "parent")

    assert result.request is original
    assert result.decision.code is OrchestrationDecisionCode.NOT_APPLICABLE
    assert registry.lineage(session, agent) is None


def test_advisory_observes_lineage_and_preserves_agent() -> None:
    policy, registry, identity = coordinator(CodexOrchestrationMode.ADVISORY)

    result = policy.reconcile(request("worker"))
    session, agent, _ = public_ids(identity, "worker")

    assert result.decision.code is OrchestrationDecisionCode.ADVISORY
    assert result.decision.depth.status == "observed"
    assert result.decision.depth.value == 1
    assert has_agent(result)
    assert registry.lineage(session, agent) is not None


def test_enforce_root_always_retains_agent() -> None:
    policy, _, _ = coordinator(CodexOrchestrationMode.ENFORCE)

    result = policy.reconcile(request())

    assert result.decision.code is OrchestrationDecisionCode.ROOT_ALLOWED
    assert result.decision.depth.status == "not_applicable"
    assert has_agent(result)


def test_unknown_ambiguous_and_cyclic_lineage_deny_regardless_of_authorization() -> None:
    policy, registry, identity = coordinator(CodexOrchestrationMode.ENFORCE)
    session = identity.public_id("session")
    registry.authorize(session, max_depth=8, duration=60)

    unknown = policy.reconcile(request("child", "missing"))
    ambiguous = policy.reconcile(request("child", "other"))
    cyclic = policy.reconcile(request("cycle", "cycle"))

    assert unknown.decision.code is OrchestrationDecisionCode.LINEAGE_UNKNOWN
    assert ambiguous.decision.code is OrchestrationDecisionCode.LINEAGE_AMBIGUOUS
    assert cyclic.decision.code is OrchestrationDecisionCode.LINEAGE_CYCLIC
    assert not has_agent(unknown)
    assert not has_agent(ambiguous)
    assert not has_agent(cyclic)


def test_same_request_shape_rechecks_authorize_revoke_and_expiry() -> None:
    clock = Clock()
    policy, registry, identity = coordinator(
        CodexOrchestrationMode.ENFORCE, clock
    )
    session, _, _ = public_ids(identity, "worker")
    nested = request("worker")

    denied = policy.reconcile(nested)
    registry.authorize(session, max_depth=2, duration=2)
    allowed = policy.reconcile(nested)
    registry.revoke(session)
    revoked = policy.reconcile(nested)
    registry.authorize(session, max_depth=2, duration=1)
    clock.value = 11.0
    expired = policy.reconcile(nested)

    assert denied.decision.code is OrchestrationDecisionCode.NESTED_DENIED
    assert allowed.decision.code is OrchestrationDecisionCode.NESTED_ALLOWED
    assert revoked.decision.code is OrchestrationDecisionCode.NESTED_DENIED
    assert expired.decision.code is OrchestrationDecisionCode.AUTHORIZATION_EXPIRED
    assert not has_agent(denied)
    assert has_agent(allowed)
    assert not has_agent(revoked)
    assert not has_agent(expired)


def test_depth_limit_is_checked_on_every_reconciliation() -> None:
    policy, registry, identity = coordinator(CodexOrchestrationMode.ENFORCE)
    session, _, _ = public_ids(identity, "parent")
    registry.authorize(session, max_depth=2, duration=60)
    policy.reconcile(request("parent"))

    result = policy.reconcile(request("child", "parent"))

    assert result.decision.code is OrchestrationDecisionCode.DEPTH_LIMIT_REACHED
    assert result.decision.depth.value == 2
    assert not has_agent(result)


def test_listing_before_expired_request_preserves_expired_decision() -> None:
    clock = Clock()
    policy, registry, identity = coordinator(
        CodexOrchestrationMode.ENFORCE,
        clock,
    )
    session, _, _ = public_ids(identity, "worker")
    registry.authorize(session, max_depth=2, duration=1)
    clock.value = 11.0

    assert registry.authorizations() == ()
    result = policy.reconcile(request("worker"))

    assert result.decision.code is OrchestrationDecisionCode.AUTHORIZATION_EXPIRED
    assert not has_agent(result)


def test_concurrent_repeated_requests_all_observe_expired_authorization() -> None:
    from concurrent.futures import ThreadPoolExecutor
    import threading

    clock = Clock()
    policy, registry, identity = coordinator(
        CodexOrchestrationMode.ENFORCE,
        clock,
    )
    session, _, _ = public_ids(identity, "worker")
    registry.authorize(session, max_depth=2, duration=1)
    clock.value = 11.0
    gate = threading.Barrier(9)

    def reconcile():
        gate.wait()
        return policy.reconcile(request("worker")).decision.code

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(reconcile) for _ in range(8)]
        gate.wait()
        decisions = [future.result(timeout=5) for future in futures]

    assert decisions == [OrchestrationDecisionCode.AUTHORIZATION_EXPIRED] * 8
