"""Request-time Codex orchestration decisions and capability filtering."""

from dataclasses import dataclass
from enum import StrEnum

from ...config import CodexOrchestrationMode
from ...domain.models import CompletionRequest
from ...performance import Measurement
from ...public_identity import PublicIdentity
from .orchestration import reconcile_codex_request
from .orchestration_registry import (
    AuthorizationStatus,
    NestingAuthorization,
    ObservedLineage,
    OrchestrationRegistry,
    ParentState,
)


class OrchestrationDecisionCode(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    ADVISORY = "advisory"
    ROOT_ALLOWED = "root_allowed"
    NESTED_ALLOWED = "nested_allowed"
    NESTED_DENIED = "nested_denied"
    LINEAGE_AMBIGUOUS = "lineage_ambiguous"
    LINEAGE_UNKNOWN = "lineage_unknown"
    LINEAGE_CYCLIC = "lineage_cyclic"
    AUTHORIZATION_EXPIRED = "authorization_expired"
    DEPTH_LIMIT_REACHED = "depth_limit_reached"


@dataclass(frozen=True, slots=True)
class OrchestrationDecision:
    mode: CodexOrchestrationMode
    code: OrchestrationDecisionCode
    depth: Measurement
    authorization_present: bool
    agent_allowed: bool


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    request: CompletionRequest
    decision: OrchestrationDecision


class OrchestrationPolicyCoordinator:
    """Observe current lineage and reconcile one request without caching decisions."""

    def __init__(
        self,
        mode: CodexOrchestrationMode,
        identity: PublicIdentity,
        registry: OrchestrationRegistry,
    ) -> None:
        self._mode = mode
        self._identity = identity
        self._registry = registry

    @property
    def mode(self) -> CodexOrchestrationMode:
        return self._mode

    def reconcile(self, request: CompletionRequest) -> OrchestrationResult:
        if self._mode is CodexOrchestrationMode.OFF:
            decision = OrchestrationDecision(
                self._mode,
                OrchestrationDecisionCode.NOT_APPLICABLE,
                Measurement.not_applicable(),
                False,
                True,
            )
            return OrchestrationResult(request, decision)

        identity = request.client_identity
        raw_agent = (identity.agent_id or "").strip()
        if not raw_agent:
            return self._reconciled(
                request,
                OrchestrationDecisionCode.ADVISORY
                if self._mode is CodexOrchestrationMode.ADVISORY
                else OrchestrationDecisionCode.ROOT_ALLOWED,
                Measurement.not_applicable(),
                authorization_present=False,
                agent_allowed=True,
            )

        observed = self._observe(identity.session_id, raw_agent, identity.parent_agent_id)
        if self._mode is CodexOrchestrationMode.ADVISORY:
            return self._advisory(request, observed)
        return self._enforced(request, observed)

    def _observe(
        self,
        session_id: str | None,
        agent_id: str,
        parent_agent_id: str | None,
    ) -> ObservedLineage | None:
        raw_session = (session_id or "").strip()
        if not raw_session:
            return None
        public_session = self._identity.public_id(raw_session)
        public_agent = self._identity.public_agent_id(raw_session, agent_id)
        raw_parent = (parent_agent_id or "").strip()
        public_parent = (
            self._identity.public_agent_id(raw_session, raw_parent)
            if raw_parent
            else None
        )
        return self._registry.observe_lineage(
            public_session, public_agent, public_parent
        )

    def _advisory(
        self, request: CompletionRequest, lineage: ObservedLineage | None
    ) -> OrchestrationResult:
        depth = _lineage_depth(lineage)
        authorization = self._authorization(lineage)
        return self._reconciled(
            request,
            OrchestrationDecisionCode.ADVISORY,
            depth,
            authorization_present=(
                authorization.status is not AuthorizationStatus.ABSENT
            ),
            agent_allowed=True,
        )

    def _enforced(
        self, request: CompletionRequest, lineage: ObservedLineage | None
    ) -> OrchestrationResult:
        depth = _lineage_depth(lineage)
        unavailable_code = _lineage_denial(lineage)
        authorization = self._authorization(lineage)
        authorization_present = (
            authorization.status is not AuthorizationStatus.ABSENT
        )
        if unavailable_code is not None:
            return self._reconciled(
                request,
                unavailable_code,
                depth,
                authorization_present=authorization_present,
                agent_allowed=False,
            )
        if authorization.status is AuthorizationStatus.EXPIRED:
            return self._reconciled(
                request,
                OrchestrationDecisionCode.AUTHORIZATION_EXPIRED,
                depth,
                authorization_present=True,
                agent_allowed=False,
            )
        if authorization.status is AuthorizationStatus.ABSENT:
            return self._reconciled(
                request,
                OrchestrationDecisionCode.NESTED_DENIED,
                depth,
                authorization_present=False,
                agent_allowed=False,
            )
        assert lineage is not None and lineage.depth is not None
        assert authorization.max_depth is not None
        if lineage.depth >= authorization.max_depth:
            return self._reconciled(
                request,
                OrchestrationDecisionCode.DEPTH_LIMIT_REACHED,
                depth,
                authorization_present=True,
                agent_allowed=False,
            )
        return self._reconciled(
            request,
            OrchestrationDecisionCode.NESTED_ALLOWED,
            depth,
            authorization_present=True,
            agent_allowed=True,
        )

    def _authorization(
        self, lineage: ObservedLineage | None
    ) -> NestingAuthorization:
        if lineage is None:
            return NestingAuthorization(
                "unavailable",
                AuthorizationStatus.ABSENT,
                None,
                None,
                None,
            )
        return self._registry.authorization(lineage.session_id)

    def _reconciled(
        self,
        request: CompletionRequest,
        code: OrchestrationDecisionCode,
        depth: Measurement,
        *,
        authorization_present: bool,
        agent_allowed: bool,
    ) -> OrchestrationResult:
        reconciled = reconcile_codex_request(
            request,
            mode=self._mode,
            agent_allowed=agent_allowed,
            authorization_present=authorization_present,
        )
        decision = OrchestrationDecision(
            self._mode,
            code,
            depth,
            authorization_present,
            agent_allowed,
        )
        return OrchestrationResult(reconciled, decision)


def _lineage_depth(lineage: ObservedLineage | None) -> Measurement:
    if lineage is None or lineage.depth is None:
        return Measurement.unavailable()
    return Measurement.observed(lineage.depth)


def _lineage_denial(
    lineage: ObservedLineage | None,
) -> OrchestrationDecisionCode | None:
    if lineage is None or lineage.parent_state is ParentState.UNKNOWN:
        return OrchestrationDecisionCode.LINEAGE_UNKNOWN
    if lineage.parent_state is ParentState.AMBIGUOUS:
        return OrchestrationDecisionCode.LINEAGE_AMBIGUOUS
    if lineage.parent_state is ParentState.CYCLIC:
        return OrchestrationDecisionCode.LINEAGE_CYCLIC
    return None
