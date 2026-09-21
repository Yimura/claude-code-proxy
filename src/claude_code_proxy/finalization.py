"""Request-finalization result and cross-mode validation."""

from dataclasses import dataclass

from .failures import FailureDiagnostic
from .performance import RequestOutcome, RequestPerformanceSnapshot


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    """Report whether a request transitioned and its optional performance data."""

    finalized: bool
    performance: RequestPerformanceSnapshot | None


def validate_finalization(
    outcome: RequestOutcome,
    failure: FailureDiagnostic | None,
) -> None:
    if outcome not in {
        "completed",
        "failed",
        "cancelled",
        "client_disconnected",
    }:
        raise ValueError("finish requires a terminal outcome")
    if outcome != "failed" and failure is not None:
        raise ValueError(
            "non-failed outcome cannot retain a failure diagnostic"
        )
