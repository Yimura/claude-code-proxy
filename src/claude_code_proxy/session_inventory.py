"""Shared base-session lifecycle helpers independent of performance data."""

from datetime import datetime
from typing import Literal, Protocol


SessionState = Literal["active", "idle", "failed"]
SessionResult = Literal["completed", "failed"]


class ActivityRecord(Protocol):
    metadata: object
    last_seen: datetime
    requests: int
    active: dict[str, float]
    latest_duration: float
    last_result: SessionResult | None


def begin_activity(
    record: ActivityRecord,
    metadata: object,
    request_id: str,
    started: float,
    seen_at: datetime,
) -> None:
    record.metadata = metadata
    record.last_seen = max(record.last_seen, seen_at)
    record.requests += 1
    record.active[request_id] = started


def finish_activity(
    record: ActivityRecord,
    request_id: str,
    finished_at: float,
    seen_at: datetime,
    result: SessionResult,
) -> bool:
    started = record.active.pop(request_id, None)
    if started is None:
        return False
    record.latest_duration = max(0.0, finished_at - started)
    record.last_seen = max(record.last_seen, seen_at)
    record.last_result = result
    return True


def session_state(record: ActivityRecord) -> SessionState:
    if record.active:
        return "active"
    if record.last_result == "failed":
        return "failed"
    return "idle"


def elapsed_seconds(record: ActivityRecord, now: float) -> float:
    if not record.active:
        return record.latest_duration
    return max(0.0, now - min(record.active.values()))


def strip_model_prefix(model: str) -> str:
    _, separator, unprefixed = model.partition("/")
    return unprefixed if separator else model
