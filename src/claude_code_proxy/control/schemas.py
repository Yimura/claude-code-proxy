"""Validated response schemas for the local control API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictInt, field_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


def _require_wire_duration(value: object) -> object:
    if type(value) not in (int, float):
        raise ValueError("duration must be a JSON number")
    return value


class SessionCounts(_FrozenModel):
    active: StrictInt
    retained: StrictInt


class HealthResponse(_FrozenModel):
    protocol_version: Literal[1] = 1
    application_version: str
    pid: StrictInt
    started_at: datetime
    uptime_seconds: float
    capabilities: tuple[str, ...] = ("sessions",)
    sessions: SessionCounts
    inactive_limit: StrictInt

    @field_validator("uptime_seconds", mode="before")
    @classmethod
    def validate_uptime_wire_type(cls, value: object) -> object:
        return _require_wire_duration(value)


class SessionResponse(_FrozenModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: str
    state: Literal["active", "idle", "failed"]
    active_requests: StrictInt
    requests: StrictInt
    client_model: str
    model: str
    provider: str
    transport: str
    effort: str
    context_window: StrictInt | None
    first_seen: datetime
    last_seen: datetime
    elapsed_seconds: float
    last_result: Literal["completed", "failed"] | None

    @field_validator("elapsed_seconds", mode="before")
    @classmethod
    def validate_elapsed_wire_type(cls, value: object) -> object:
        return _require_wire_duration(value)


class SessionListResponse(_FrozenModel):
    captured_at: datetime
    sessions: tuple[SessionResponse, ...]
