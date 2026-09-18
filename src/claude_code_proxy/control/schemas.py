"""Validated response schemas for the local control API."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..limits import MAX_CONTROL_INTEGER


NonNegativeControlInteger = Annotated[
    int, Field(strict=True, ge=0, le=MAX_CONTROL_INTEGER)
]
PositiveControlInteger = Annotated[
    int, Field(strict=True, ge=1, le=MAX_CONTROL_INTEGER)
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


def _require_wire_duration(value: object) -> object:
    if type(value) not in (int, float):
        raise ValueError("duration must be a JSON number")
    return value


class SessionCounts(_FrozenModel):
    active: NonNegativeControlInteger
    retained: NonNegativeControlInteger


class HealthResponse(_FrozenModel):
    protocol_version: Literal[1] = 1
    application_version: str
    pid: PositiveControlInteger
    started_at: datetime
    uptime_seconds: float
    capabilities: tuple[str, ...] = ("sessions", "agents")
    sessions: SessionCounts
    inactive_limit: NonNegativeControlInteger

    @field_validator("uptime_seconds", mode="before")
    @classmethod
    def validate_uptime_wire_type(cls, value: object) -> object:
        return _require_wire_duration(value)


class _ActivityResponse(_FrozenModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: str
    state: Literal["active", "idle", "failed"]
    active_requests: NonNegativeControlInteger
    requests: NonNegativeControlInteger
    client_model: str
    model: str
    provider: str
    transport: str
    effort: str
    context_window: PositiveControlInteger | None
    first_seen: datetime
    last_seen: datetime
    elapsed_seconds: float
    last_result: Literal["completed", "failed"] | None

    @field_validator("elapsed_seconds", mode="before")
    @classmethod
    def validate_elapsed_wire_type(cls, value: object) -> object:
        return _require_wire_duration(value)


class AgentResponse(_ActivityResponse):
    parent_id: str | None


class SessionResponse(_ActivityResponse):
    agents: tuple[AgentResponse, ...] = ()


class SessionListResponse(_FrozenModel):
    captured_at: datetime
    sessions: tuple[SessionResponse, ...]
