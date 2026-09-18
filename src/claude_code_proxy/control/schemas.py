"""Validated response schemas for the local control API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class SessionCounts(_FrozenModel):
    active: int
    retained: int


class HealthResponse(_FrozenModel):
    protocol_version: Literal[1] = 1
    application_version: str
    pid: int
    started_at: datetime
    uptime_seconds: float
    capabilities: tuple[str, ...] = ("sessions",)
    sessions: SessionCounts
    inactive_limit: int


class SessionResponse(_FrozenModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: str
    state: Literal["active", "idle", "failed"]
    active_requests: int
    requests: int
    client_model: str
    model: str
    provider: str
    transport: str
    effort: str
    context_window: int | None
    first_seen: datetime
    last_seen: datetime
    elapsed_seconds: float
    last_result: Literal["completed", "failed"] | None


class SessionListResponse(_FrozenModel):
    captured_at: datetime
    sessions: tuple[SessionResponse, ...]
