from datetime import UTC, datetime
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from claude_code_proxy.api.routes import build_router
from claude_code_proxy.config import ModelConfig
from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.control.schemas import PerformanceActivityResponse
from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionResponse,
    TextBlock,
    TokenUsage,
)
from claude_code_proxy.logging import RequestLoggingMiddleware
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.observability import SessionMetadata, SessionRegistry
from claude_code_proxy.service import ProxyService
import claude_code_proxy.text_safety as text_safety

TELEMETRY_MODEL_MAX_LENGTH = 256
TELEMETRY_ATTRIBUTE_MAX_LENGTH = 64


class RecordingProvider:
    name = "fake"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request, telemetry=None):
        self.requests.append(request)
        return CompletionResponse(
            "msg-1",
            "safe-response-model",
            (TextBlock("hello"),),
            "end_turn",
            TokenUsage(2, 1),
        )

    async def count_tokens(self, request, telemetry=None):
        self.requests.append(request)
        return 7

    async def stream(self, request, telemetry=None):
        if False:
            yield


def public_app(
    provider: RecordingProvider, sessions: SessionRegistry
) -> FastAPI:
    service = ProxyService(
        ModelResolver(ModelConfig({}, {}, {})),
        "litellm",
        provider,
        provider,
    )
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, sessions=sessions)
    app.include_router(build_router(service, sessions))
    return app


def metadata(**changes: str) -> SessionMetadata:
    values: dict[str, object] = {
        "client_identity": ClientIdentity("session"),
        "client_model": "claude-opus",
        "upstream_model": "openai/gpt-test",
        "provider": "openai",
        "transport": "codex",
        "effort": "high",
        "context_window": 1000,
    }
    values.update(changes)
    return SessionMetadata(**values)


def activity_payload() -> dict[str, object]:
    return {
        "id": "session-public",
        "state": "idle",
        "active_requests": 0,
        "requests": 1,
        "client_model": "client-model",
        "model": "provider-model",
        "provider": "provider",
        "transport": "transport",
        "effort": "high",
        "context_window": 1000,
        "first_seen": "2026-01-02T03:04:05Z",
        "last_seen": "2026-01-02T03:04:06Z",
        "elapsed_seconds": 1,
        "last_result": "completed",
    }


def test_retained_text_encodes_atoms_and_truncates_without_splitting() -> None:
    value = "a" * 252 + "\n" + "discarded"

    normalizer = getattr(text_safety, "retained_telemetry_text", None)
    assert callable(normalizer)
    assert getattr(text_safety, "TELEMETRY_MODEL_MAX_LENGTH", None) == 256
    assert getattr(text_safety, "TELEMETRY_ATTRIBUTE_MAX_LENGTH", None) == 64
    result = normalizer(value, max_length=TELEMETRY_MODEL_MAX_LENGTH)

    assert result == "a" * 252 + "..."
    assert len(result) <= TELEMETRY_MODEL_MAX_LENGTH
    assert result.isprintable()
    assert not result.endswith(("\\", "\\x", "\\u", "\\U"))


@pytest.mark.parametrize("value", [b"bytes", "", " \t\n"])
def test_retained_text_requires_nonblank_str(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalizer = getattr(text_safety, "retained_telemetry_text", None)
        assert callable(normalizer)
        normalizer(value, max_length=64)


async def test_large_metadata_is_bounded_in_every_retained_event() -> None:
    raw_model = "m" * 1_000_000
    sessions = SessionRegistry(10, secret=b"x" * 32)
    retained = metadata(
        client_model=raw_model,
        upstream_model=f"openai/{raw_model}",
    )
    handle = sessions.begin(retained)
    sessions.record_retry(handle)
    sessions.finish(handle, "completed")

    subscription = sessions.events.subscribe(0)
    try:
        events = subscription.replay
    finally:
        subscription.close()

    assert len(retained.client_model) == TELEMETRY_MODEL_MAX_LENGTH
    assert len(retained.upstream_model) == TELEMETRY_MODEL_MAX_LENGTH
    assert retained.client_model.endswith("...")
    snapshot = sessions.snapshots()[0]
    assert len(snapshot.client_model) == TELEMETRY_MODEL_MAX_LENGTH
    assert len(snapshot.model) <= TELEMETRY_MODEL_MAX_LENGTH
    assert events
    assert all(
        len(event.activity.client_model) <= TELEMETRY_MODEL_MAX_LENGTH
        and len(event.activity.model) <= TELEMETRY_MODEL_MAX_LENGTH
        for event in events
    )
    retained_characters = sum(
        len(event.activity.client_model) + len(event.activity.model)
        for event in events
    )
    assert retained_characters <= len(events) * TELEMETRY_MODEL_MAX_LENGTH * 2


def test_attribute_metadata_is_printable_and_bounded() -> None:
    raw = "value\n\x00‮界" + "x" * 1000
    retained = metadata(provider=raw, transport=raw, effort=raw)

    for value in (retained.provider, retained.transport, retained.effort):
        assert len(value) <= TELEMETRY_ATTRIBUTE_MAX_LENGTH
        assert value.isprintable()
        assert "\\x0a\\x00\\u202e界" in value
        assert value.endswith("...")


@pytest.mark.parametrize(
    ("field", "limit"),
    [
        ("client_model", TELEMETRY_MODEL_MAX_LENGTH),
        ("model", TELEMETRY_MODEL_MAX_LENGTH),
        ("provider", TELEMETRY_ATTRIBUTE_MAX_LENGTH),
        ("transport", TELEMETRY_ATTRIBUTE_MAX_LENGTH),
        ("effort", TELEMETRY_ATTRIBUTE_MAX_LENGTH),
    ],
)
def test_performance_activity_schema_enforces_retention_limits(
    field: str, limit: int
) -> None:
    payload = activity_payload()
    payload[field] = "x" * limit
    assert getattr(
        PerformanceActivityResponse.model_validate(payload), field
    ) == "x" * limit

    payload[field] = "x" * (limit + 1)
    with pytest.raises(ValidationError, match="string_too_long"):
        PerformanceActivityResponse.model_validate(payload)


async def test_public_control_and_watch_escape_model_without_routing_change() -> None:
    raw_model = "ordinary\n\x00‮界"
    expected = "ordinary\\x0a\\x00\\u202e界"
    provider = RecordingProvider()
    sessions = SessionRegistry(10, secret=b"x" * 32)
    public = TestClient(public_app(provider, sessions))

    response = public.post(
        "/v1/messages",
        content=json.dumps(
            {
                "model": raw_model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode(),
        headers={"content-type": "application/json"},
    )
    control = TestClient(
        create_control_app(sessions, started_at=datetime.now(UTC))
    )
    performance = control.get("/v1/performance")
    subscription = sessions.events.subscribe(0)
    try:
        watched = subscription.replay
    finally:
        subscription.close()

    assert response.status_code == 200
    assert provider.requests[0].original_model == raw_model
    assert provider.requests[0].model == raw_model
    assert performance.status_code == 200
    session = performance.json()["sessions"][0]["session"]
    assert session["client_model"] == expected
    assert session["model"] == expected
    assert all(event.activity.client_model == expected for event in watched)
    assert all(event.activity.model == expected for event in watched)
    assert all(event.activity.client_model.isprintable() for event in watched)
