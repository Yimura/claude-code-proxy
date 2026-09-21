from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import TypeAdapter
import pytest

from claude_code_proxy.api.routes import build_router
from claude_code_proxy.cli_common import OutputFormat
from claude_code_proxy.config import ModelConfig, Settings
from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.control.schemas import (
    PerformanceEventResponse,
    PerformanceListResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
)
from claude_code_proxy.domain.models import (
    CompletionResponse,
    StreamComplete,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolUseStart,
)
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.observability import SessionRegistry
from claude_code_proxy.performance_cli import render_performance
from claude_code_proxy.performance_watch_cli import render_watch_event
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.providers.litellm import LiteLLMProvider
from claude_code_proxy.service import ProxyService

_MARKERS = {
    "header": "PRIVACY_HEADER_7cb944",
    "session": "PRIVACY_SESSION_e4d221",
    "agent": "PRIVACY_AGENT_134bcb",
    "parent": "PRIVACY_PARENT_27d79a",
    "auth": "PRIVACY_AUTH_f12a8d",
    "api_key": "PRIVACY_API_KEY_88d473",
    "system": "PRIVACY_SYSTEM_356c88",
    "user": "PRIVACY_USER_b4cb13",
    "tool_name": "PRIVACY_TOOL_NAME_75056d",
    "tool_description": "PRIVACY_TOOL_DESCRIPTION_27df21",
    "tool_schema": "PRIVACY_TOOL_SCHEMA_80e298",
    "tool_input": "PRIVACY_TOOL_INPUT_7832fb",
    "tool_result": "PRIVACY_TOOL_RESULT_12f98c",
    "reasoning": "PRIVACY_REASONING_53a2d8",
    "provider_body": "PRIVACY_PROVIDER_BODY_479041",
    "exception": "PRIVACY_EXCEPTION_29bcf8",
}


class _PrivacyProvider:
    name = "privacy-fake"

    def __init__(self) -> None:
        self.requests: list[object] = []
        self.telemetry: list[object | None] = []
        self.fail_next = False

    async def complete(self, request, telemetry=None):
        self.requests.append(request)
        self.telemetry.append(telemetry)
        if self.fail_next:
            self.fail_next = False
            raise ProviderError(
                f"{_MARKERS['provider_body']} {_MARKERS['exception']}",
                provider=self.name,
                status_code=503,
            )
        return CompletionResponse(
            "provider-response-id",
            request.response_model,
            (TextBlock("client-visible response"),),
            "end_turn",
            TokenUsage(13, 5, cache_read_input_tokens=2, thinking_tokens=1),
        )

    async def stream(self, request, telemetry=None):
        self.requests.append(request)
        self.telemetry.append(telemetry)
        yield TextDelta(f"client-visible {_MARKERS['provider_body']}")
        yield ToolUseStart("slot", "provider-tool-id", _MARKERS["tool_name"])
        yield StreamComplete("tool_use", TokenUsage(17, 7, thinking_tokens=2))

    async def count_tokens(self, request, telemetry=None):
        self.requests.append(request)
        self.telemetry.append(telemetry)
        return 23


def _request_payload(*, stream: bool = False) -> dict[str, object]:
    return {
        "model": "claude-sonnet",
        "max_tokens": 100,
        "system": _MARKERS["system"],
        "messages": [
            {"role": "user", "content": _MARKERS["user"]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "redacted_thinking",
                        "data": _MARKERS["reasoning"],
                    },
                    {
                        "type": "tool_use",
                        "id": "tool-call-id",
                        "name": _MARKERS["tool_name"],
                        "input": {"secret": _MARKERS["tool_input"]},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-call-id",
                        "content": _MARKERS["tool_result"],
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": _MARKERS["tool_name"],
                "description": _MARKERS["tool_description"],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "secret": {
                            "type": "string",
                            "description": _MARKERS["tool_schema"],
                        }
                    },
                },
            }
        ],
        "stream": stream,
    }


def _headers() -> dict[str, str]:
    return {
        "x-privacy-marker": _MARKERS["header"],
        "x-claude-code-session-id": _MARKERS["session"],
        "x-claude-code-agent-id": _MARKERS["agent"],
        "x-claude-code-parent-agent-id": _MARKERS["parent"],
        "authorization": f"Bearer {_MARKERS['auth']}",
        "x-api-key": _MARKERS["api_key"],
    }


def _apps() -> tuple[FastAPI, FastAPI, SessionRegistry, _PrivacyProvider]:
    provider = _PrivacyProvider()
    sessions = SessionRegistry(
        20,
        secret=b"privacy-regression-secret",
        performance_enabled=True,
        performance_logging_enabled=True,
    )
    service = ProxyService(
        ModelResolver(ModelConfig({}, {}, {})),
        "litellm",
        provider,
        provider,
    )
    public = FastAPI()
    public.include_router(build_router(service, sessions))
    control = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="privacy-test",
        pid=4242,
    )
    return public, control, sessions, provider


async def _event_stream(control: FastAPI):
    route = next(
        route
        for route in control.routes
        if route.path == "/v1/performance/events"
    )
    return await route.endpoint(
        filter=None,
        after=None,
        pid=None,
        started_at=None,
    )


async def _next_frame(response) -> tuple[str, PerformanceStreamEvent]:
    frame = await anext(response.body_iterator)
    assert isinstance(frame, str)
    event = TypeAdapter(PerformanceStreamEvent).validate_json(frame)
    return frame, event


def _litellm_settings() -> Settings:
    return Settings(
        anthropic_api_key=_MARKERS["api_key"],
        openai_api_key=_MARKERS["api_key"],
        gemini_api_key=_MARKERS["api_key"],
        vertex_project="unused",
        vertex_location="unused",
        use_vertex_auth=False,
        openai_base_url=None,
        openai_transport="litellm",
        opencode_data_dir=Path("/unused"),
        model_mapping_path=Path("unused.json"),
    )


@dataclass(frozen=True)
class _ControlEvidence:
    snapshot: PerformanceListResponse
    response_text: str
    reset_line: str
    reset: PerformanceResetResponse
    journal_lines: tuple[str, ...]
    journal_events: tuple[PerformanceStreamEvent, ...]


async def _exercise_public_routes(
    public: FastAPI,
    provider: _PrivacyProvider,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=public, raise_app_exceptions=False),
        base_url="http://public",
    ) as client:
        with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
            complete = await client.post(
                "/v1/messages", headers=_headers(), json=_request_payload()
            )
            streamed = await client.post(
                "/v1/messages",
                headers=_headers(),
                json=_request_payload(stream=True),
            )
            provider.fail_next = True
            failed = await client.post(
                "/v1/messages", headers=_headers(), json=_request_payload()
            )
            counted = await client.post(
                "/v1/messages/count_tokens",
                headers=_headers(),
                json=_request_payload(),
            )

    assert complete.status_code == streamed.status_code == counted.status_code == 200
    assert failed.status_code == 503
    assert _MARKERS["provider_body"] in streamed.text
    assert _MARKERS["provider_body"] in failed.text
    assert _MARKERS["exception"] in failed.text


async def _collect_control_evidence(
    control: FastAPI,
    sessions: SessionRegistry,
    stream,
    reset_line: str,
    reset: PerformanceResetResponse,
) -> _ControlEvidence:
    async with AsyncClient(
        transport=ASGITransport(app=control), base_url="http://control"
    ) as client:
        response = await client.get("/v1/performance")
    assert response.status_code == 200
    snapshot = PerformanceListResponse.model_validate(response.json())

    lines: list[str] = []
    events: list[PerformanceStreamEvent] = []
    for _ in range(sessions.events.current_sequence):
        line, event = await _next_frame(stream)
        lines.append(line)
        events.append(event)
    await stream.body_iterator.aclose()
    return _ControlEvidence(
        snapshot,
        response.text,
        reset_line,
        reset,
        tuple(lines),
        tuple(events),
    )


def _assert_provider_boundary(provider: _PrivacyProvider) -> None:
    assert len(provider.requests) == len(provider.telemetry) == 4
    assert all(item is not None for item in provider.telemetry)
    first_request = provider.requests[0]
    identity = first_request.client_identity
    assert identity.session_id == _MARKERS["session"]
    assert identity.agent_id == _MARKERS["agent"]
    assert identity.parent_agent_id == _MARKERS["parent"]
    request_text = repr(first_request)
    for name in (
        "system",
        "user",
        "tool_name",
        "tool_description",
        "tool_schema",
        "tool_input",
        "tool_result",
        "reasoning",
    ):
        assert _MARKERS[name] in request_text

    outbound = LiteLLMProvider(
        _litellm_settings(), object()
    ).build_request(first_request, stream=False)
    outbound_text = repr(outbound)
    assert _MARKERS["user"] in outbound_text
    assert _MARKERS["tool_input"] in outbound_text
    for name in ("header", "session", "agent", "parent", "auth"):
        assert _MARKERS[name] not in outbound_text


def _render_exposed_surfaces(
    sessions: SessionRegistry,
    provider: _PrivacyProvider,
    evidence: _ControlEvidence,
    log_text: str,
) -> str:
    rendered_watch = [
        output
        for event in (evidence.reset, *evidence.journal_events)
        for output_format in (OutputFormat.JSON, OutputFormat.TABLE)
        for output in render_watch_event(event, output_format, no_trunc=True)
    ]
    return "\n".join(
        [
            repr(sessions.performance_snapshots()),
            repr(sessions.snapshots()),
            repr(provider.telemetry),
            evidence.response_text,
            evidence.reset_line,
            *evidence.journal_lines,
            render_performance(
                evidence.snapshot, OutputFormat.JSON, no_trunc=True
            ),
            render_performance(
                evidence.snapshot, OutputFormat.TABLE, no_trunc=True
            ),
            *rendered_watch,
            log_text,
        ]
    )


def _assert_nonempty_safe_evidence(
    evidence: _ControlEvidence,
    exposed_surfaces: str,
    log_text: str,
) -> None:
    assert evidence.snapshot.sessions
    public_id = evidence.snapshot.sessions[0].session.id
    assert re.fullmatch(r"[0-9a-f]{64}", public_id)
    assert public_id in exposed_surfaces
    assert {event.type for event in evidence.journal_events} >= {
        "request_started",
        "completed",
        "failed",
        "tool_use",
    }
    recent = evidence.snapshot.sessions[0].performance.recent_requests
    assert {request.operation for request in recent} == {
        "messages",
        "count_tokens",
    }
    assert '"input_tokens"' in exposed_surfaces
    assert "performance outcome=completed" in log_text
    assert "provider request failed" in log_text
    assert json.loads(evidence.reset_line)["type"] == "reset"
    assert any(
        isinstance(event, PerformanceEventResponse)
        for event in evidence.journal_events
    )


@pytest.mark.asyncio
async def test_performance_surfaces_exclude_all_sensitive_request_content(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    public, control, sessions, provider = _apps()
    stream = await _event_stream(control)
    reset_line, reset_event = await _next_frame(stream)
    assert isinstance(reset_event, PerformanceResetResponse)

    await _exercise_public_routes(public, provider, caplog)
    evidence = await _collect_control_evidence(
        control, sessions, stream, reset_line, reset_event
    )
    _assert_provider_boundary(provider)
    exposed_surfaces = _render_exposed_surfaces(
        sessions, provider, evidence, caplog.text
    )

    for marker in _MARKERS.values():
        assert marker not in exposed_surfaces
    _assert_nonempty_safe_evidence(evidence, exposed_surfaces, caplog.text)
