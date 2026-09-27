"""Explicit dependency construction for local control app tests."""

from typing import Any

from fastapi import FastAPI

from claude_code_proxy.config import CodexOrchestrationMode
from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.observability import SessionRegistry
from claude_code_proxy.public_identity import PublicIdentity
from claude_code_proxy.providers.codex.orchestration_registry import (
    OrchestrationRegistry,
)


def create_test_control_app(
    sessions: SessionRegistry,
    **kwargs: Any,
) -> FastAPI:
    """Create a control app with an isolated but explicit orchestration context."""
    kwargs.setdefault(
        "orchestration_registry",
        OrchestrationRegistry(
            identity=PublicIdentity(secret=b"control-app-test-identity")
        ),
    )
    kwargs.setdefault(
        "orchestration_mode",
        CodexOrchestrationMode.ADVISORY,
    )
    return create_control_app(sessions, **kwargs)
