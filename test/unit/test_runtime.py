from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

import claude_code_proxy.runtime as runtime_module
from claude_code_proxy.config import Settings
from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionRequest,
    Message,
    TextBlock,
)
from claude_code_proxy.observability import SessionMetadata, SessionRegistry
from claude_code_proxy.public_identity import PublicIdentity
from claude_code_proxy.providers.codex.orchestration_policy import (
    OrchestrationPolicyCoordinator,
)
from claude_code_proxy.providers.codex.orchestration_registry import (
    OrchestrationRegistry,
)
from claude_code_proxy.providers.codex.auth import CodexAuth
from claude_code_proxy.reasoning import ReasoningPolicy
from claude_code_proxy.service import ProxyService


def settings(tmp_path, *, transport="litellm", retention_limit=17):
    return Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        openai_base_url=None,
        gemini_api_key=None,
        use_vertex_auth=False,
        vertex_project="unset",
        vertex_location="unset",
        openai_transport=transport,
        opencode_data_dir=tmp_path,
        model_mapping_path=tmp_path / "missing.json",
        session_retention_limit=retention_limit,
    )


def request(model):
    return CompletionRequest(
        original_model=model,
        model=model,
        response_model=model,
        max_tokens=100,
        messages=(Message("user", (TextBlock("hello"),)),),
        reasoning=ReasoningPolicy(None, None),
    )


def test_create_runtime_uses_exact_settings_and_configured_retention(tmp_path):
    configured = settings(tmp_path)

    runtime = runtime_module.create_runtime(configured)

    assert runtime.settings is configured
    assert isinstance(runtime.service, ProxyService)
    assert isinstance(runtime.codex_auth, CodexAuth)
    assert isinstance(runtime.sessions, SessionRegistry)
    assert runtime.sessions.inactive_limit == 17
    assert runtime.started_at.utcoffset() == timedelta(0)


def test_runtime_services_is_frozen(tmp_path):
    runtime = runtime_module.create_runtime(settings(tmp_path))

    with pytest.raises(FrozenInstanceError):
        runtime.started_at = runtime.started_at


def test_create_runtime_defaults_to_environment_settings(tmp_path, monkeypatch):
    configured = settings(tmp_path)
    monkeypatch.setattr(
        runtime_module.Settings,
        "from_environment",
        staticmethod(lambda: configured),
    )

    runtime = runtime_module.create_runtime()

    assert runtime.settings is configured


def test_create_runtime_shares_auth_with_codex_service(tmp_path, monkeypatch):
    class FakeAuth:
        def __init__(self, data_dir):
            self.data_dir = data_dir

    class FakeLiteLLMProvider:
        name = "litellm"
        instances = []

        def __init__(self, configured):
            self.settings = configured
            self.instances.append(self)

        async def count_tokens(self, completion_request):
            return 0

    class FakeCodexProvider:
        name = "codex"

        def __init__(self, auth, *, token_counter, orchestration):
            self.auth = auth
            self.token_counter = token_counter
            self.orchestration = orchestration

    monkeypatch.setattr(runtime_module, "LiteLLMProvider", FakeLiteLLMProvider)
    monkeypatch.setattr(runtime_module, "CodexAuth", FakeAuth)
    monkeypatch.setattr(runtime_module, "CodexProvider", FakeCodexProvider)

    configured = settings(tmp_path, transport="codex")
    runtime = runtime_module.create_runtime(configured)
    provider = runtime.service.provider_for(request("openai/gpt-5.6-sol"))
    litellm_provider = FakeLiteLLMProvider.instances[0]

    assert litellm_provider.settings is configured
    assert provider.auth is runtime.codex_auth
    assert provider.token_counter == litellm_provider.count_tokens
    assert provider.orchestration is runtime.orchestration


def test_create_runtime_shares_one_event_journal_with_registry(tmp_path):
    runtime = runtime_module.create_runtime(settings(tmp_path))

    assert runtime.events is runtime.sessions.events


def test_create_runtime_shares_identity_and_wires_orchestration_eviction(tmp_path):
    runtime = runtime_module.create_runtime(
        settings(tmp_path, transport="codex", retention_limit=0)
    )

    assert isinstance(runtime.identity, PublicIdentity)
    assert isinstance(runtime.orchestration_registry, OrchestrationRegistry)
    assert isinstance(runtime.orchestration, OrchestrationPolicyCoordinator)
    assert runtime.sessions.public_id("session") == runtime.identity.public_id("session")

    public_session = runtime.identity.public_id("session")
    public_agent = runtime.identity.public_agent_id("session", "agent")
    runtime.orchestration_registry.observe_lineage(
        public_session, public_agent, None
    )
    handle = runtime.sessions.begin(
        SessionMetadata(
            client_identity=ClientIdentity("session"),
            client_model="claude",
            upstream_model="openai/gpt",
            provider="codex",
            transport="codex",
            effort="high",
            context_window=None,
        )
    )
    runtime.sessions.finish(handle, "completed")

    assert runtime.orchestration_registry.lineage(
        public_session, public_agent
    ) is None
