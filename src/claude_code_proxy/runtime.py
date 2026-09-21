"""Shared runtime service construction."""

from dataclasses import dataclass
from datetime import UTC, datetime

from .config import Settings, load_model_mapping
from .event_journal import EventJournal
from .model_mapping import ModelResolver
from .observability import SessionRegistry
from .providers.codex.auth import CodexAuth
from .providers.codex.provider import CodexProvider
from .providers.litellm import LiteLLMProvider
from .service import ProxyService


@dataclass(frozen=True)
class RuntimeServices:
    """Long-lived services shared by the proxy application."""

    settings: Settings
    service: ProxyService
    codex_auth: CodexAuth
    sessions: SessionRegistry
    events: EventJournal
    started_at: datetime


def create_runtime(settings: Settings | None = None) -> RuntimeServices:
    """Construct the process-wide services from one settings object."""
    configured = settings or Settings.from_environment()
    resolver = ModelResolver(load_model_mapping(configured.model_mapping_path))
    litellm_provider = LiteLLMProvider(configured)
    codex_auth = CodexAuth(configured.opencode_data_dir)
    codex_provider = CodexProvider(
        codex_auth,
        token_counter=litellm_provider.count_tokens,
    )
    service = ProxyService(
        resolver,
        configured.openai_transport,
        litellm_provider,
        codex_provider,
    )
    events = EventJournal(4096, 64)
    sessions = SessionRegistry(
        configured.session_retention_limit, events=events
    )
    started_at = datetime.now(UTC)
    return RuntimeServices(
        settings=configured,
        service=service,
        codex_auth=codex_auth,
        sessions=sessions,
        events=events,
        started_at=started_at,
    )
