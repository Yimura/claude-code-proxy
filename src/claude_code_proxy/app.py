"""FastAPI application construction and dependency wiring."""

from fastapi import FastAPI
from .api.routes import build_router
from .config import Settings, load_model_mapping
from .logging import SessionTracker, configure_logging, request_logging_middleware
from .model_mapping import ModelResolver
from .providers.codex.auth import CodexAuth
from .providers.codex.provider import CodexProvider
from .providers.litellm import LiteLLMProvider
from .service import ProxyService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    resolver = ModelResolver(load_model_mapping(settings.model_mapping_path))
    litellm_provider = LiteLLMProvider(settings)
    codex_provider = CodexProvider(
        CodexAuth(settings.opencode_data_dir),
        token_counter=litellm_provider.count_tokens,
    )
    service = ProxyService(
        resolver,
        settings.openai_transport,
        litellm_provider,
        codex_provider,
    )
    session_tracker = SessionTracker()
    application = FastAPI()
    application.middleware("http")(request_logging_middleware(session_tracker))
    application.include_router(build_router(service, session_tracker))
    return application


configure_logging()
app = create_app()
