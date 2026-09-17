"""FastAPI application construction and dependency wiring."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import build_router
from .logging import SessionTracker, request_logging_middleware
from .runtime import RuntimeServices, create_runtime


def create_app(runtime: RuntimeServices | None = None) -> FastAPI:
    runtime = runtime or create_runtime()
    session_tracker = SessionTracker()

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        if runtime.settings.openai_transport == "codex":
            await runtime.codex_auth.initialize()
        yield

    application = FastAPI(lifespan=lifespan)
    application.state.runtime = runtime
    application.middleware("http")(
        request_logging_middleware(session_tracker)
    )
    application.include_router(build_router(runtime.service, session_tracker))
    return application
