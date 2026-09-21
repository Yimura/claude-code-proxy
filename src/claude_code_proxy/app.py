"""FastAPI application construction and dependency wiring."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import build_router
from .logging import RequestLoggingMiddleware, log_startup_summary
from .runtime import RuntimeServices, create_runtime


def create_app(runtime: RuntimeServices | None = None) -> FastAPI:
    runtime = runtime or create_runtime()

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        identity = None
        if runtime.settings.openai_transport == "codex":
            identity = await runtime.codex_auth.initialize()
        log_startup_summary(runtime.settings.openai_transport, identity)
        yield

    application = FastAPI(lifespan=lifespan)
    application.state.runtime = runtime
    application.add_middleware(
        RequestLoggingMiddleware, sessions=runtime.sessions
    )
    application.include_router(build_router(runtime.service, runtime.sessions))
    return application
