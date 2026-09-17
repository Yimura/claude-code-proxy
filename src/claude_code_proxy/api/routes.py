"""Anthropic-compatible HTTP routes."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..logging import (
    FAILURE_LOGGED,
    REQUEST_LOG_CONTEXT,
    SESSION_HEADER,
    RequestLogContext,
    SessionTracker,
    effective_effort,
    log_provider_failure,
    log_session_started,
    log_unexpected_failure,
    observe_stream,
)
from ..providers.base import ProviderError
from ..service import ProxyService
from .schemas import MessagesRequest, TokenCountRequest, TokenCountResponse
from .translation import normalize_request, serialize_stream, to_api_response


def build_router(
    service: ProxyService, session_tracker: SessionTracker | None = None
) -> APIRouter:
    router = APIRouter()
    tracker = session_tracker or SessionTracker()

    @router.post("/v1/messages")
    async def create_message(request: MessagesRequest, raw_request: Request):
        normalized = normalize_request(
            request, session_id=_session_id(raw_request)
        )
        prepared = service.prepare(normalized)
        context = _request_context(raw_request, prepared, service, tracker)
        _record_context(raw_request, context)
        if request.stream:
            return StreamingResponse(
                serialize_stream(
                    prepared,
                    observe_stream(service.stream_prepared(prepared), context),
                ),
                media_type="text/event-stream",
            )
        try:
            return to_api_response(await service.complete_prepared(prepared))
        except ProviderError as error:
            _log_provider_error(raw_request, context, error)
            raise _http_error(error) from error
        except Exception as error:
            _log_unexpected_error(raw_request, context, error)
            raise

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(request: TokenCountRequest, raw_request: Request):
        normalized = normalize_request(
            _as_messages_request(request),
            session_id=_session_id(raw_request),
        )
        prepared = service.prepare(normalized)
        context = _request_context(raw_request, prepared, service, tracker)
        _record_context(raw_request, context)
        try:
            count = await service.count_tokens_prepared(prepared)
            return TokenCountResponse(input_tokens=count)
        except ProviderError as error:
            _log_provider_error(raw_request, context, error)
            raise _http_error(error) from error
        except Exception as error:
            _log_unexpected_error(raw_request, context, error)
            raise

    @router.head("/api/hello")
    async def hello():
        return Response()

    @router.get("/")
    async def root():
        return {"message": "Anthropic Proxy for LiteLLM"}

    return router


def _as_messages_request(request: TokenCountRequest) -> MessagesRequest:
    return MessagesRequest(
        model=request.model,
        max_tokens=100,
        messages=request.messages,
        system=request.system,
        tools=request.tools,
        thinking=request.thinking,
        tool_choice=request.tool_choice,
    )


def _session_id(raw_request: Request) -> str | None:
    value = raw_request.headers.get(SESSION_HEADER)
    if value is None or not value.strip():
        return None
    return value


def _request_context(raw_request, prepared, service, tracker):
    provider = service.provider_for(prepared)
    return RequestLogContext(
        session=tracker.observe(prepared.session_id),
        method=raw_request.method,
        endpoint=raw_request.url.path,
        original_model=prepared.original_model,
        upstream_model=prepared.model,
        provider=provider.name,
        effort=effective_effort(prepared.reasoning),
    )


def _record_context(raw_request: Request, context: RequestLogContext) -> None:
    setattr(raw_request.state, REQUEST_LOG_CONTEXT, context)
    if context.session.is_new:
        log_session_started(context)


def _log_provider_error(
    raw_request: Request, context: RequestLogContext, error: ProviderError
) -> None:
    log_provider_failure(context, error.status_code)
    setattr(raw_request.state, FAILURE_LOGGED, True)


def _log_unexpected_error(
    raw_request: Request, context: RequestLogContext, error: Exception
) -> None:
    log_unexpected_failure(context, type(error).__name__)
    setattr(raw_request.state, FAILURE_LOGGED, True)


def _http_error(error: ProviderError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=str(error))
