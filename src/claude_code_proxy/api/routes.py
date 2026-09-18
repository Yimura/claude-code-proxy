"""Anthropic-compatible HTTP routes."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..domain.models import CompletionRequest
from ..logging import (
    FAILURE_LOGGED,
    REQUEST_LOG_CONTEXT,
    client_identity_from_headers,
    RequestLogContext,
    agent_identity,
    effective_effort,
    log_agent_started,
    log_provider_failure,
    log_session_started,
    log_stream_failure,
    log_unexpected_failure,
    observe_stream,
    session_identity,
)
from ..observability import (
    ObservationHandle,
    SessionMetadata,
    SessionRegistry,
    SessionResult,
)
from ..providers.base import ProviderError
from ..service import ProxyService
from .schemas import MessagesRequest, TokenCountRequest, TokenCountResponse
from .translation import (
    DONE_FRAME,
    normalize_request,
    serialize_stream,
    to_api_response,
)


def build_router(service: ProxyService, sessions: SessionRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/messages")
    async def create_message(request: MessagesRequest, raw_request: Request):
        normalized = normalize_request(
            request,
            client_identity=client_identity_from_headers(raw_request.headers),
        )
        prepared = service.prepare(normalized)
        observation, context = _observe_request(
            raw_request, prepared, service, sessions
        )
        _record_context(raw_request, context)
        if request.stream:
            return _streaming_response(
                raw_request,
                prepared,
                service,
                sessions,
                observation,
                context,
            )

        result: SessionResult = "failed"
        try:
            response = to_api_response(
                await service.complete_prepared(prepared)
            )
            result = "completed"
            return response
        except ProviderError as error:
            _log_provider_error(raw_request, context, error)
            raise _http_error(error) from error
        except Exception as error:
            _log_unexpected_error(raw_request, context, error)
            raise
        finally:
            sessions.finish(observation, result)

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(request: TokenCountRequest, raw_request: Request):
        normalized = normalize_request(
            _as_messages_request(request),
            client_identity=client_identity_from_headers(raw_request.headers),
        )
        prepared = service.prepare(normalized)
        observation, context = _observe_request(
            raw_request, prepared, service, sessions
        )
        _record_context(raw_request, context)
        result: SessionResult = "failed"
        try:
            count = await service.count_tokens_prepared(prepared)
            response = TokenCountResponse(input_tokens=count)
            result = "completed"
            return response
        except ProviderError as error:
            _log_provider_error(raw_request, context, error)
            raise _http_error(error) from error
        except Exception as error:
            _log_unexpected_error(raw_request, context, error)
            raise
        finally:
            sessions.finish(observation, result)

    @router.head("/api/hello")
    async def hello():
        return Response()

    @router.get("/")
    async def root():
        return {"message": "Anthropic Proxy for LiteLLM"}

    return router


def _streaming_response(
    raw_request: Request,
    prepared: CompletionRequest,
    service: ProxyService,
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
) -> StreamingResponse:
    try:
        events = service.stream_prepared(prepared)
        observed = observe_stream(events, context)
        serialized = serialize_stream(
            prepared,
            observed,
            on_error=lambda error: log_stream_failure(context, error),
        )
        return StreamingResponse(
            _record_stream_lifecycle(serialized, sessions, observation),
            media_type="text/event-stream",
        )
    except ProviderError as error:
        sessions.finish(observation, "failed")
        _log_provider_error(raw_request, context, error)
        raise _http_error(error) from error
    except Exception as error:
        sessions.finish(observation, "failed")
        _log_unexpected_error(raw_request, context, error)
        raise


async def _record_stream_lifecycle(
    frames: AsyncIterator[str],
    sessions: SessionRegistry,
    observation: ObservationHandle,
) -> AsyncIterator[str]:
    iterator = aiter(frames)
    result: SessionResult = "failed"
    try:
        async for frame in iterator:
            yield frame
            if frame == DONE_FRAME:
                result = "completed"
    finally:
        try:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
        finally:
            sessions.finish(observation, result)


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



def _upstream_provider(model: str) -> str:
    prefix, separator, _ = model.partition("/")
    if separator and prefix in {"openai", "gemini", "anthropic"}:
        return prefix
    return "openai"


def _session_metadata(
    prepared: CompletionRequest, transport: str
) -> SessionMetadata:
    return SessionMetadata(
        client_identity=prepared.client_identity,
        client_model=prepared.original_model,
        upstream_model=prepared.model,
        provider=_upstream_provider(prepared.model),
        transport=transport,
        effort=effective_effort(prepared.reasoning),
        context_window=prepared.context_window,
    )


def _observe_request(
    raw_request: Request,
    prepared: CompletionRequest,
    service: ProxyService,
    sessions: SessionRegistry,
) -> tuple[ObservationHandle, RequestLogContext]:
    transport = service.provider_for(prepared).name
    effort = effective_effort(prepared.reasoning)
    observation = sessions.begin(_session_metadata(prepared, transport))
    identity = session_identity(
        observation.public_id,
        request_scoped=observation.request_scoped,
        is_new=observation.is_new,
    )
    context = RequestLogContext(
        session=identity,
        agent=agent_identity(
            observation.agent_public_id,
            observation.parent_agent_public_id,
            is_new=observation.agent_is_new,
        ),
        method=raw_request.method,
        endpoint=raw_request.url.path,
        original_model=prepared.original_model,
        upstream_model=prepared.model,
        provider=transport,
        effort=effort,
    )
    return observation, context


def _record_context(raw_request: Request, context: RequestLogContext) -> None:
    setattr(raw_request.state, REQUEST_LOG_CONTEXT, context)
    if context.session.is_new:
        log_session_started(context)
    if context.agent is not None and context.agent.is_new:
        log_agent_started(context)


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
