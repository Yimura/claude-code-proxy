"""Anthropic-compatible HTTP routes."""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.requests import ClientDisconnect

from ..domain.models import CompletionRequest, StreamError
from ..failures import (
    FailureDiagnostic,
    FailureStage,
    unexpected_failure_diagnostic,
)
from ..logging import (
    FAILURE_LOGGED,
    REQUEST_FINALIZER,
    REQUEST_LOG_CONTEXT,
    RequestLogContext,
    agent_identity,
    client_identity_from_headers,
    effective_effort,
    log_agent_started,
    log_finalization_failure,
    log_performance,
    log_provider_failure,
    log_session_started,
    log_stream_failure,
    log_telemetry_failure,
    log_unexpected_failure,
    observe_stream,
    provider_failure_diagnostic,
    session_identity,
    stream_failure_diagnostic,
)
from ..observability import ObservationHandle, SessionMetadata, SessionRegistry
from ..performance import (
    OperationKind,
    RequestOutcome,
    RequestTelemetry,
    notify_telemetry,
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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _monotonic_now() -> float:
    return time.monotonic()


def build_router(service: ProxyService, sessions: SessionRegistry) -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/v1/messages",
        _message_handler(service, sessions),
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/messages/count_tokens",
        _count_handler(service, sessions),
        methods=["POST"],
    )
    router.add_api_route("/api/hello", _hello, methods=["HEAD"], name="hello")
    router.add_api_route("/", _root, methods=["GET"], name="root")
    return router


def _message_handler(service: ProxyService, sessions: SessionRegistry):
    async def create_message(request: MessagesRequest, raw_request: Request):
        started_at = _utc_now()
        started_monotonic = _monotonic_now()
        normalized = normalize_request(
            request,
            client_identity=client_identity_from_headers(raw_request.headers),
        )
        prepared = service.prepare(normalized)
        observation, context = _observe_request(
            raw_request,
            prepared,
            service,
            sessions,
            "messages",
            started_at,
            started_monotonic,
        )
        telemetry = _request_telemetry(sessions, observation)
        _record_context(raw_request, context)
        _install_finalizer(raw_request, sessions, observation, context)
        if request.stream:
            return _streaming_response(
                raw_request,
                prepared,
                service,
                sessions,
                observation,
                context,
                telemetry,
            )
        return await _complete_response(
            raw_request,
            prepared,
            service,
            sessions,
            observation,
            context,
            telemetry,
        )

    return create_message


def _count_handler(service: ProxyService, sessions: SessionRegistry):
    async def count_tokens(request: TokenCountRequest, raw_request: Request):
        started_at = _utc_now()
        started_monotonic = _monotonic_now()
        normalized = normalize_request(
            _as_messages_request(request),
            client_identity=client_identity_from_headers(raw_request.headers),
        )
        prepared = service.prepare(normalized)
        observation, context = _observe_request(
            raw_request,
            prepared,
            service,
            sessions,
            "count_tokens",
            started_at,
            started_monotonic,
        )
        telemetry = _request_telemetry(sessions, observation)
        _record_context(raw_request, context)
        _install_finalizer(raw_request, sessions, observation, context)
        return await _count_response(
            raw_request,
            prepared,
            service,
            sessions,
            observation,
            context,
            telemetry,
        )

    return count_tokens


async def _hello():
    return Response()


async def _root():
    return {"message": "Anthropic Proxy for LiteLLM"}


async def _complete_response(
    raw_request: Request,
    prepared: CompletionRequest,
    service: ProxyService,
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
    telemetry: RequestTelemetry | None,
):
    try:
        response = to_api_response(
            await service.complete_prepared(prepared, telemetry)
        )
    except ProviderError as error:
        _log_provider_error(raw_request, context, error)
        _finalize_request(
            sessions,
            observation,
            context,
            "failed",
            provider_failure_diagnostic(error),
        )
        raise _http_error(error) from error
    except asyncio.CancelledError:
        _finalize_request(sessions, observation, context, "cancelled")
        raise
    except Exception as error:
        _log_unexpected_error(raw_request, context, error)
        _finalize_unexpected(sessions, observation, context, error)
        raise
    _finalize_request(sessions, observation, context, "completed")
    return response


async def _count_response(
    raw_request: Request,
    prepared: CompletionRequest,
    service: ProxyService,
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
    telemetry: RequestTelemetry | None,
):
    try:
        count = await service.count_tokens_prepared(prepared, telemetry)
        notify_telemetry(telemetry, "count_tokens", count)
        response = TokenCountResponse(input_tokens=count)
    except ProviderError as error:
        _log_provider_error(raw_request, context, error)
        _finalize_request(
            sessions,
            observation,
            context,
            "failed",
            provider_failure_diagnostic(error),
        )
        raise _http_error(error) from error
    except asyncio.CancelledError:
        _finalize_request(sessions, observation, context, "cancelled")
        raise
    except Exception as error:
        _log_unexpected_error(raw_request, context, error)
        _finalize_unexpected(sessions, observation, context, error)
        raise
    _finalize_request(sessions, observation, context, "completed")
    return response


def _streaming_response(
    raw_request: Request,
    prepared: CompletionRequest,
    service: ProxyService,
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
    telemetry: RequestTelemetry | None,
) -> StreamingResponse:
    try:
        events = service.stream_prepared(prepared, telemetry)
        observed = observe_stream(
            events,
            context,
            on_error=lambda error: _mark_stream_error(raw_request, error),
        )
        serialized = serialize_stream(
            prepared,
            observed,
            on_error=lambda error: _record_stream_error(
                raw_request, context, error
            ),
        )
        lifecycle = _record_stream_lifecycle(
            serialized, raw_request, context, sessions, observation
        )
        return StreamingResponse(lifecycle, media_type="text/event-stream")
    except ProviderError as error:
        _log_provider_error(raw_request, context, error)
        _finalize_request(
            sessions,
            observation,
            context,
            "failed",
            provider_failure_diagnostic(error),
        )
        raise _http_error(error) from error
    except Exception as error:
        _log_unexpected_error(raw_request, context, error)
        _finalize_unexpected(sessions, observation, context, error)
        raise


async def _record_stream_lifecycle(
    frames: AsyncIterator[str],
    raw_request: Request,
    context: RequestLogContext,
    sessions: SessionRegistry,
    observation: ObservationHandle,
) -> AsyncIterator[str]:
    iterator = aiter(frames)
    stream_state = _stream_terminal_state(raw_request)
    outcome: RequestOutcome = "failed"
    original: BaseException | None = None
    try:
        async for frame in iterator:
            if frame == DONE_FRAME:
                outcome = "completed"
            yield frame
    except GeneratorExit as error:
        original = error
        if not stream_state.has_error:
            outcome = "client_disconnected"
        raise
    except ClientDisconnect as error:
        original = error
        if not stream_state.has_error:
            outcome = "client_disconnected"
        raise
    except asyncio.CancelledError as error:
        original = error
        if not stream_state.has_error:
            outcome = await _cancelled_outcome(raw_request)
        raise
    except BaseException as error:
        original = error
        raise
    finally:
        try:
            await _close_stream_iterator(iterator, original)
        finally:
            _finalize_stream_request(
                raw_request, sessions, observation, context, outcome
            )


_STREAM_TERMINAL_STATE = "stream_terminal_state"


@dataclass(slots=True)
class _StreamTerminalState:
    has_error: bool = False
    failure: FailureDiagnostic | None = None


def _stream_terminal_state(raw_request: Request) -> _StreamTerminalState:
    state = getattr(raw_request.state, _STREAM_TERMINAL_STATE, None)
    if isinstance(state, _StreamTerminalState):
        return state
    state = _StreamTerminalState()
    setattr(raw_request.state, _STREAM_TERMINAL_STATE, state)
    return state


def _mark_stream_error(raw_request: Request, error: StreamError) -> None:
    state = _stream_terminal_state(raw_request)
    state.has_error = True
    if state.failure is None:
        state.failure = stream_failure_diagnostic(error)


def _record_stream_error(
    raw_request: Request,
    context: RequestLogContext,
    error: StreamError,
) -> None:
    _mark_stream_error(raw_request, error)
    try:
        log_stream_failure(context, error)
    except BaseException:
        pass


async def _cancelled_outcome(raw_request: Request) -> RequestOutcome:
    try:
        with anyio.CancelScope(shield=True):
            disconnected = await raw_request.is_disconnected()
    except BaseException:
        return "cancelled"
    return "client_disconnected" if disconnected else "cancelled"


async def _close_stream_iterator(
    iterator: object, original: BaseException | None
) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except BaseException:
        if original is None:
            raise


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
    operation: OperationKind,
    started_at: datetime,
    started_monotonic: float,
) -> tuple[ObservationHandle, RequestLogContext]:
    transport = service.provider_for(prepared).name
    observation = sessions.begin(
        _session_metadata(prepared, transport),
        operation=operation,
        started_at=started_at,
        started_monotonic=started_monotonic,
    )
    context = _request_context(raw_request, prepared, observation, transport)
    return observation, context


def _request_context(
    raw_request: Request,
    prepared: CompletionRequest,
    observation: ObservationHandle,
    transport: str,
) -> RequestLogContext:
    identity = session_identity(
        observation.public_id,
        request_scoped=observation.request_scoped,
        is_new=observation.is_new,
    )
    return RequestLogContext(
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
        effort=effective_effort(prepared.reasoning),
    )


def _request_telemetry(
    sessions: SessionRegistry, observation: ObservationHandle
) -> RequestTelemetry | None:
    try:
        return sessions.observer(observation)
    except BaseException:
        log_telemetry_failure()
        return None


def _record_context(raw_request: Request, context: RequestLogContext) -> None:
    setattr(raw_request.state, REQUEST_LOG_CONTEXT, context)
    if context.session.is_new:
        log_session_started(context)
    if context.agent is not None and context.agent.is_new:
        log_agent_started(context)


def _install_finalizer(
    raw_request: Request,
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
) -> None:
    def finalize(
        outcome: RequestOutcome,
        failure: FailureDiagnostic | None = None,
    ) -> None:
        _finalize_stream_request(
            raw_request,
            sessions,
            observation,
            context,
            outcome,
            failure,
        )

    setattr(raw_request.state, REQUEST_FINALIZER, finalize)


def _finalize_stream_request(
    raw_request: Request,
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
    outcome: RequestOutcome,
    failure: FailureDiagnostic | None = None,
) -> None:
    stream_state = getattr(raw_request.state, _STREAM_TERMINAL_STATE, None)
    if isinstance(stream_state, _StreamTerminalState) and stream_state.has_error:
        outcome = "failed"
        failure = stream_state.failure or failure
    _finalize_request(sessions, observation, context, outcome, failure)


def _finalize_unexpected(
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
    error: Exception,
) -> None:
    diagnostic = unexpected_failure_diagnostic(error, stage=FailureStage.ROUTE)
    _finalize_request(sessions, observation, context, "failed", diagnostic)


def _finalize_request(
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
    outcome: RequestOutcome,
    failure: FailureDiagnostic | None = None,
) -> None:
    try:
        snapshot = sessions.finish(observation, outcome, failure)
        if snapshot is not None:
            log_performance(snapshot, context)
    except BaseException:
        log_finalization_failure()


def _log_provider_error(
    raw_request: Request, context: RequestLogContext, error: ProviderError
) -> None:
    try:
        log_provider_failure(context, error)
    except BaseException:
        pass
    setattr(raw_request.state, FAILURE_LOGGED, True)


def _log_unexpected_error(
    raw_request: Request, context: RequestLogContext, error: Exception
) -> None:
    try:
        log_unexpected_failure(context, error)
    except BaseException:
        pass
    setattr(raw_request.state, FAILURE_LOGGED, True)


def _http_error(error: ProviderError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=str(error))
