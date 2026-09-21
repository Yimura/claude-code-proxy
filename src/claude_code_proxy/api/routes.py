"""Anthropic-compatible HTTP routes."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive

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
        started_at, started_monotonic = sessions.sample_clocks()
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
        started_at, started_monotonic = sessions.sample_clocks()
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
        completed = await service.complete_prepared(prepared, telemetry)
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
    try:
        response = _json_response(to_api_response(completed))
    except Exception as error:
        _log_unexpected_error(
            raw_request,
            context,
            error,
            stage=FailureStage.CLIENT_TRANSLATION,
        )
        _finalize_unexpected(
            sessions,
            observation,
            context,
            error,
            stage=FailureStage.CLIENT_TRANSLATION,
        )
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
        response = _json_response(TokenCountResponse(input_tokens=count))
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


def _json_response(content: object) -> JSONResponse:
    return JSONResponse(jsonable_encoder(content))


class _DisconnectTrackingReceive:
    def __init__(self, receive: Receive) -> None:
        self._receive = receive
        self.disconnected = False

    async def __call__(self) -> Message:
        message = await self._receive()
        if message["type"] == "http.disconnect":
            self.disconnected = True
        return message


class _LifecycleStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncIterator[str],
        raw_request: Request,
        context: RequestLogContext,
    ) -> None:
        super().__init__(content, media_type="text/event-stream")
        self._raw_request = raw_request
        self._context = context

    async def __call__(self, scope, receive, send) -> None:
        outcome: RequestOutcome | None = None
        failure: FailureDiagnostic | None = None
        original: BaseException | None = None
        tracked_receive = _DisconnectTrackingReceive(receive)
        try:
            await super().__call__(scope, tracked_receive, send)
            state = _stream_terminal_state(self._raw_request)
            if tracked_receive.disconnected and not state.finalized:
                outcome = "client_disconnected"
                _set_stream_response_outcome(self._raw_request, outcome)
        except (OSError, ClientDisconnect) as error:
            original = error
            outcome = "client_disconnected"
            _set_stream_response_outcome(self._raw_request, outcome)
            raise
        except asyncio.CancelledError as error:
            original = error
            outcome = await _cancelled_outcome(self._raw_request)
            _set_stream_response_outcome(self._raw_request, outcome)
            raise
        except Exception as error:
            original = error
            outcome = "failed"
            failure = _record_unexpected_stream_exception(
                self._raw_request,
                self._context,
                error,
                stage=FailureStage.CLIENT_TRANSLATION,
            )
            raise
        finally:
            try:
                await _close_stream_iterator(self.body_iterator, original)
            finally:
                state = _stream_terminal_state(self._raw_request)
                if outcome is not None and not state.finalized:
                    _invoke_request_finalizer(
                        self._raw_request, outcome, failure
                    )


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
            on_exception=lambda diagnostic: _mark_observed_stream_exception(
                raw_request, diagnostic
            ),
            on_provider_error=lambda error: _mark_observed_provider_error(
                raw_request, error
            ),
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
        return _LifecycleStreamingResponse(lifecycle, raw_request, context)
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
    except BaseException as error:
        original = error
        outcome = await _classify_stream_exception(
            raw_request, context, stream_state, error, outcome
        )
        raise
    finally:
        try:
            await _close_stream_iterator(iterator, original)
        finally:
            _finalize_stream_request(
                raw_request, sessions, observation, context, outcome
            )


async def _classify_stream_exception(
    raw_request: Request,
    context: RequestLogContext,
    state: "_StreamTerminalState",
    error: BaseException,
    outcome: RequestOutcome,
) -> RequestOutcome:
    if isinstance(error, (GeneratorExit, ClientDisconnect)):
        if state.has_error:
            return outcome
        return state.response_outcome or "client_disconnected"
    if isinstance(error, asyncio.CancelledError):
        if state.has_error:
            return outcome
        if state.response_outcome is not None:
            return state.response_outcome
        return await _cancelled_outcome(raw_request)
    if isinstance(error, Exception) and not state.exception_logged:
        _record_unexpected_stream_exception(
            raw_request,
            context,
            error,
            stage=FailureStage.CLIENT_TRANSLATION,
        )
    return outcome


_STREAM_TERMINAL_STATE = "stream_terminal_state"


@dataclass(slots=True)
class _StreamTerminalState:
    has_error: bool = False
    failure: FailureDiagnostic | None = None
    exception_logged: bool = False
    response_outcome: RequestOutcome | None = None
    finalized: bool = False


def _stream_terminal_state(raw_request: Request) -> _StreamTerminalState:
    state = getattr(raw_request.state, _STREAM_TERMINAL_STATE, None)
    if isinstance(state, _StreamTerminalState):
        return state
    state = _StreamTerminalState()
    setattr(raw_request.state, _STREAM_TERMINAL_STATE, state)
    return state


def _set_stream_response_outcome(
    raw_request: Request, outcome: RequestOutcome
) -> None:
    _stream_terminal_state(raw_request).response_outcome = outcome


def _mark_stream_error(raw_request: Request, error: StreamError) -> None:
    state = _stream_terminal_state(raw_request)
    state.has_error = True
    if state.failure is None:
        state.failure = stream_failure_diagnostic(error)


def _mark_observed_provider_error(
    raw_request: Request, error: ProviderError
) -> None:
    state = _stream_terminal_state(raw_request)
    state.has_error = True
    if state.failure is None:
        state.failure = provider_failure_diagnostic(error)
    state.exception_logged = True
    setattr(raw_request.state, FAILURE_LOGGED, True)


def _mark_observed_stream_exception(
    raw_request: Request, diagnostic: FailureDiagnostic
) -> None:
    state = _stream_terminal_state(raw_request)
    state.has_error = True
    if state.failure is None:
        state.failure = diagnostic
    state.exception_logged = True
    setattr(raw_request.state, FAILURE_LOGGED, True)


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


def _record_unexpected_stream_exception(
    raw_request: Request,
    context: RequestLogContext,
    error: Exception,
    *,
    stage: FailureStage,
) -> FailureDiagnostic:
    state = _stream_terminal_state(raw_request)
    if state.exception_logged and state.failure is not None:
        return state.failure
    diagnostic = unexpected_failure_diagnostic(error, stage=stage)
    state.has_error = True
    if state.failure is None:
        state.failure = diagnostic
    state.exception_logged = True
    _log_unexpected_error(raw_request, context, error, stage=stage)
    return state.failure


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
    if not sessions.performance_enabled:
        return None
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


def _invoke_request_finalizer(
    raw_request: Request,
    outcome: RequestOutcome,
    failure: FailureDiagnostic | None = None,
) -> None:
    finalizer = getattr(raw_request.state, REQUEST_FINALIZER, None)
    if not callable(finalizer):
        return
    try:
        finalizer(outcome, failure)
    except BaseException:
        pass


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
) -> bool:
    stream_state = _stream_terminal_state(raw_request)
    if stream_state.has_error:
        outcome = "failed"
        failure = stream_state.failure or failure
    finalized = _finalize_request(
        sessions, observation, context, outcome, failure
    )
    stream_state.finalized = stream_state.finalized or finalized
    return finalized


def _finalize_unexpected(
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
    error: Exception,
    *,
    stage: FailureStage = FailureStage.ROUTE,
) -> None:
    diagnostic = unexpected_failure_diagnostic(error, stage=stage)
    _finalize_request(sessions, observation, context, "failed", diagnostic)


def _finalize_request(
    sessions: SessionRegistry,
    observation: ObservationHandle,
    context: RequestLogContext,
    outcome: RequestOutcome,
    failure: FailureDiagnostic | None = None,
) -> bool:
    try:
        result = sessions.finish_with_status(observation, outcome, failure)
        if (
            result.performance is not None
            and sessions.performance_logging_enabled
        ):
            log_performance(result.performance, context)
        return result.finalized
    except BaseException:
        log_finalization_failure()
        return False


def _log_provider_error(
    raw_request: Request, context: RequestLogContext, error: ProviderError
) -> None:
    try:
        log_provider_failure(context, error)
    except BaseException:
        pass
    setattr(raw_request.state, FAILURE_LOGGED, True)


def _log_unexpected_error(
    raw_request: Request,
    context: RequestLogContext,
    error: Exception,
    *,
    stage: FailureStage = FailureStage.ROUTE,
) -> None:
    try:
        log_unexpected_failure(context, error, stage=stage)
    except BaseException:
        pass
    setattr(raw_request.state, FAILURE_LOGGED, True)


def _http_error(error: ProviderError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=str(error))
