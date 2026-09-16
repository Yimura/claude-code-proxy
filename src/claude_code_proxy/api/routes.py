"""Anthropic-compatible HTTP routes."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from ..logging import log_request_summary
from ..providers.base import ProviderError
from ..service import ProxyService
from .schemas import MessagesRequest, TokenCountRequest, TokenCountResponse
from .translation import normalize_request, serialize_stream, to_api_response


def build_router(service: ProxyService) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/messages")
    async def create_message(request: MessagesRequest, raw_request: Request):
        normalized = normalize_request(request)
        prepared = service.prepare(normalized)
        _log_request(raw_request, normalized, prepared.model)
        if request.stream:
            return StreamingResponse(
                serialize_stream(prepared, service.stream_prepared(prepared)),
                media_type="text/event-stream",
            )
        try:
            return to_api_response(await service.complete_prepared(prepared))
        except ProviderError as error:
            raise _http_error(error) from error

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(request: TokenCountRequest, raw_request: Request):
        normalized = normalize_request(_as_messages_request(request))
        prepared = service.prepare(normalized)
        _log_request(raw_request, normalized, prepared.model)
        try:
            count = await service.count_tokens_prepared(prepared)
            return TokenCountResponse(input_tokens=count)
        except ProviderError as error:
            raise _http_error(error) from error

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


def _log_request(raw_request, normalized, upstream_model):
    log_request_summary(
        "POST",
        raw_request.url.path,
        normalized.original_model.rsplit("/", 1)[-1],
        upstream_model,
        len(normalized.messages),
        len(normalized.tools),
        200,
    )


def _http_error(error: ProviderError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=str(error))
