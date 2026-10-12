"""Provider-neutral request orchestration."""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace

from .domain.models import (
    CompletionRequest,
    CompletionResponse,
    Message,
    StreamComplete,
    StreamError,
    StreamEvent,
    TextBlock,
    ToolInputDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseStart,
)
from .model_mapping import ModelResolver
from .performance import (
    RequestTelemetry,
    SafeProviderTelemetry,
    notify_telemetry,
)
from .prompt_identity import reconcile_message_identities, reconcile_system_identity
from .providers.base import Provider, ProviderError, protocol_error, stream_error_from_exception
from .reasoning import resolve_reasoning_policy

logger = logging.getLogger(__name__)

WEB_SEARCH_TOOL = "WebSearch"
# One upstream request plus up to four search-and-refeed rounds.
_MAX_SEARCH_ROUNDS = 5
_MAX_FETCHED_PAGES = 2
_RESULT_URL_LINE = "\n  "


class ProxyService:
    def __init__(self, resolver: ModelResolver, openai_transport: str, litellm_provider: Provider, codex_provider: Provider, web_search=None) -> None:
        self._resolver = resolver
        self._openai_transport = openai_transport
        self._litellm_provider = litellm_provider
        self._codex_provider = codex_provider
        self._web_search = web_search

    def prepare(self, request: CompletionRequest) -> CompletionRequest:
        resolved = self._resolver.resolve(request.model)
        system = reconcile_system_identity(
            request.system,
            request.original_model,
            resolved.model,
            resolved.mapped,
        )
        messages = reconcile_message_identities(
            request.messages,
            request.original_model,
            resolved.model,
            resolved.mapped,
        )
        return replace(
            request,
            model=resolved.model,
            response_model=resolved.response_model,
            context_window=resolved.context_window,
            max_output_tokens=resolved.max_output_tokens,
            backend=resolved.backend,
            messages=messages,
            system=system,
            reasoning=resolve_reasoning_policy(
                output_config=request.output_config,
                thinking=request.thinking,
                mapping_effort=resolved.effort,
            ),
        )

    def provider_for(self, request: CompletionRequest) -> Provider:
        if self._openai_transport == "codex" and request.model.startswith("openai/"):
            return self._codex_provider
        return self._litellm_provider

    async def complete(
        self,
        request: CompletionRequest,
        telemetry: RequestTelemetry | None = None,
    ) -> CompletionResponse:
        return await self.complete_prepared(self.prepare(request), telemetry)

    async def complete_prepared(
        self,
        request: CompletionRequest,
        telemetry: RequestTelemetry | None = None,
    ) -> CompletionResponse:
        if not _search_active(self, request):
            return await self._complete_once(request, telemetry)
        current = request
        for _ in range(_MAX_SEARCH_ROUNDS - 1):
            response = await self._complete_once(current, telemetry)
            followup = await _search_followup(self, current, response)
            if followup is None:
                return response
            current = followup
        return await self._complete_once(current, telemetry)

    async def _complete_once(
        self,
        request: CompletionRequest,
        telemetry: RequestTelemetry | None = None,
    ) -> CompletionResponse:
        provider = self.provider_for(request)
        notify_telemetry(telemetry, "upstream_started")
        try:
            if telemetry is None:
                response = await provider.complete(request)
            else:
                response = await provider.complete(
                    request, telemetry=SafeProviderTelemetry(telemetry)
                )
            notify_telemetry(telemetry, "response", response)
            return response
        finally:
            notify_telemetry(telemetry, "upstream_finished")

    def stream(
        self,
        request: CompletionRequest,
        telemetry: RequestTelemetry | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self.stream_prepared(self.prepare(request), telemetry)

    def stream_prepared(
        self,
        request: CompletionRequest,
        telemetry: RequestTelemetry | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if not _search_active(self, request):
            provider = self.provider_for(request)
            provider_events = _provider_stream(provider, request, telemetry)
            return _observed_stream(provider_events, telemetry)
        return _search_intercept_stream(self, request, telemetry)

    async def count_tokens(
        self,
        request: CompletionRequest,
        telemetry: RequestTelemetry | None = None,
    ) -> int:
        return await self.count_tokens_prepared(self.prepare(request), telemetry)

    async def count_tokens_prepared(
        self,
        request: CompletionRequest,
        telemetry: RequestTelemetry | None = None,
    ) -> int:
        provider = self.provider_for(request)
        notify_telemetry(telemetry, "upstream_started")
        try:
            if telemetry is None:
                return await provider.count_tokens(request)
            return await provider.count_tokens(
                request, telemetry=SafeProviderTelemetry(telemetry)
            )
        finally:
            notify_telemetry(telemetry, "upstream_finished")


def _search_active(service: "ProxyService", request: CompletionRequest) -> bool:
    """Whether the WebSearch intercept loop runs for this request.

    Runs only when a search client is configured, the client declared a
    WebSearch tool, and the backend cannot run search server-side (the
    `zai` backend swaps the tool for its own built-in one in the provider).
    """
    return (
        service._web_search is not None
        and request.backend != "zai"
        and any(tool.name == WEB_SEARCH_TOOL for tool in request.tools)
    )


def _web_search_calls(response: CompletionResponse) -> list[ToolUseBlock]:
    return [
        block
        for block in response.content
        if isinstance(block, ToolUseBlock) and block.name == WEB_SEARCH_TOOL
    ]


async def _search_followup(
    service: "ProxyService",
    request: CompletionRequest,
    response: CompletionResponse,
) -> CompletionRequest | None:
    """Build the next request when the response only calls WebSearch.

    Runs each search, fetches the top result pages, and appends an
    assistant message with the calls plus a user message with the results.
    Returns None when the response makes no WebSearch call, mixes in
    another tool, or the intercept is off.
    """
    calls = _web_search_calls(response)
    if not calls or not _search_active(service, request):
        return None
    if len(calls) != len([
        block for block in response.content if isinstance(block, ToolUseBlock)
    ]):
        return None
    messages = [
        *request.messages,
        Message("assistant", tuple(response.content)),
        Message("user", tuple(await _run_searches(service, calls))),
    ]
    return replace(request, messages=tuple(messages))


async def _run_searches(service: "ProxyService", calls: list[ToolUseBlock]):
    results = []
    for call in calls:
        query = str(call.input.get("query", "")) if isinstance(call.input, dict) else ""
        try:
            text = (
                await service._web_search.search(query)
                if query
                else "Web search failed: empty query"
            )
        except Exception as error:  # noqa: BLE001 - degrade to a text result
            logger.warning("web search execution failed: %s", error)
            text = f"Web search failed: {error}"
        text = await _append_page_excerpts(service, text)
        results.append(ToolResultBlock(call.id, text))
    return results


async def _append_page_excerpts(service: "ProxyService", search_text: str) -> str:
    """Fetch the top result pages listed in the search text, append bodies."""
    urls = [
        line.strip()
        for line in search_text.splitlines()
        if line.strip().startswith("http")
    ][:_MAX_FETCHED_PAGES]
    excerpts = []
    for url in urls:
        body = await service._web_search.fetch_page(url)
        if body:
            excerpts.append(f"Excerpt from {url}:\n{body}")
    if not excerpts:
        return search_text
    return search_text + "\n\n" + "\n\n".join(excerpts)


async def _search_intercept_stream(
    service: "ProxyService",
    request: CompletionRequest,
    telemetry: RequestTelemetry | None,
) -> AsyncIterator[StreamEvent]:
    """Stream with the WebSearch intercept loop applied per round.

    Intermediate rounds are buffered and discarded; only the final round's
    events reach the client, so the client never sees the intercepted calls.
    """
    current = request
    for _ in range(_MAX_SEARCH_ROUNDS - 1):
        collected: list[StreamEvent] = []
        terminal = None
        async for event in _observed_stream(
            _provider_stream(service.provider_for(current), current, telemetry),
            telemetry,
        ):
            if isinstance(event, (StreamComplete, StreamError)):
                terminal = event
            else:
                collected.append(event)
        if terminal is None or isinstance(terminal, StreamError):
            if terminal is not None:
                yield terminal
            return
        followup = await _search_followup(
            service, current, _response_from_events(current, collected, terminal)
        )
        if followup is None:
            for event in collected:
                yield event
            yield terminal
            return
        current = followup
    async for event in _observed_stream(
        _provider_stream(service.provider_for(current), current, telemetry),
        telemetry,
    ):
        yield event


def _response_from_events(
    request: CompletionRequest,
    events: list[StreamEvent],
    terminal: StreamComplete,
) -> CompletionResponse:
    """Rebuild a CompletionResponse from one buffered stream round."""
    blocks: list = []
    tool_ids: dict[str, tuple[str, str]] = {}
    tool_json: dict[str, list[str]] = {}
    for event in events:
        if isinstance(event, TextBlock):
            blocks.append(event)
        elif isinstance(event, ToolUseStart):
            tool_ids[event.slot] = (event.id, event.name)
            tool_json.setdefault(event.slot, [])
        elif isinstance(event, ToolInputDelta) and event.slot in tool_ids:
            tool_json[event.slot].append(event.partial_json)
    for slot, (call_id, name) in tool_ids.items():
        try:
            arguments = json.loads("".join(tool_json[slot]) or "{}")
        except ValueError:
            arguments = {}
        blocks.append(ToolUseBlock(call_id, name, arguments))
    return CompletionResponse(
        f"msg_{uuid.uuid4().hex}",
        request.response_model,
        tuple(blocks),
        terminal.stop_reason,
        terminal.usage,
    )


async def _provider_stream(
    provider: Provider,
    request: CompletionRequest,
    telemetry: RequestTelemetry | None,
) -> AsyncIterator[StreamEvent]:
    try:
        if telemetry is None:
            events = provider.stream(request)
        else:
            events = provider.stream(
                request, telemetry=SafeProviderTelemetry(telemetry)
            )
    except Exception as error:
        yield stream_error_from_exception(error, provider=provider.name)
        return

    iterator = aiter(_validated_stream(events, provider.name))
    try:
        async for event in iterator:
            yield event
    finally:
        await _close_iterator(iterator)


async def _validated_stream(
    events: AsyncIterator[StreamEvent], provider: str
) -> AsyncIterator[StreamEvent]:
    iterator = aiter(events)
    try:
        async for event in iterator:
            yield event
            if isinstance(event, (StreamComplete, StreamError)):
                return
    except ProviderError as error:
        yield stream_error_from_exception(error, provider=provider)
        return
    except Exception as error:
        yield stream_error_from_exception(error, provider=provider)
        return
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()

    yield protocol_error("missing_terminal_event", provider=provider)


async def _observed_stream(
    events: AsyncIterator[StreamEvent],
    telemetry: RequestTelemetry | None,
) -> AsyncIterator[StreamEvent]:
    notify_telemetry(telemetry, "upstream_started")
    iterator = None
    try:
        iterator = aiter(events)
        async for event in iterator:
            notify_telemetry(telemetry, "stream_event", event)
            yield event
    finally:
        try:
            if iterator is not None:
                await _close_iterator(iterator)
        finally:
            notify_telemetry(telemetry, "upstream_finished")


async def _close_iterator(iterator: object) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()
