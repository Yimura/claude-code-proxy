"""Provider-neutral request orchestration."""

from collections.abc import AsyncIterator
from dataclasses import replace

from .domain.models import (
    CompletionRequest,
    CompletionResponse,
    StreamComplete,
    StreamError,
    StreamEvent,
)
from .model_mapping import ModelResolver
from .performance import RequestTelemetry, notify_telemetry
from .prompt_identity import reconcile_message_identities, reconcile_system_identity
from .providers.base import Provider, ProviderError, protocol_error, stream_error_from_exception
from .reasoning import resolve_reasoning_policy


class ProxyService:
    def __init__(self, resolver: ModelResolver, openai_transport: str, litellm_provider: Provider, codex_provider: Provider) -> None:
        self._resolver = resolver
        self._openai_transport = openai_transport
        self._litellm_provider = litellm_provider
        self._codex_provider = codex_provider

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
        provider = self.provider_for(request)
        notify_telemetry(telemetry, "upstream_started")
        try:
            response = await provider.complete(request, telemetry=telemetry)
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
        provider = self.provider_for(request)
        provider_events = _provider_stream(provider, request, telemetry)
        return _observed_stream(provider_events, telemetry)

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
            return await provider.count_tokens(request, telemetry=telemetry)
        finally:
            notify_telemetry(telemetry, "upstream_finished")


async def _provider_stream(
    provider: Provider,
    request: CompletionRequest,
    telemetry: RequestTelemetry | None,
) -> AsyncIterator[StreamEvent]:
    try:
        events = provider.stream(request, telemetry=telemetry)
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
