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
from .prompt_identity import reconcile_system_identity
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
        return replace(
            request,
            model=resolved.model,
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

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await self.complete_prepared(self.prepare(request))

    async def complete_prepared(self, request: CompletionRequest) -> CompletionResponse:
        return await self.provider_for(request).complete(request)

    def stream(self, request: CompletionRequest):
        return self.stream_prepared(self.prepare(request))

    def stream_prepared(self, request: CompletionRequest):
        provider = self.provider_for(request)
        return _validated_stream(provider.stream(request), provider.name)

    async def count_tokens(self, request: CompletionRequest) -> int:
        return await self.count_tokens_prepared(self.prepare(request))

    async def count_tokens_prepared(self, request: CompletionRequest) -> int:
        return await self.provider_for(request).count_tokens(request)


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
        yield stream_error_from_exception(
            error, provider=provider, expose_message=True
        )
        return
    except Exception as error:
        yield stream_error_from_exception(error, provider=provider)
        return
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()

    yield protocol_error(
        "provider stream ended without terminal outcome", provider=provider
    )
