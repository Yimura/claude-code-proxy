"""Provider-neutral request orchestration."""

from dataclasses import replace
from .domain.models import CompletionRequest, CompletionResponse
from .model_mapping import ModelResolver
from .providers.base import Provider
from .reasoning import resolve_reasoning_policy


class ProxyService:
    def __init__(self, resolver: ModelResolver, openai_transport: str, litellm_provider: Provider, codex_provider: Provider) -> None:
        self._resolver = resolver
        self._openai_transport = openai_transport
        self._litellm_provider = litellm_provider
        self._codex_provider = codex_provider

    def prepare(self, request: CompletionRequest) -> CompletionRequest:
        resolved = self._resolver.resolve(request.model)
        return replace(request, model=resolved.model, reasoning=resolve_reasoning_policy(output_config=request.output_config, thinking=request.thinking, mapping_effort=resolved.effort))

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
        return self.provider_for(request).stream(request)

    async def count_tokens(self, request: CompletionRequest) -> int:
        return await self.count_tokens_prepared(self.prepare(request))

    async def count_tokens_prepared(self, request: CompletionRequest) -> int:
        return await self.provider_for(request).count_tokens(request)
