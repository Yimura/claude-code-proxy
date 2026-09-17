"""Model-name mapping and provider-prefix resolution."""

from dataclasses import dataclass

from .config import ModelConfig, ModelDefinition
from .reasoning import MappingEffort, MappingEntry

OPENAI_MODELS = frozenset({"gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra"})
GEMINI_MODELS = frozenset({"gemini-2.5-flash", "gemini-2.5-pro"})
PROVIDER_PREFIXES = ("anthropic/", "openai/", "gemini/")
LONG_CONTEXT_SUFFIX = "[1m]"
LONG_CONTEXT_TOKENS = 1_000_000


@dataclass(frozen=True)
class ResolvedModel:
    original: str
    model: str
    response_model: str
    mapped: bool
    effort: MappingEffort | None


class ModelResolver:
    def __init__(self, config: ModelConfig) -> None:
        self._models = config.models
        self._tiers = config.tiers
        self._mappings = config.mappings

    def resolve(self, model: str) -> ResolvedModel:
        clean_name = self._strip_provider_prefix(model)
        for pattern, entry in self._mappings.items():
            if pattern.lower() in clean_name.lower():
                return self._resolve_mapping(model, entry)

        if clean_name in GEMINI_MODELS and not model.startswith("gemini/"):
            resolved = f"gemini/{clean_name}"
            return ResolvedModel(model, resolved, resolved, True, None)
        if clean_name in OPENAI_MODELS and not model.startswith("openai/"):
            resolved = f"openai/{clean_name}"
            return ResolvedModel(model, resolved, resolved, True, None)
        return ResolvedModel(model, model, model, False, None)

    def _resolve_mapping(
        self, original: str, entry: MappingEntry
    ) -> ResolvedModel:
        definition = self._models[self._model_name(entry)]
        upstream = self._upstream_target(definition)
        response_model = self._response_model(original, upstream, definition)
        return ResolvedModel(
            original,
            upstream,
            response_model,
            True,
            entry.effort,
        )

    def _model_name(self, entry: MappingEntry) -> str:
        if entry.model is not None:
            return entry.model
        if entry.tier is None:
            raise ValueError("mapping entry has no model selector")
        return self._tiers[entry.tier]

    @staticmethod
    def _upstream_target(definition: ModelDefinition) -> str:
        if definition.target.startswith(PROVIDER_PREFIXES):
            return definition.target
        return f"openai/{definition.target}"

    @staticmethod
    def _response_model(
        original: str, upstream: str, definition: ModelDefinition
    ) -> str:
        if definition.context_window is None:
            return upstream
        base = original.removesuffix(LONG_CONTEXT_SUFFIX)
        if definition.context_window >= LONG_CONTEXT_TOKENS:
            return f"{base}{LONG_CONTEXT_SUFFIX}"
        return base

    @staticmethod
    def _strip_provider_prefix(model: str) -> str:
        for prefix in PROVIDER_PREFIXES:
            if model.startswith(prefix):
                return model[len(prefix):]
        return model
