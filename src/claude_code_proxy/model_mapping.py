"""Model-name mapping and provider-prefix resolution."""

from dataclasses import dataclass
from .reasoning import MappingEffort, MappingEntry

OPENAI_MODELS = frozenset({"gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra"})
GEMINI_MODELS = frozenset({"gemini-2.5-flash", "gemini-2.5-pro"})
PROVIDER_PREFIXES = ("anthropic/", "openai/", "gemini/")


@dataclass(frozen=True)
class ResolvedModel:
    original: str
    model: str
    mapped: bool
    effort: MappingEffort | None


class ModelResolver:
    def __init__(self, mappings: dict[str, MappingEntry], preferred_provider: str, big_model: str, small_model: str) -> None:
        self._mappings = mappings
        self._preferred_provider = preferred_provider
        self._big_model = big_model
        self._small_model = small_model

    def resolve(self, model: str) -> ResolvedModel:
        clean_name = self._strip_provider_prefix(model)
        if self._preferred_provider == "anthropic":
            return ResolvedModel(model, f"anthropic/{clean_name}", True, None)
        for pattern, entry in self._mappings.items():
            if pattern.lower() in clean_name.lower():
                return ResolvedModel(model, self._mapping_target(entry), True, entry.effort)
        if clean_name in GEMINI_MODELS and not model.startswith("gemini/"):
            return ResolvedModel(model, f"gemini/{clean_name}", True, None)
        if clean_name in OPENAI_MODELS and not model.startswith("openai/"):
            return ResolvedModel(model, f"openai/{clean_name}", True, None)
        return ResolvedModel(model, model, False, None)

    def _mapping_target(self, entry: MappingEntry) -> str:
        if entry.model is not None:
            return entry.model if entry.model.startswith(PROVIDER_PREFIXES) else f"openai/{entry.model}"
        target = self._small_model if entry.tier == "small" else self._big_model
        if self._preferred_provider == "google" and target in GEMINI_MODELS:
            return f"gemini/{target}"
        return f"openai/{target}"

    @staticmethod
    def _strip_provider_prefix(model: str) -> str:
        for prefix in PROVIDER_PREFIXES:
            if model.startswith(prefix):
                return model[len(prefix):]
        return model
