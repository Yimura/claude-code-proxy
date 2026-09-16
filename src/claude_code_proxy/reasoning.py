"""Provider-neutral reasoning request normalization."""

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

RequestEffort = Literal["low", "medium", "high", "max"]
ProviderEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
MappingEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
Tier = Literal["small", "big"]
ThinkingType = Literal["enabled", "adaptive", "disabled"]


class MappingEntry(BaseModel):
    """Validated target and default effort for a model-name pattern."""

    tier: Optional[Tier] = None
    model: Optional[str] = None
    effort: Optional[MappingEffort] = None

    @model_validator(mode="after")
    def validate_selector(self) -> "MappingEntry":
        if (self.tier is None) == (self.model is None):
            raise ValueError("exactly one of tier or model is required")
        return self


def parse_model_mappings(raw: object) -> dict[str, MappingEntry]:
    """Parse legacy tier strings and structured model mapping entries."""
    if not isinstance(raw, dict):
        raise ValueError(f"model mappings must be an object, got {raw!r}")

    mappings: dict[str, MappingEntry] = {}
    for pattern, value in raw.items():
        try:
            mappings[pattern] = (
                MappingEntry(tier=value)
                if isinstance(value, str)
                else MappingEntry.model_validate(value)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid model mapping for pattern {pattern!r}: {value!r}"
            ) from error
    return mappings


class ThinkingConfig(BaseModel):
    """Anthropic thinking configuration, including the legacy boolean form."""

    type: Optional[ThinkingType] = None
    budget_tokens: Optional[int] = Field(default=None, strict=True, ge=1024)
    enabled: Optional[bool] = None

    @model_validator(mode="after")
    def validate_configuration(self) -> "ThinkingConfig":
        if (self.type is None) == (self.enabled is None):
            raise ValueError("exactly one of type or enabled is required")

        if self.type == "enabled":
            if self.budget_tokens is None:
                raise ValueError("budget_tokens is required when type is enabled")
        elif self.budget_tokens is not None:
            raise ValueError("budget_tokens is only valid when type is enabled")

        return self

    def is_enabled(self) -> bool:
        """Return whether this configuration enables provider reasoning."""
        if self.enabled is not None:
            return self.enabled
        return self.type in ("enabled", "adaptive")


class OutputConfig(BaseModel):
    """Output options relevant to reasoning, preserving provider extensions."""

    model_config = ConfigDict(extra="allow")

    effort: Optional[RequestEffort] = None


@dataclass(frozen=True)
class ReasoningPolicy:
    """Normalized reasoning settings to pass to a provider."""

    enabled: Optional[bool]
    effort: Optional[ProviderEffort]


def resolve_reasoning_policy(
    output_config: Optional[OutputConfig] = None,
    thinking: Optional[ThinkingConfig] = None,
    mapping_effort: Optional[MappingEffort] = None,
) -> ReasoningPolicy:
    """Resolve request and model-mapping inputs into a provider policy."""
    if output_config is not None and output_config.effort is not None:
        effort: ProviderEffort = (
            "high" if output_config.effort == "max" else output_config.effort
        )
        return ReasoningPolicy(enabled=True, effort=effort)

    if thinking is not None:
        if not thinking.is_enabled():
            return ReasoningPolicy(enabled=False, effort=None)
        if mapping_effort is None or mapping_effort == "none":
            return ReasoningPolicy(enabled=True, effort="medium")
        return ReasoningPolicy(enabled=True, effort=mapping_effort)

    if mapping_effort == "none":
        return ReasoningPolicy(enabled=False, effort=None)
    if mapping_effort is not None:
        return ReasoningPolicy(enabled=True, effort=mapping_effort)

    return ReasoningPolicy(enabled=None, effort=None)
