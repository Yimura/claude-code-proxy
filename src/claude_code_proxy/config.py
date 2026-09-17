"""Runtime configuration loading."""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .reasoning import MappingEntry, parse_model_mappings

logger = logging.getLogger(__name__)
OpenAITransport = Literal["litellm", "codex"]


class ModelDefinition(BaseModel):
    """Configured upstream model and its known context capability."""

    model_config = ConfigDict(frozen=True)

    target: str
    context_window: int | None = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def validate_target(self) -> "ModelDefinition":
        stripped = self.target.strip()
        if not stripped:
            raise ValueError("target must not be empty")
        if stripped != self.target:
            raise ValueError("target must not contain surrounding whitespace")
        return self


@dataclass(frozen=True)
class ModelConfig:
    """Validated upstream models, tiers, and Claude-name mapping rules."""

    models: dict[str, ModelDefinition]
    tiers: dict[str, str]
    mappings: dict[str, MappingEntry]


DEFAULT_MODEL_CONFIG = ModelConfig(
    models={
        "terra": ModelDefinition(
            target="openai/gpt-5.6-terra", context_window=1_000_000
        ),
        "sol": ModelDefinition(
            target="openai/gpt-5.6-sol", context_window=1_000_000
        ),
    },
    tiers={"small": "terra", "big": "sol"},
    mappings=parse_model_mappings({
        "haiku": {"tier": "small", "effort": "medium"},
        "sonnet": {"tier": "big", "effort": "medium"},
        "opus": {"tier": "big", "effort": "high"},
        "fable": {"model": "sol", "effort": "xhigh"},
    }),
)


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    vertex_project: str
    vertex_location: str
    use_vertex_auth: bool
    openai_base_url: str | None
    openai_transport: OpenAITransport
    opencode_data_dir: Path
    model_mapping_path: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        transport = os.environ.get("OPENAI_TRANSPORT", "litellm").lower()
        if transport not in ("litellm", "codex"):
            raise ValueError(
                "OPENAI_TRANSPORT must be either 'litellm' or 'codex', "
                f"got {transport!r}"
            )
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            vertex_project=os.environ.get("VERTEX_PROJECT", "unset"),
            vertex_location=os.environ.get("VERTEX_LOCATION", "unset"),
            use_vertex_auth=os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true",
            openai_base_url=os.environ.get("OPENAI_BASE_URL"),
            openai_transport=transport,
            opencode_data_dir=Path(os.environ.get("OPENCODE_DATA_DIR", "~/.local/share/opencode")).expanduser(),
            model_mapping_path=Path(os.environ.get("MODEL_MAPPING_PATH", "model_mapping.json")),
        )


def parse_model_mapping_config(data: object) -> ModelConfig:
    if not isinstance(data, dict):
        raise ValueError("configuration must be an object")

    models = _parse_model_definitions(_required_object(data, "models"))
    tiers = _parse_tiers(_required_object(data, "tiers"), models)
    mappings = parse_model_mappings(_required_object(data, "mappings"))
    _validate_mapping_references(mappings, tiers, models)
    return ModelConfig(models=models, tiers=tiers, mappings=mappings)


def _required_object(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration requires a top-level {key!r} object")
    return value


def _parse_model_definitions(raw: dict) -> dict[str, ModelDefinition]:
    models: dict[str, ModelDefinition] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid model definition name {name!r}")
        try:
            models[name] = ModelDefinition.model_validate(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid model definition {name!r}: {value!r}: {error}"
            ) from error
    return models


def _parse_tiers(
    raw: dict, models: dict[str, ModelDefinition]
) -> dict[str, str]:
    tiers: dict[str, str] = {}
    for name, model_name in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid tier name {name!r}")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"invalid model definition for tier {name!r}: {model_name!r}")
        if model_name not in models:
            raise ValueError(
                f"tier {name!r} references unknown model {model_name!r}"
            )
        tiers[name] = model_name
    return tiers


def _validate_mapping_references(
    mappings: dict[str, MappingEntry],
    tiers: dict[str, str],
    models: dict[str, ModelDefinition],
) -> None:
    for pattern, entry in mappings.items():
        if entry.tier is not None and entry.tier not in tiers:
            raise ValueError(
                f"model mapping for pattern {pattern!r} references unknown tier "
                f"{entry.tier!r}"
            )
        if entry.model is not None and entry.model not in models:
            raise ValueError(
                f"model mapping for pattern {pattern!r} references unknown model "
                f"{entry.model!r}"
            )


def load_model_mapping(path: Path) -> ModelConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Model mapping file not found at %s, using defaults", path)
        return ModelConfig(
            models=DEFAULT_MODEL_CONFIG.models.copy(),
            tiers=DEFAULT_MODEL_CONFIG.tiers.copy(),
            mappings=DEFAULT_MODEL_CONFIG.mappings.copy(),
        )
    try:
        return parse_model_mapping_config(data)
    except ValueError as error:
        raise ValueError(f"Invalid model mapping configuration at {path}: {error}") from error
