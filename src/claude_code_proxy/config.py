"""Runtime configuration loading."""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from .reasoning import MappingEntry, parse_model_mappings

logger = logging.getLogger(__name__)
OpenAITransport = Literal["litellm", "codex"]


@dataclass(frozen=True)
class ModelConfig:
    """Validated model tiers and Claude-name mapping rules."""

    tiers: dict[str, str]
    mappings: dict[str, MappingEntry]


DEFAULT_MODEL_CONFIG = ModelConfig(
    tiers={
        "small": "openai/gpt-5.6-terra",
        "big": "openai/gpt-5.6-sol",
    },
    mappings=parse_model_mappings({
        "haiku": {"tier": "small", "effort": "medium"},
        "sonnet": {"tier": "big", "effort": "medium"},
        "opus": {"tier": "big", "effort": "high"},
        "fable": {"model": "openai/gpt-daybreak-blue-latest", "effort": "high"},
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
    if "tiers" not in data or not isinstance(data["tiers"], dict):
        raise ValueError("configuration requires a top-level 'tiers' object")
    if "mappings" not in data or not isinstance(data["mappings"], dict):
        raise ValueError("configuration requires a top-level 'mappings' object")

    tiers: dict[str, str] = {}
    for name, target in data["tiers"].items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid tier name {name!r}")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"invalid target for tier {name!r}: {target!r}")
        tiers[name] = target

    mappings = parse_model_mappings(data["mappings"])
    for pattern, entry in mappings.items():
        if entry.tier is not None and entry.tier not in tiers:
            raise ValueError(
                f"model mapping for pattern {pattern!r} references unknown tier "
                f"{entry.tier!r}"
            )
    return ModelConfig(tiers=tiers, mappings=mappings)


def load_model_mapping(path: Path) -> ModelConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Model mapping file not found at %s, using defaults", path)
        return ModelConfig(
            tiers=DEFAULT_MODEL_CONFIG.tiers.copy(),
            mappings=DEFAULT_MODEL_CONFIG.mappings.copy(),
        )
    try:
        return parse_model_mapping_config(data)
    except ValueError as error:
        raise ValueError(f"Invalid model mapping configuration at {path}: {error}") from error
