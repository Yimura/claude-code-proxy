"""Runtime configuration loading."""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from .reasoning import MappingEntry, parse_model_mappings

logger = logging.getLogger(__name__)
DEFAULT_MODEL_MAPPING = parse_model_mappings({
    "haiku": {"tier": "small", "effort": "medium"},
    "sonnet": {"tier": "big", "effort": "medium"},
    "opus": {"tier": "big", "effort": "high"},
    "fable": {"model": "gpt-daybreak-blue-latest", "effort": "high"},
})


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    vertex_project: str
    vertex_location: str
    use_vertex_auth: bool
    openai_base_url: str | None
    preferred_provider: str
    big_model: str
    small_model: str
    opencode_data_dir: Path
    model_mapping_path: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            vertex_project=os.environ.get("VERTEX_PROJECT", "unset"),
            vertex_location=os.environ.get("VERTEX_LOCATION", "unset"),
            use_vertex_auth=os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true",
            openai_base_url=os.environ.get("OPENAI_BASE_URL"),
            preferred_provider=os.environ.get("PREFERRED_PROVIDER", "openai").lower(),
            big_model=os.environ.get("BIG_MODEL", "gpt-5.6-sol"),
            small_model=os.environ.get("SMALL_MODEL", "gpt-5.6-terra"),
            opencode_data_dir=Path(os.environ.get("OPENCODE_DATA_DIR", "~/.local/share/opencode")).expanduser(),
            model_mapping_path=Path(os.environ.get("MODEL_MAPPING_PATH", "model_mapping.json")),
        )


def parse_model_mapping_config(data: object) -> dict[str, MappingEntry]:
    if not isinstance(data, dict) or "mappings" not in data:
        raise ValueError("configuration requires a top-level 'mappings' object")
    return parse_model_mappings(data["mappings"])


def load_model_mapping(path: Path) -> dict[str, MappingEntry]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Model mapping file not found at %s, using defaults", path)
        return DEFAULT_MODEL_MAPPING.copy()
    try:
        return parse_model_mapping_config(data)
    except ValueError as error:
        raise ValueError(f"Invalid model mapping configuration at {path}: {error}") from error
