import json
from pathlib import Path

import pytest

from claude_code_proxy.config import ModelConfig, Settings, load_model_mapping
from claude_code_proxy.reasoning import MappingEntry


def test_settings_reads_runtime_environment(monkeypatch, tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text('{"tiers": {}, "mappings": {}}')
    monkeypatch.setenv("OPENAI_TRANSPORT", "CODEX")
    monkeypatch.setenv("MODEL_MAPPING_PATH", str(mapping_path))
    monkeypatch.setenv("PREFERRED_PROVIDER", "anthropic")
    monkeypatch.setenv("BIG_MODEL", "legacy-big")
    monkeypatch.setenv("SMALL_MODEL", "legacy-small")

    settings = Settings.from_environment()

    assert settings.openai_transport == "codex"
    assert settings.model_mapping_path == mapping_path
    assert not hasattr(settings, "preferred_provider")
    assert not hasattr(settings, "big_model")
    assert not hasattr(settings, "small_model")


def test_settings_rejects_invalid_openai_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_TRANSPORT", "invalid")

    with pytest.raises(ValueError, match="OPENAI_TRANSPORT"):
        Settings.from_environment()


def test_loads_tiers_and_mappings(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "tiers": {"small": "openai/gpt-small", "big": "gemini/gemini-big"},
        "mappings": {
            "haiku": {"tier": "small", "effort": "medium"},
            "opus": {"model": "anthropic/claude-opus-5", "effort": "high"},
        },
    }))

    assert load_model_mapping(path) == ModelConfig(
        tiers={"small": "openai/gpt-small", "big": "gemini/gemini-big"},
        mappings={
            "haiku": MappingEntry(tier="small", effort="medium"),
            "opus": MappingEntry(model="anthropic/claude-opus-5", effort="high"),
        },
    )


def test_missing_mapping_file_uses_fresh_default_config(tmp_path):
    first = load_model_mapping(tmp_path / "missing.json")
    first.tiers.pop("small")
    first.mappings.pop("haiku")

    second = load_model_mapping(tmp_path / "missing.json")

    assert "small" in second.tiers
    assert "haiku" in second.mappings


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"mappings": {}}, "tiers"),
        ({"tiers": {}, "models": {}}, "mappings"),
        ({"tiers": {"big": ""}, "mappings": {}}, "big"),
        ({"tiers": {}, "mappings": {"sonnet": {"tier": "big"}}}, "unknown tier"),
    ],
)
def test_invalid_mapping_file_names_path_and_problem(tmp_path, data, message):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=str(path)) as error:
        load_model_mapping(path)

    assert message in str(error.value)
