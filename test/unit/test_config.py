import json
from pathlib import Path

import pytest

from claude_code_proxy.config import (
    ModelConfig,
    ModelDefinition,
    Settings,
    load_model_mapping,
)
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
        "models": {
            "small-model": {"target": "openai/gpt-small", "context_window": None},
            "big-model": {"target": "gemini/gemini-big", "context_window": 1_000_000},
        },
        "tiers": {"small": "small-model", "big": "big-model"},
        "mappings": {
            "haiku": {"tier": "small", "effort": "medium"},
            "opus": {"model": "big-model", "effort": "high"},
        },
    }))

    assert load_model_mapping(path) == ModelConfig(
        models={
            "small-model": ModelDefinition(
                target="openai/gpt-small", context_window=None
            ),
            "big-model": ModelDefinition(
                target="gemini/gemini-big", context_window=1_000_000
            ),
        },
        tiers={"small": "small-model", "big": "big-model"},
        mappings={
            "haiku": MappingEntry(tier="small", effort="medium"),
            "opus": MappingEntry(model="big-model", effort="high"),
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
        ({"tiers": {}, "mappings": {}}, "models"),
        ({"models": {}, "mappings": {}}, "tiers"),
        ({"models": {}, "tiers": {}}, "mappings"),
        ({"models": {}, "tiers": {"big": ""}, "mappings": {}}, "big"),
        ({"models": {}, "tiers": {}, "mappings": {"sonnet": {"tier": "big"}}}, "unknown tier"),
    ],
)
def test_invalid_mapping_file_names_path_and_problem(tmp_path, data, message):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=str(path)) as error:
        load_model_mapping(path)

    assert message in str(error.value)


def test_loads_required_model_definitions(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "models": {
            "terra": {
                "target": "openai/gpt-5.6-terra",
                "context_window": 1_000_000,
            },
            "sol": {
                "target": "openai/gpt-5.6-sol",
                "context_window": None,
            },
        },
        "tiers": {"small": "terra"},
        "mappings": {
            "haiku": {"tier": "small", "effort": "medium"},
            "opus": {"model": "sol", "effort": "high"},
        },
    }))

    config = load_model_mapping(path)

    assert config.models["terra"].target == "openai/gpt-5.6-terra"
    assert config.models["terra"].context_window == 1_000_000
    assert config.models["sol"].context_window is None
    assert config.tiers == {"small": "terra"}
    assert config.mappings["opus"] == MappingEntry(model="sol", effort="high")


def test_mapping_file_requires_models_object(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"tiers": {}, "mappings": {}}))

    with pytest.raises(ValueError, match="models"):
        load_model_mapping(path)


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        ({"target": "openai/model"}, "context_window"),
        ({"target": "openai/model", "context_window": 0}, "context_window"),
        ({"target": "openai/model", "context_window": "1000000"}, "context_window"),
        ({"target": "openai/model", "context_window": True}, "context_window"),
        ({"target": "   ", "context_window": None}, "target"),
        ({"target": " openai/model", "context_window": None}, "target"),
        ({"target": "openai/model ", "context_window": None}, "target"),
    ],
)
def test_rejects_invalid_model_definition(tmp_path, definition, message):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "models": {"broken": definition},
        "tiers": {},
        "mappings": {},
    }))

    with pytest.raises(ValueError, match=message):
        load_model_mapping(path)


def test_rejects_tier_referencing_unknown_model_definition(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "models": {},
        "tiers": {"big": "missing"},
        "mappings": {},
    }))

    with pytest.raises(ValueError, match="unknown model"):
        load_model_mapping(path)


def test_rejects_direct_mapping_referencing_unknown_model_definition(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "models": {},
        "tiers": {},
        "mappings": {"opus": {"model": "missing"}},
    }))

    with pytest.raises(ValueError, match="unknown model"):
        load_model_mapping(path)


def test_rejects_legacy_raw_targets(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "models": {
            "sol": {
                "target": "openai/gpt-5.6-sol",
                "context_window": 1_000_000,
            }
        },
        "tiers": {"big": "openai/gpt-5.6-sol"},
        "mappings": {"opus": {"model": "openai/gpt-5.6-sol"}},
    }))

    with pytest.raises(ValueError, match="unknown model"):
        load_model_mapping(path)
