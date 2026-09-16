import json
import pytest
from claude_code_proxy.config import Settings, load_model_mapping


def test_settings_reads_existing_environment_names(monkeypatch, tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text('{"mappings": {"sonnet": "big"}}')
    monkeypatch.setenv("PREFERRED_PROVIDER", "CODEX")
    monkeypatch.setenv("BIG_MODEL", "gpt-big")
    monkeypatch.setenv("SMALL_MODEL", "gpt-small")
    monkeypatch.setenv("MODEL_MAPPING_PATH", str(mapping_path))
    settings = Settings.from_environment()
    assert settings.preferred_provider == "codex"
    assert settings.big_model == "gpt-big"
    assert settings.small_model == "gpt-small"
    assert settings.model_mapping_path == mapping_path


def test_missing_mapping_file_uses_fresh_default_copy(tmp_path):
    first = load_model_mapping(tmp_path / "missing.json")
    first.pop("haiku")
    assert "haiku" in load_model_mapping(tmp_path / "missing.json")


def test_invalid_mapping_file_names_path(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"models": {}}))
    with pytest.raises(ValueError, match=str(path)):
        load_model_mapping(path)
