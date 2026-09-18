import json
import logging
from pathlib import Path

import pytest

from claude_code_proxy import config as config_module
from claude_code_proxy.config import (
    ModelConfig,
    ModelDefinition,
    Settings,
    load_model_mapping,
)
from claude_code_proxy.reasoning import MappingEntry


@pytest.fixture(autouse=True)
def isolate_dotenv_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)


def test_settings_loads_only_current_directory_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_load(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(config_module, "load_dotenv", record_load)

    Settings.from_environment()

    assert calls == [
        ((), {"dotenv_path": tmp_path / ".env", "override": False})
    ]


@pytest.mark.parametrize(
    ("dotenv_contents", "expected_port"),
    [(None, 8082), ("PROXY_PORT=9000\n", 9000)],
)
def test_settings_does_not_supplement_missing_or_partial_dotenv_from_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dotenv_contents: str | None,
    expected_port: int,
) -> None:
    ancestor = tmp_path / "ancestor"
    working_directory = ancestor / "working"
    working_directory.mkdir(parents=True)
    (ancestor / ".env").write_text(
        "PROXY_HOST=ancestor-host\nCONTROL_SOCKET_PATH=/ancestor.sock\n"
    )
    if dotenv_contents is not None:
        (working_directory / ".env").write_text(dotenv_contents)
    monkeypatch.chdir(working_directory)
    monkeypatch.delenv("PROXY_HOST", raising=False)
    monkeypatch.delenv("PROXY_PORT", raising=False)
    monkeypatch.delenv("CONTROL_SOCKET_PATH", raising=False)

    settings = Settings.from_environment()

    assert settings.proxy_host == "0.0.0.0"
    assert settings.proxy_port == expected_port
    assert settings.control_socket_path is None


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


def test_settings_uses_runtime_defaults(monkeypatch):
    for name in (
        "PROXY_HOST",
        "PROXY_PORT",
        "CONTROL_SOCKET_PATH",
        "SESSION_RETENTION_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_environment()

    assert settings.proxy_host == "0.0.0.0"
    assert settings.proxy_port == 8082
    assert settings.control_socket_path is None
    assert settings.session_retention_limit == 1000


def test_settings_reads_runtime_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("PROXY_PORT", "9000")
    monkeypatch.setenv("CONTROL_SOCKET_PATH", "~/proxy.sock")
    monkeypatch.setenv("SESSION_RETENTION_LIMIT", "0")

    settings = Settings.from_environment()

    assert settings.proxy_host == "127.0.0.1"
    assert settings.proxy_port == 9000
    assert settings.control_socket_path == tmp_path / "proxy.sock"
    assert settings.session_retention_limit == 0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-an-integer", "PROXY_PORT must be an integer"),
        ("0", "PROXY_PORT must be between 1 and 65535"),
    ],
)
def test_settings_rejects_invalid_proxy_port(monkeypatch, value, message):
    monkeypatch.setenv("PROXY_PORT", value)

    with pytest.raises(ValueError, match=message):
        Settings.from_environment()


def test_session_retention_accepts_signed_64_maximum(monkeypatch):
    maximum = 2**63 - 1
    monkeypatch.setenv("SESSION_RETENTION_LIMIT", str(maximum))

    assert Settings.from_environment().session_retention_limit == maximum


def test_session_retention_rejects_above_signed_64_maximum(monkeypatch):
    monkeypatch.setenv("SESSION_RETENTION_LIMIT", str(2**63))

    with pytest.raises(ValueError, match="between 0 and 9223372036854775807"):
        Settings.from_environment()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-an-integer", "must be a non-negative integer"),
        ("-1", "must be between 0 and 9223372036854775807"),
    ],
)
def test_settings_rejects_invalid_session_retention_limit(
    monkeypatch, value, message
):
    monkeypatch.setenv("SESSION_RETENTION_LIMIT", value)

    with pytest.raises(ValueError, match=message):
        Settings.from_environment()


def test_settings_strips_control_socket_path_before_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CONTROL_SOCKET_PATH", "  ~/proxy.sock  ")

    settings = Settings.from_environment()

    assert settings.control_socket_path == tmp_path / "proxy.sock"


@pytest.mark.parametrize("value", ["", "   "])
def test_settings_rejects_empty_control_socket_path(monkeypatch, value):
    monkeypatch.setenv("CONTROL_SOCKET_PATH", value)

    with pytest.raises(ValueError, match="CONTROL_SOCKET_PATH must not be empty"):
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


def test_missing_mapping_path_is_safely_encoded_in_log(caplog, tmp_path):
    path = tmp_path / "missing\n\x1b\u202e.json"

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.config"):
        load_model_mapping(path)

    assert "missing\\x0a\\x1b\\u202e.json" in caplog.text
    assert "missing\n" not in caplog.text
    assert "\x1b" not in caplog.text
    assert "\u202e" not in caplog.text


def test_mapping_file_requires_models_object(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"tiers": {}, "mappings": {}}))

    with pytest.raises(ValueError, match="models"):
        load_model_mapping(path)


def test_model_context_window_enforces_signed_64_range():
    maximum = 2**63 - 1

    assert ModelDefinition(
        target="openai/model", context_window=maximum
    ).context_window == maximum
    with pytest.raises(ValueError, match="less than or equal"):
        ModelDefinition(target="openai/model", context_window=maximum + 1)


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
