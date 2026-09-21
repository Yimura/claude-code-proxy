from pathlib import Path

import pytest

from claude_code_proxy.config import PerformanceMode, Settings


def settings(**changes: object) -> Settings:
    values = {
        "anthropic_api_key": None,
        "openai_api_key": None,
        "gemini_api_key": None,
        "vertex_project": "unset",
        "vertex_location": "unset",
        "use_vertex_auth": False,
        "openai_base_url": None,
        "openai_transport": "litellm",
        "opencode_data_dir": Path("/data"),
        "model_mapping_path": Path("models.json"),
    }
    values.update(changes)
    return Settings(**values)


def test_direct_settings_default_performance_off() -> None:
    configured = settings()

    assert configured.performance_mode is PerformanceMode.OFF
    assert configured.performance_enabled is False
    assert configured.performance_logging_enabled is False


@pytest.mark.parametrize(
    ("mode", "enabled", "logging_enabled"),
    [
        (PerformanceMode.OFF, False, False),
        (PerformanceMode.COLLECTOR, True, False),
        (PerformanceMode.LOGGING, True, True),
    ],
)
def test_settings_derives_performance_capabilities(
    mode: PerformanceMode, enabled: bool, logging_enabled: bool
) -> None:
    configured = settings(performance_mode=mode)

    assert configured.performance_enabled is enabled
    assert configured.performance_logging_enabled is logging_enabled


def test_environment_cannot_enable_performance(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PERFORMANCE_MODE", "logging")
    monkeypatch.setenv("PERFORMANCE_ENABLED", "true")
    monkeypatch.setenv("PERFORMANCE", "logging")

    configured = Settings.from_environment()
    assert configured.performance_mode is PerformanceMode.OFF
    assert configured.performance_enabled is False
