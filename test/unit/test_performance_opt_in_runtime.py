from pathlib import Path

import pytest

from claude_code_proxy.config import PerformanceMode, Settings
from claude_code_proxy.runtime import create_runtime


def settings(tmp_path: Path, *, mode: PerformanceMode) -> Settings:
    return Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        gemini_api_key=None,
        vertex_project="unset",
        vertex_location="unset",
        use_vertex_auth=False,
        openai_base_url=None,
        openai_transport="litellm",
        opencode_data_dir=tmp_path,
        model_mapping_path=tmp_path / "missing.json",
        performance_mode=mode,
    )


@pytest.mark.parametrize(
    ("mode", "enabled", "logging_enabled"),
    [
        (PerformanceMode.OFF, False, False),
        (PerformanceMode.COLLECTOR, True, False),
        (PerformanceMode.LOGGING, True, True),
    ],
)
def test_runtime_passes_performance_policy_to_registry(
    tmp_path: Path,
    mode: PerformanceMode,
    enabled: bool,
    logging_enabled: bool,
) -> None:
    runtime = create_runtime(settings(tmp_path, mode=mode))

    assert runtime.sessions.performance_enabled is enabled
    assert runtime.sessions.performance_logging_enabled is logging_enabled
    assert runtime.events is runtime.sessions.events
