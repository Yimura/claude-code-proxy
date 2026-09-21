import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from claude_code_proxy.config import Settings, load_model_mapping

ROOT = Path(__file__).parents[2]
COMPOSE_PATH = ROOT / "docker-compose.yml"
DOCKERFILE_PATH = ROOT / "Dockerfile"
MAPPING_PATH = ROOT / "model_mapping.json"


def rendered_compose(tmp_path: Path, port: str | None = None) -> dict:
    project = tmp_path / "project"
    project.mkdir()
    for name in (
        "docker-compose.yml",
        "Dockerfile",
        "model_mapping.json",
        "pyproject.toml",
        "uv.lock",
        "README.md",
    ):
        shutil.copy2(ROOT / name, project / name)
    assert not (project / ".env").exists()
    (project / ".env").write_text("")

    env_file = tmp_path / "compose.env"
    env_file.write_text("")
    safe_home = tmp_path / "home"
    safe_home.mkdir()
    environment = {"HOME": str(safe_home), "PATH": os.environ["PATH"]}
    if port is not None:
        environment["PROXY_PORT"] = port
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "--project-directory",
            str(project),
            "-f",
            str(project / "docker-compose.yml"),
            "config",
            "--no-env-resolution",
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return yaml.safe_load(completed.stdout)


def test_compose_uses_repository_model_mapping_as_application_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    proxy = compose["services"]["proxy"]
    config = compose["configs"]["model_mapping"]
    grant = next(item for item in proxy["configs"] if item["source"] == "model_mapping")

    assert config["file"] == "./model_mapping.json"
    assert grant["target"] == "/claude-code-proxy/model_mapping.json"
    assert all("model_mapping.json" not in volume for volume in proxy.get("volumes", []))

    workdir = next(
        line.split(maxsplit=1)[1]
        for line in DOCKERFILE_PATH.read_text().splitlines()
        if line.startswith("WORKDIR ")
    )
    monkeypatch.delenv("MODEL_MAPPING_PATH", raising=False)
    assert grant["target"] == str(Path(workdir) / Settings.from_environment().model_mapping_path)
    assert config["file"] == f"./{MAPPING_PATH.relative_to(ROOT)}"

    loaded = load_model_mapping(MAPPING_PATH)
    assert loaded.models["terra"].context_window == 1_000_000
    assert loaded.models["sol"].context_window == 1_000_000
    assert loaded.tiers == {"small": "terra", "big": "sol"}
    assert loaded.mappings["fable"].model == "sol"
    assert loaded.mappings["fable"].effort == "xhigh"


def test_compose_keeps_proxy_listener_and_publication_aligned():
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    proxy = compose["services"]["proxy"]

    assert proxy["environment"]["PROXY_HOST"] == "0.0.0.0"
    assert proxy["environment"]["PROXY_PORT"] == "${PROXY_PORT:-8082}"
    assert proxy["ports"] == [
        "127.0.0.1:${PROXY_PORT:-8082}:${PROXY_PORT:-8082}"
    ]


@pytest.mark.parametrize(("configured", "expected"), [(None, 8082), ("9000", 9000)])
def test_rendered_compose_keeps_loopback_port_in_sync(
    tmp_path: Path, configured: str | None, expected: int
):
    compose = rendered_compose(tmp_path, configured)
    proxy = compose["services"]["proxy"]
    published = proxy["ports"][0]

    assert proxy["environment"]["PROXY_HOST"] == "0.0.0.0"
    assert proxy["environment"]["PROXY_PORT"] == str(expected)
    assert published["host_ip"] == "127.0.0.1"
    assert published["target"] == expected
    assert published["published"] == str(expected)


def test_compose_uses_image_cli_and_daemon_lifecycle():
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    proxy = compose["services"]["proxy"]

    assert proxy["restart"] == "unless-stopped"
    assert {"tty", "entrypoint"}.isdisjoint(proxy)
    assert proxy["command"] == [
        "claude-code-proxy",
        "proxy",
        "--performance",
        "collector",
    ]


def test_control_socket_remains_internal_to_container():
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    proxy = compose["services"]["proxy"]
    dockerfile = DOCKERFILE_PATH.read_text().splitlines()
    socket_environment = next(
        line.split("=", maxsplit=1)[1]
        for line in dockerfile
        if line.startswith("ENV CONTROL_SOCKET_PATH=")
    )

    volumes = proxy.get("volumes", [])
    auth_volume = (
        "${OPENCODE_DATA_DIR:-~/.local/share/opencode}:"
        "/opencode-auth:ro"
    )

    assert socket_environment == "/run/claude-code-proxy/control.sock"
    assert (
        proxy["environment"]["CONTROL_SOCKET_PATH"]
        == "/run/claude-code-proxy/control.sock"
    )
    assert volumes
    assert auth_volume in volumes
    source, target, access = auth_volume.rsplit(":", maxsplit=2)
    assert source == "${OPENCODE_DATA_DIR:-~/.local/share/opencode}"
    assert target == "/opencode-auth"
    assert access == "ro"
    for volume in volumes:
        serialized = yaml.safe_dump(volume)
        assert "/run/claude-code-proxy" not in serialized
        assert "control.sock" not in serialized
