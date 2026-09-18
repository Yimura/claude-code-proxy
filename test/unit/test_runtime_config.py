from pathlib import Path
import tomllib
import yaml

ROOT = Path(__file__).parents[2]
PYTHON_SERIES = "3.14"


def test_python_runtime_contract_is_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    docker_base = (ROOT / "Dockerfile").read_text().splitlines()[0]

    assert (ROOT / ".python-version").read_text().strip() == PYTHON_SERIES
    assert project["project"]["requires-python"] == ">=3.14,<3.15"
    assert project["project"]["scripts"] == {
        "claude-code-proxy": "claude_code_proxy.cli:app"
    }
    assert docker_base == "FROM python:3.14-slim"


def test_ci_uses_repository_python_version():
    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
    steps = workflow["jobs"]["unit-tests"]["steps"]
    setup = next(step for step in steps if step["name"] == "Set up Python")

    assert setup["with"]["python-version-file"] == ".python-version"


def test_docker_image_runs_cli_from_installed_virtual_environment():
    dockerfile = (ROOT / "Dockerfile").read_text()
    instructions = [line.strip() for line in dockerfile.splitlines() if line.strip()]

    sync_steps = [
        line
        for line in instructions
        if line.startswith("RUN ") and "uv sync" in line
    ]
    assert len(sync_steps) == 2
    dependency_step, application_step = sync_steps

    assert "ENV PATH=/claude-code-proxy/.venv/bin:$PATH" in instructions
    assert instructions[-1] == 'CMD ["claude-code-proxy", "proxy"]'
    assert all("--locked" in line and "--no-dev" in line for line in sync_steps)
    assert "--no-install-project" in dependency_step
    assert "--no-install-project" not in application_step
    assert [line for line in instructions if line.startswith("EXPOSE ")] == [
        "EXPOSE 8082"
    ]
    assert "claude_code_proxy.app:app" not in dockerfile
    assert "uvicorn" not in dockerfile


def test_docker_image_provisions_private_control_socket_runtime_directory():
    dockerfile = (ROOT / "Dockerfile").read_text()
    instructions = [line.strip() for line in dockerfile.splitlines() if line.strip()]

    assert "RUN install -d -m 0700 /run/claude-code-proxy" in instructions
    assert (
        "ENV CONTROL_SOCKET_PATH=/run/claude-code-proxy/control.sock"
        in instructions
    )


def test_example_environment_documents_runtime_defaults_and_credentials():
    environment = (ROOT / ".env.example").read_text()
    lines = [line.strip() for line in environment.splitlines()]
    assignments = dict(
        line.split("=", maxsplit=1)
        for line in lines
        if line and not line.startswith("#") and "=" in line
    )
    comments = "\n".join(line for line in lines if line.startswith("#"))
    documented_names = {
        line.removeprefix("# ").split("=", maxsplit=1)[0]
        for line in lines
        if "=" in line
    }

    assert assignments["PROXY_HOST"] == "127.0.0.1"
    assert assignments["PROXY_PORT"] == "8082"
    assert assignments["SESSION_RETENTION_LIMIT"] == "1000"
    assert "CONTROL_SOCKET_PATH" not in assignments
    assert "CONTROL_SOCKET_PATH" in comments
    assert "absolute" in comments
    assert "existing private parent" in comments
    assert "container-internal" in comments
    assert "/run/claude-code-proxy/control.sock" in comments
    assert "~/.local/run" not in environment
    assert {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GEMINI_API_KEY",
        "USE_VERTEX_AUTH",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "OPENAI_TRANSPORT",
        "OPENCODE_DATA_DIR",
        "MODEL_MAPPING_PATH",
    } <= documented_names
