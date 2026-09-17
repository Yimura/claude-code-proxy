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
    assert docker_base == "FROM python:3.14-slim"


def test_ci_uses_repository_python_version():
    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
    steps = workflow["jobs"]["unit-tests"]["steps"]
    setup = next(step for step in steps if step["name"] == "Set up Python")

    assert setup["with"]["python-version-file"] == ".python-version"
