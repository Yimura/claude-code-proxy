from pathlib import Path

import yaml

from claude_code_proxy.config import Settings, load_model_mapping

ROOT = Path(__file__).parents[2]
COMPOSE_PATH = ROOT / "docker-compose.yml"
DOCKERFILE_PATH = ROOT / "Dockerfile"
MAPPING_PATH = ROOT / "model_mapping.json"


def test_compose_uses_repository_model_mapping_as_application_config(monkeypatch):
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
    assert loaded.tiers
    assert loaded.mappings


def test_compose_publishes_proxy_only_on_ipv4_loopback():
    compose = yaml.safe_load(COMPOSE_PATH.read_text())

    assert compose["services"]["proxy"]["ports"] == ["127.0.0.1:8082:8082"]
