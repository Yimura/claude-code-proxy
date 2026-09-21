from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_container_defaults_do_not_enable_performance() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = yaml.safe_load(
        (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )

    command = next(
        line for line in dockerfile.splitlines() if line.startswith("CMD ")
    )
    assert "--performance" not in command
    assert "--performance" not in yaml.safe_dump(compose["services"]["proxy"])
