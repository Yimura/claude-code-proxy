from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_image_defaults_off_but_compose_enables_collector() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = yaml.safe_load(
        (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )

    command = next(
        line for line in dockerfile.splitlines() if line.startswith("CMD ")
    )
    assert "--performance" not in command
    assert compose["services"]["proxy"]["command"] == [
        "claude-code-proxy",
        "proxy",
        "--performance",
        "collector",
    ]
