from pathlib import Path

import pytest

from claude_code_proxy import orchestration_cli
from claude_code_proxy.cli import app
from claude_code_proxy.control.schemas import (
    OrchestrationAuthorizationListResponse,
    OrchestrationAuthorizationResponse,
)
from test.unit.cli_test_support import runner


class FakeClient:
    instances = []
    rows = OrchestrationAuthorizationListResponse(authorizations=())

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.calls = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def allow_nesting(self, session_id, *, max_depth, duration_seconds):
        self.calls.append(("allow", session_id, max_depth, duration_seconds))
        return OrchestrationAuthorizationResponse(
            session_id="a" * 64,
            max_depth=max_depth,
            remaining_seconds=float(duration_seconds),
        )

    def revoke_nesting(self, session_id):
        self.calls.append(("revoke", session_id))

    def orchestration_authorizations(self):
        self.calls.append(("list",))
        return type(self).rows


@pytest.fixture(autouse=True)
def fake_client(monkeypatch, tmp_path):
    FakeClient.instances = []
    FakeClient.rows = OrchestrationAuthorizationListResponse(authorizations=())
    monkeypatch.setattr(orchestration_cli, "ControlClient", FakeClient)
    monkeypatch.setattr(orchestration_cli, "resolve_socket_path", lambda value: tmp_path / "control.sock")
    monkeypatch.setattr(orchestration_cli, "load_current_directory_environment", lambda: None)


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("1", 1), ("1s", 1), ("2m", 120), ("1h", 3600), ("24h", 86400)],
)
def test_duration_grammar_accepts_exact_positive_integer_units(value, seconds):
    assert orchestration_cli.parse_duration(value) == seconds


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0",
        "0s",
        "86401",
        "25h",
        "1.5m",
        "+1m",
        " 1m",
        "1m ",
        "1M",
        "1ms",
        "9" * 5000,
    ],
)
def test_duration_grammar_rejects_invalid_or_out_of_range_values(value):
    with pytest.raises(ValueError, match="duration"):
        orchestration_cli.parse_duration(value)


def test_allow_nesting_prints_warning_before_authorizing_and_defaults_to_60m():
    result = runner.invoke(
        app,
        ["orchestration", "allow-nesting", "--session-id", "raw", "--max-depth", "3"],
    )

    assert result.exit_code == 0
    assert "recursive Agent delegation can increase fan-out and token use" in result.stderr
    assert FakeClient.instances[0].calls == [("allow", "raw", 3, 3600)]
    assert "raw" not in result.stdout
    assert "a" * 64 in result.stdout


def test_revoke_nesting_calls_private_client_without_rendering_raw_id():
    result = runner.invoke(
        app,
        ["orchestration", "revoke-nesting", "--session-id", "raw-secret"],
    )

    assert result.exit_code == 0
    assert FakeClient.instances[0].calls == [("revoke", "raw-secret")]
    assert "raw-secret" not in result.stdout + result.stderr


def test_authorizations_json_contains_only_safe_row_fields():
    FakeClient.rows = OrchestrationAuthorizationListResponse(
        authorizations=(
            OrchestrationAuthorizationResponse(
                session_id="b" * 64,
                max_depth=4,
                remaining_seconds=30.5,
            ),
        )
    )

    result = runner.invoke(app, ["orchestration", "authorizations", "--format", "json"])

    assert result.exit_code == 0
    assert result.stdout.strip() == (
        '[\n  {\n    "session_id": "' + "b" * 64 +
        '",\n    "max_depth": 4,\n    "remaining_seconds": 30.5\n  }\n]'
    )


def test_orchestration_subgroup_is_registered_in_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "orchestration" in result.stdout
