from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from wcwidth import wcswidth

from claude_code_proxy import cli_common, performance_cli
from claude_code_proxy.cli import app
from claude_code_proxy.cli_common import OutputFormat
from claude_code_proxy.control.client import (
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)
from claude_code_proxy.control.schemas import PerformanceListResponse
from test.unit.cli_test_support import runner

_CAPTURED_AT = "2026-01-02T03:04:06Z"
_HEADERS = (
    "SESSION  MODEL  STATE  REQS  ACTIVE  LAST  TTFT  TOKENS  CACHE  "
    "TOOLS  RETRIES  RESULT"
)
_REQUEST_METRICS = (
    "duration",
    "upstream_duration",
    "ttft",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "tool_calls",
    "retries",
    "peak_concurrency",
)
_DEFAULT_LATEST = object()
_AGGREGATE_METRICS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "tool_calls",
    "retries",
)


def metric(status: str = "observed", value: int | float = 0) -> dict[str, object]:
    return {"status": status, "value": value if status == "observed" else None}


def aggregate(
    value: int | float = 0,
    observed: int = 1,
    unavailable: int = 0,
    not_applicable: int = 0,
) -> dict[str, object]:
    return {
        "value": value,
        "observed_samples": observed,
        "unavailable_samples": unavailable,
        "not_applicable_samples": not_applicable,
    }


def request_payload(
    *,
    duration: dict[str, object] | None = None,
    ttft: dict[str, object] | None = None,
    outcome: str = "completed",
) -> dict[str, object]:
    payload = {
        "id": "request-public",
        "session_id": "session-public",
        "operation": "messages",
        "outcome": outcome,
        "started_at": "2026-01-02T03:04:05Z",
        "finished_at": "2026-01-02T03:04:06Z",
        **{name: metric() for name in _REQUEST_METRICS},
        "reasoning_continuation": "not_applicable",
        "failure": None,
    }
    payload["duration"] = duration or metric(value=1.25)
    payload["ttft"] = ttft or metric(value=0.5)
    return payload


def view_payload(
    *,
    identifier: str = "session-public",
    model: str = "provider-model",
    aggregates: dict[str, dict[str, object]] | None = None,
    latest: dict[str, object] | None | object = _DEFAULT_LATEST,
    requests: int = 1,
) -> dict[str, object]:
    latest_request = request_payload() if latest is _DEFAULT_LATEST else latest
    if latest_request is not None:
        latest_request = deepcopy(latest_request)
        latest_request["session_id"] = identifier
    values = {name: aggregate() for name in _AGGREGATE_METRICS}
    values.update(aggregates or {})
    recent = [] if latest_request is None else [latest_request]
    outcomes = {} if latest_request is None else {latest_request["outcome"]: requests}
    return {
        "session": {
            "id": identifier,
            "state": "idle",
            "active_requests": 0,
            "requests": requests,
            "client_model": "client-model",
            "model": model,
            "provider": "provider",
            "transport": "transport",
            "effort": "high",
            "context_window": 1000,
            "first_seen": "2026-01-02T03:04:05Z",
            "last_seen": "2026-01-02T03:04:06Z",
            "elapsed_seconds": 1,
            "last_result": "completed",
        },
        "performance": {
            "session_id": identifier,
            "requests": requests,
            "active_requests": [],
            "recent_requests": recent,
            "outcomes": outcomes,
            **values,
            "current_concurrency": 0,
            "peak_concurrency": 1,
            "latest_request": latest_request,
        },
    }


def performance_response(
    *views: dict[str, object],
) -> PerformanceListResponse:
    return PerformanceListResponse.model_validate(
        {
            "process": {"pid": 42, "started_at": "2026-01-02T03:00:00Z"},
            "captured_at": _CAPTURED_AT,
            "cursor": 7,
            "sessions": list(views),
        }
    )


def mixed_numeric_overflow_response() -> PerformanceListResponse:
    return performance_response(
        view_payload(
            aggregates={
                "input_tokens": aggregate(10**1000),
                "cache_read_tokens": aggregate(0.0),
                "cache_creation_tokens": aggregate(0),
            }
        )
    )


def render_table(result: PerformanceListResponse, no_trunc: bool = False) -> str:
    return performance_cli.render_performance(result, OutputFormat.TABLE, no_trunc)


def row_cells(rendered: str) -> list[str]:
    return rendered.splitlines()[1].split()


def test_table_empty_result_has_exact_headers_only() -> None:
    rendered = render_table(performance_response())

    assert rendered == _HEADERS


def test_table_preserves_server_order_and_renders_latest_metrics() -> None:
    result = performance_response(
        view_payload(identifier="b" * 64, model="newest", requests=1),
        view_payload(identifier="a" * 64, model="older", requests=1),
    )

    rendered = render_table(result)

    assert rendered.index("bbbbbbbbbbbb") < rendered.index("aaaaaaaaaaaa")
    assert "1.25s" in rendered
    assert "0.50s" in rendered
    assert rendered.count("completed") == 2


def test_table_formats_token_totals_and_cache_ratio() -> None:
    result = performance_response(
        view_payload(
            aggregates={
                "input_tokens": aggregate(100),
                "cache_read_tokens": aggregate(40),
                "cache_creation_tokens": aggregate(60),
                "output_tokens": aggregate(25),
            }
        )
    )

    rendered = render_table(result)

    assert "200 / 25" in rendered
    assert "20%" in rendered


def test_cache_ratio_unavailable_for_count_token_samples() -> None:
    result = performance_response(
        view_payload(
            requests=2,
            aggregates={
                "input_tokens": aggregate(960, observed=2),
                "cache_read_tokens": aggregate(
                    40, observed=1, not_applicable=1
                ),
                "cache_creation_tokens": aggregate(
                    0, observed=1, not_applicable=1
                ),
            },
        )
    )

    assert row_cells(render_table(result))[10] == "—"


@pytest.mark.parametrize(
    ("aggregates", "tokens", "cache", "tools", "retries"),
    [
        ({}, "0 / 0", "0%", "0", "0"),
        (
            {
                name: aggregate(observed=0, unavailable=1)
                for name in _AGGREGATE_METRICS
            },
            "— / —",
            "—",
            "—",
            "—",
        ),
        (
            {
                name: aggregate(observed=0, not_applicable=1)
                for name in _AGGREGATE_METRICS
            },
            "— / —",
            "—",
            "—",
            "—",
        ),
        (
            {
                "input_tokens": aggregate(10, unavailable=1),
                "cache_read_tokens": aggregate(5),
                "cache_creation_tokens": aggregate(0),
                "output_tokens": aggregate(2, unavailable=1),
                "tool_calls": aggregate(3, unavailable=1),
                "retries": aggregate(0, unavailable=1),
            },
            "15+? / 2+?",
            "33.3%+?",
            "3+?",
            "0+?",
        ),
    ],
)
def test_table_distinguishes_observed_unavailable_not_applicable_and_partial(
    aggregates: dict[str, dict[str, object]],
    tokens: str,
    cache: str,
    tools: str,
    retries: str,
) -> None:
    result = performance_response(view_payload(aggregates=aggregates))
    cells = row_cells(render_table(result))

    assert " ".join(cells[7:10]).startswith(tokens)
    assert cache in cells
    assert tools in cells
    assert retries in cells


@pytest.mark.parametrize("status", ["unavailable", "not_applicable"])
def test_table_uses_dash_for_unobserved_latest_duration_and_ttft(status: str) -> None:
    latest = request_payload(duration=metric(status), ttft=metric(status))

    cells = row_cells(render_table(performance_response(view_payload(latest=latest))))

    assert cells[5:7] == ["—", "—"]


def test_table_uses_dash_when_no_latest_request() -> None:
    rendered = render_table(performance_response(view_payload(latest=None, requests=0)))

    assert row_cells(rendered)[5:7] == ["—", "—"]
    assert row_cells(rendered)[-1] == "—"


def test_table_sanitizes_untrusted_fields_and_aligns_display_width() -> None:
    safe = performance_response(
        view_payload(identifier="a" * 64, model="界界e\N{COMBINING ACUTE ACCENT}"),
        view_payload(identifier="b" * 64, model="1234567"),
    )
    first = safe.sessions[0]
    poisoned_activity = first.session.model_copy(
        update={
            "id": "id\n\x1b‮\ud800" + "x" * 64,
            "model": "model\n\x1b​\ud800" + "\N{COMBINING ACUTE ACCENT}" * 200,
            "provider": "provider\rforged",
        }
    )
    poisoned = safe.model_copy(
        update={
            "sessions": (
                first.model_copy(update={"session": poisoned_activity}),
                safe.sessions[1],
            )
        }
    )

    rendered = render_table(poisoned)
    rows = rendered.splitlines()[1:]
    state_offsets = [wcswidth(row[: row.index("idle")]) for row in rows]

    assert "\\x0a" in rendered
    assert "\\x1b" in rendered
    assert "\\u200b" in rendered
    assert "\\ud800" not in rendered  # truncated safely before the surrogate
    assert "provider" not in rendered
    assert all(value not in rendered for value in ("\n\x1b", "‮", "​", "\ud800"))
    assert len(rows[0]) < 250
    assert state_offsets == [state_offsets[0], state_offsets[0]]
    rendered.encode("utf-8", errors="strict")


def test_table_no_trunc_preserves_full_safe_session_and_model() -> None:
    identifier = "s" * 64
    model = "m" * 80
    result = performance_response(view_payload(identifier=identifier, model=model))

    truncated = render_table(result)
    full = render_table(result, no_trunc=True)

    assert identifier not in truncated
    assert model not in truncated
    assert identifier in full
    assert model in full


def test_json_is_exact_complete_object_with_standard_escaping() -> None:
    result = performance_response(view_payload())
    first = result.sessions[0]
    activity = first.session.model_copy(update={"model": "line\nansi\x1b"})
    bypassed = result.model_copy(
        update={"sessions": (first.model_copy(update={"session": activity}),)}
    )

    rendered = performance_cli.render_performance(bypassed, OutputFormat.JSON, False)

    assert json.loads(rendered) == bypassed.model_dump(mode="json")
    assert "line\\nansi\\u001b" in rendered
    assert tuple(json.loads(rendered)) == (
        "process",
        "captured_at",
        "cursor",
        "sessions",
    )


def test_json_empty_result_is_complete_object_with_empty_sessions() -> None:
    result = performance_response()

    payload = json.loads(
        performance_cli.render_performance(result, OutputFormat.JSON, False)
    )

    assert payload["sessions"] == []
    assert payload["process"]["pid"] == 42
    assert payload["cursor"] == 7


def test_json_nonfinite_contract_bypass_raises_generic_safe_error() -> None:
    result = performance_response(view_payload())
    first = result.sessions[0]
    broken_performance = first.performance.model_copy(
        update={
            "input_tokens": first.performance.input_tokens.model_copy(
                update={"value": float("nan")}
            )
        }
    )
    bypassed = result.model_copy(
        update={
            "sessions": (
                first.model_copy(update={"performance": broken_performance}),
            )
        }
    )

    with pytest.raises(ControlError, match="invalid performance response") as raised:
        performance_cli.render_performance(bypassed, OutputFormat.JSON, False)

    assert "nan" not in str(raised.value).lower()


class FakePerformanceClient:
    instances: list["FakePerformanceClient"] = []
    result = performance_response()
    error: Exception | None = None

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.filters: tuple[str, ...] | None = None
        self.entered = False
        self.exited = False
        type(self).instances.append(self)

    def __enter__(self) -> "FakePerformanceClient":
        self.entered = True
        return self

    def __exit__(self, *args: object) -> None:
        self.exited = True

    def performance(self, filters: tuple[str, ...]) -> PerformanceListResponse:
        self.filters = filters
        if type(self).error is not None:
            raise type(self).error
        return type(self).result


@pytest.fixture(autouse=True)
def reset_fake_performance_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakePerformanceClient.instances = []
    FakePerformanceClient.result = performance_response()
    FakePerformanceClient.error = None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        performance_cli, "ControlClient", FakePerformanceClient, raising=False
    )


def test_perf_success_forwards_filters_and_closes_client(tmp_path: Path) -> None:
    FakePerformanceClient.result = performance_response(view_payload())

    result = runner.invoke(
        app,
        [
            "perf",
            "--socket",
            str(tmp_path / "control.sock"),
            "--filter",
            " state = idle ",
            "--filter",
            "model = provider-model",
        ],
    )

    client = FakePerformanceClient.instances[0]
    assert result.exit_code == 0
    assert result.stdout.startswith("SESSION")
    assert client.filters == ("state=idle", "model=provider-model")
    assert client.entered and client.exited


@pytest.mark.parametrize(
    "arguments",
    [
        ["--format", "yaml"],
        ["--filter", "unknown=value"],
        ["--filter", "missing-separator"],
    ],
)
def test_perf_invalid_usage_exits_two_before_network(arguments: list[str]) -> None:
    result = runner.invoke(app, ["perf", *arguments])

    assert result.exit_code == 2
    assert FakePerformanceClient.instances == []


@pytest.mark.parametrize(
    ("exported", "cli_socket", "expected_name"),
    [
        (None, None, "file.sock"),
        ("exported.sock", None, "exported.sock"),
        ("exported.sock", "cli.sock", "cli.sock"),
    ],
)
def test_perf_socket_precedence_in_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exported: str | None,
    cli_socket: str | None,
    expected_name: str,
) -> None:
    (tmp_path / ".env").write_text(
        f"CONTROL_SOCKET_PATH={tmp_path / 'file.sock'}\n"
    )
    monkeypatch.delenv("CONTROL_SOCKET_PATH", raising=False)
    if exported is not None:
        monkeypatch.setenv("CONTROL_SOCKET_PATH", str(tmp_path / exported))
    arguments = ["perf"]
    if cli_socket is not None:
        arguments.extend(["--socket", str(tmp_path / cli_socket)])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert FakePerformanceClient.instances[0].socket_path == (
        tmp_path / expected_name
    ).absolute()


def test_perf_loads_current_directory_dotenv_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        cli_common,
        "load_dotenv",
        lambda *, dotenv_path, override: calls.append((dotenv_path, override)),
    )

    result = runner.invoke(
        app, ["perf", "--socket", str(tmp_path / "control.sock")]
    )

    assert result.exit_code == 0
    assert calls == [(tmp_path / ".env", False)]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ControlUnavailable(Path("/safe/control.sock"), "not found"),
            "Control API unavailable",
        ),
        (
            IncompatibleProtocol(
                "Control API does not advertise the required performance capability"
            ),
            "required performance capability",
        ),
        (ControlError("safe failure"), "safe failure"),
    ],
)
def test_perf_errors_exit_one_safely_and_close_client(
    error: Exception, expected: str
) -> None:
    FakePerformanceClient.error = error

    result = runner.invoke(app, ["perf", "--socket", "/safe/control.sock"])

    client = FakePerformanceClient.instances[0]
    assert result.exit_code == 1
    assert expected in result.stderr
    assert "Traceback" not in result.stderr
    assert client.entered and client.exited
    if isinstance(error, (ControlUnavailable, IncompatibleProtocol)):
        guidance = (
            "source: start `claude-code-proxy proxy --performance collector`\n"
            "Docker: configure service command "
            "`claude-code-proxy proxy --performance collector`, recreate the "
            "service, then run "
            "`docker compose exec proxy claude-code-proxy perf`\n"
        )
        assert result.stderr.endswith(guidance)
        assert "source: start `claude-code-proxy proxy`\n" not in result.stderr
        assert "claude-code-proxy ps" not in result.stderr


def test_perf_mixed_numeric_overflow_closes_client_and_hides_raw_error() -> None:
    FakePerformanceClient.result = mixed_numeric_overflow_response()

    invoked = runner.invoke(app, ["perf"])

    assert invoked.exit_code == 1
    assert invoked.stderr == (
        "Error: Control API returned an invalid performance response\n"
    )
    assert "int too large to convert to float" not in invoked.stderr
    assert "OverflowError" not in invoked.stderr
    assert "Traceback" not in invoked.stderr
    assert FakePerformanceClient.instances[0].exited


def test_perf_serialization_failure_closes_client_and_hides_raw_value() -> None:
    result = performance_response(view_payload())
    first = result.sessions[0]
    invalid = first.performance.input_tokens.model_copy(
        update={"value": float("nan")}
    )
    FakePerformanceClient.result = result.model_copy(
        update={
            "sessions": (
                first.model_copy(
                    update={
                        "performance": first.performance.model_copy(
                            update={"input_tokens": invalid}
                        )
                    }
                ),
            )
        }
    )

    invoked = runner.invoke(app, ["perf", "--format", "json"])

    assert invoked.exit_code == 1
    assert "invalid performance response" in invoked.stderr
    assert "nan" not in invoked.stderr.lower()
    assert FakePerformanceClient.instances[0].exited


def test_perf_module_has_no_provider_or_settings_dependencies() -> None:
    source = Path(performance_cli.__file__).read_text(encoding="utf-8")

    assert ".providers" not in source
    assert "Settings" not in source


def test_cache_ratio_is_unavailable_without_observed_cache_read_samples() -> None:
    for unavailable, not_applicable in ((1, 0), (0, 1)):
        result = performance_response(
            view_payload(
                aggregates={
                    "input_tokens": aggregate(100),
                    "cache_read_tokens": aggregate(
                        observed=0,
                        unavailable=unavailable,
                        not_applicable=not_applicable,
                    ),
                    "cache_creation_tokens": aggregate(0),
                }
            )
        )

        assert row_cells(render_table(result))[10] == "—"


def test_table_mixed_numeric_overflow_raises_generic_safe_error() -> None:
    with pytest.raises(ControlError) as raised:
        render_table(mixed_numeric_overflow_response())

    assert str(raised.value) == (
        "Control API returned an invalid performance response"
    )


def test_table_nonfinite_contract_bypass_raises_generic_safe_error() -> None:
    result = performance_response(view_payload())
    first = result.sessions[0]
    latest = first.performance.latest_request
    assert latest is not None
    broken_latest = latest.model_copy(
        update={"duration": latest.duration.model_copy(update={"value": float("inf")})}
    )
    broken_performance = first.performance.model_copy(
        update={"latest_request": broken_latest}
    )
    bypassed = result.model_copy(
        update={
            "sessions": (
                first.model_copy(update={"performance": broken_performance}),
            )
        }
    )

    with pytest.raises(ControlError, match="invalid performance response") as raised:
        render_table(bypassed)

    assert "inf" not in str(raised.value).lower()


def test_table_no_trunc_escapes_surrogates_and_controls() -> None:
    result = performance_response(view_payload())
    first = result.sessions[0]
    activity = first.session.model_copy(
        update={"id": "id\ud800", "model": "model\x1b\ud800"}
    )
    bypassed = result.model_copy(
        update={"sessions": (first.model_copy(update={"session": activity}),)}
    )

    rendered = render_table(bypassed, no_trunc=True)

    assert "id\\ud800" in rendered
    assert "model\\x1b\\ud800" in rendered
    assert "\x1b" not in rendered
    assert "\ud800" not in rendered
    rendered.encode("utf-8", errors="strict")


def test_perf_accepts_filter_count_length_and_supported_key_boundaries() -> None:
    filters = [f"state=value-{index}" for index in range(25)]
    filters.extend(
        [
            "id=public-prefix",
            "session_id=raw-session",
            "provider=provider",
            "transport=transport",
            "model=" + "m" * 250,
            "effort=high",
            "state=idle",
        ]
    )
    arguments = ["perf"]
    for item in filters:
        arguments.extend(["--filter", item])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert FakePerformanceClient.instances[0].filters == tuple(filters)
    assert len(filters) == 32
    assert len(filters[-3]) == 256
