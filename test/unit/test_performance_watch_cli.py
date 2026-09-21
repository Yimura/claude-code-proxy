from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
import json
from pathlib import Path

import pytest
import typer
from wcwidth import wcswidth

from claude_code_proxy import cli_common, performance_cli
from claude_code_proxy.cli import app
from claude_code_proxy.cli_common import OutputFormat
from claude_code_proxy.control.client import (
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)
from claude_code_proxy.control.schemas import (
    PerformanceCursorResponse,
    PerformanceEventResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
)
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.performance_watch_cli import (
    WATCH_HEADERS,
    render_watch_event,
)
from test.unit.cli_test_support import runner
from test.unit.test_performance_cli import (
    metric,
    performance_response,
    request_payload,
    view_payload,
)

_PROCESS = {"pid": 42, "started_at": "2026-01-02T03:00:00Z"}
_OCCURRED = "2026-01-02T03:04:06Z"


def reset_event(*views: dict[str, object], sequence: int = 7) -> PerformanceResetResponse:
    snapshot = performance_response(*views).model_dump(mode="json")
    snapshot["cursor"] = sequence
    return PerformanceResetResponse.model_validate(
        {
            "process": _PROCESS,
            "sequence": sequence,
            "occurred_at": _OCCURRED,
            "type": "reset",
            "snapshot": snapshot,
        }
    )


def ordinary_event(
    *,
    sequence: int = 8,
    request: dict[str, object] | None = None,
    event_type: str = "completed",
) -> PerformanceEventResponse:
    request_data = deepcopy(request or request_payload())
    view = view_payload(latest=request_data)
    return PerformanceEventResponse.model_validate(
        {
            "process": _PROCESS,
            "sequence": sequence,
            "occurred_at": _OCCURRED,
            "type": event_type,
            "session_id": "session-public",
            "activity": view["session"],
            "request": request_data,
            "session": view["performance"],
        }
    )


def cursor_event(sequence: int = 9) -> PerformanceCursorResponse:
    return PerformanceCursorResponse.model_validate(
        {
            "process": _PROCESS,
            "sequence": sequence,
            "occurred_at": _OCCURRED,
            "type": "cursor",
        }
    )


def cells(line: str) -> list[str]:
    return line.split()


def test_table_reset_snapshot_rows_precede_live_event() -> None:
    reset = render_watch_event(
        reset_event(view_payload(identifier="snapshot-session")),
        OutputFormat.TABLE,
        False,
    )
    live = render_watch_event(ordinary_event(), OutputFormat.TABLE, False)

    assert len(reset) == 1
    assert len(live) == 1
    assert "SNAPSHOT" in reset[0]
    assert "completed" in live[0]
    assert reset[0] != live[0]


def test_table_empty_and_repeated_resets_are_visible_without_headers() -> None:
    first = render_watch_event(reset_event(sequence=0), OutputFormat.TABLE, False)
    second = render_watch_event(reset_event(sequence=12), OutputFormat.TABLE, False)

    assert len(first) == len(second) == 1
    assert cells(first[0]) == ["03:04:06Z", "0", "—", "—", "—", "SNAPSHOT", "—", "—", "—", "—", "—"]
    assert cells(second[0])[1] == "12"
    assert all(header not in first[0] for header in ("TIME", "SESSION", "EVENT"))


def test_table_cursor_is_silently_omitted() -> None:
    assert render_watch_event(cursor_event(), OutputFormat.TABLE, False) == ()


def test_table_headers_and_columns_align_for_truncated_rows() -> None:
    header = "  ".join(WATCH_HEADERS)
    rows = (
        render_watch_event(
            reset_event(view_payload(identifier="session-public")),
            OutputFormat.TABLE,
            False,
        )[0],
        render_watch_event(ordinary_event(), OutputFormat.TABLE, False)[0],
    )

    assert WATCH_HEADERS == (
        "TIME", "SEQ", "SESSION", "REQUEST", "OPERATION", "EVENT",
        "ELAPSED", "TTFT", "TOKENS", "TOOLS", "RETRIES",
    )
    assert all(wcswidth(row) >= wcswidth(header) for row in rows)
    for column in ("03:04:06Z", "session-publ", "request-publ"):
        assert all(column in row for row in rows)


def test_table_truncation_and_no_trunc_are_terminal_safe() -> None:
    event = ordinary_event()
    poisoned = event.model_copy(
        update={
            "session_id": "session\n\x1b‮\ud800" + "s" * 40,
            "request": event.request.model_copy(
                update={
                    "id": "request\r\x1b​\ud800" + "r" * 40,
                }
            ),
        }
    )

    truncated = render_watch_event(poisoned, OutputFormat.TABLE, False)[0]
    full = render_watch_event(poisoned, OutputFormat.TABLE, True)[0]

    assert "s" * 20 not in truncated
    assert "s" * 20 in full
    assert "\\x0a" in full and "\\x1b" in full and "\\ud800" in full
    assert all(value not in full for value in ("\n", "\r", "\x1b", "‮", "​", "\ud800"))
    full.encode("utf-8", errors="strict")


def test_table_request_metrics_token_math_and_observed_zero() -> None:
    request = request_payload(duration=metric(value=0), ttft=metric(value=0))
    request.update(
        {
            "input_tokens": metric(value=10),
            "cache_read_tokens": metric(value=4),
            "cache_creation_tokens": metric(value=6),
            "output_tokens": metric(value=3),
            "tool_calls": metric(value=0),
            "retries": metric(value=0),
        }
    )

    row = cells(render_watch_event(
        ordinary_event(request=request), OutputFormat.TABLE, False
    )[0])

    assert row[6:] == ["0.00s", "0.00s", "20", "/", "3", "0", "0"]


@pytest.mark.parametrize("status", ["unavailable", "not_applicable"])
def test_table_request_metrics_render_unobserved_as_dash(status: str) -> None:
    request = request_payload(duration=metric(status), ttft=metric(status))
    for name in (
        "input_tokens", "cache_read_tokens", "cache_creation_tokens",
        "output_tokens", "tool_calls", "retries",
    ):
        request[name] = metric(status)

    row = render_watch_event(
        ordinary_event(request=request), OutputFormat.TABLE, False
    )[0]

    assert cells(row)[6:] == ["—", "—", "—", "/", "—", "—", "—"]


def test_table_alignment_handles_maximum_valid_metrics() -> None:
    baseline = render_watch_event(ordinary_event(), OutputFormat.TABLE, False)[0]
    request = request_payload(duration=metric(value=MAX_CONTROL_INTEGER))
    request["tool_calls"] = metric(value=MAX_CONTROL_INTEGER)
    request["retries"] = metric(value=7)

    boundary = render_watch_event(
        ordinary_event(request=request), OutputFormat.TABLE, False
    )[0]

    assert f"{MAX_CONTROL_INTEGER}.00s" in boundary
    assert boundary.index("0.50s") == baseline.index("0.50s")
    assert boundary.rfind("7") == baseline.rfind("0")


def test_table_rejects_nonfinite_derived_token_total() -> None:
    request = request_payload()
    for name in ("input_tokens", "cache_read_tokens", "cache_creation_tokens"):
        request[name] = metric(value=float.fromhex("0x1.fffffffffffffp+1023"))

    with pytest.raises(ControlError, match="invalid performance event stream"):
        render_watch_event(
            ordinary_event(request=request), OutputFormat.TABLE, False
        )


def test_table_count_tokens_output_and_request_only_metrics_are_not_applicable() -> None:
    request = request_payload()
    request["operation"] = "count_tokens"
    for name in ("ttft", "output_tokens", "tool_calls", "retries"):
        request[name] = metric("not_applicable")

    row = cells(render_watch_event(
        ordinary_event(request=request), OutputFormat.TABLE, False
    )[0])

    assert "count_tokens" in row
    assert row[-4:] == ["/", "—", "—", "—"]


def test_json_events_are_compact_exact_ndjson_objects() -> None:
    events = (reset_event(view_payload()), ordinary_event(), cursor_event())
    lines = [
        render_watch_event(event, OutputFormat.JSON, False)[0]
        for event in events
    ]

    assert [json.loads(line) for line in lines] == [
        event.model_dump(mode="json") for event in events
    ]
    assert all("\n" not in line and ": " not in line for line in lines)
    assert [json.loads(line)["type"] for line in lines] == [
        "reset", "completed", "cursor"
    ]


def test_json_uses_standard_escaping_and_rejects_nonfinite_bypass() -> None:
    event = ordinary_event()
    escaped = event.model_copy(
        update={"session_id": "line\nansi\x1b"}
    )
    line = render_watch_event(escaped, OutputFormat.JSON, False)[0]
    invalid = event.model_copy(
        update={
            "request": event.request.model_copy(
                update={
                    "duration": event.request.duration.model_copy(
                        update={"value": float("nan")}
                    )
                }
            )
        }
    )

    assert "line\\nansi\\u001b" in line
    with pytest.raises(ControlError, match="invalid performance event stream") as raised:
        render_watch_event(invalid, OutputFormat.JSON, False)
    assert "nan" not in str(raised.value).lower()


class FakeStream(Iterator[PerformanceStreamEvent]):
    def __init__(
        self,
        events: tuple[PerformanceStreamEvent, ...],
        error: BaseException | None = None,
    ) -> None:
        self.events = iter(events)
        self.error = error
        self.enter_count = 0
        self.close_count = 0
        self.closed = False

    def __iter__(self) -> FakeStream:
        return self

    def __next__(self) -> PerformanceStreamEvent:
        if self.error is not None:
            error, self.error = self.error, None
            self.close()
            raise error
        try:
            return next(self.events)
        except StopIteration:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.close_count += 1

    def __enter__(self) -> FakeStream:
        self.enter_count += 1
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FakeWatchClient:
    instances: list[FakeWatchClient] = []
    events: tuple[PerformanceStreamEvent, ...] = (reset_event(),)
    stream_error: BaseException | None = None
    open_error: Exception | None = None

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.filters: tuple[str, ...] | None = None
        self.enter_count = 0
        self.close_count = 0
        self.closed = False
        self.stream: FakeStream | None = None
        type(self).instances.append(self)

    def __enter__(self) -> FakeWatchClient:
        self.enter_count += 1
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.close_count += 1

    def performance_events(self, filters: tuple[str, ...]) -> FakeStream:
        self.filters = filters
        if type(self).open_error is not None:
            raise type(self).open_error
        self.stream = FakeStream(type(self).events, type(self).stream_error)
        return self.stream


@pytest.fixture(autouse=True)
def fake_watch_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeWatchClient.instances = []
    FakeWatchClient.events = (reset_event(),)
    FakeWatchClient.stream_error = None
    FakeWatchClient.open_error = None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(performance_cli, "ControlClient", FakeWatchClient)


def invoke_watch(*arguments: str):
    return runner.invoke(app, ["perf", "--watch", *arguments])


def assert_closed_once() -> FakeWatchClient:
    client = FakeWatchClient.instances[0]
    assert client.enter_count == client.close_count == 1
    if client.stream is not None:
        assert client.stream.enter_count == client.stream.close_count == 1
    return client


def test_watch_table_header_once_snapshot_then_live_and_clean_eof() -> None:
    FakeWatchClient.events = (
        reset_event(view_payload(identifier="snapshot-session")),
        ordinary_event(),
        reset_event(sequence=9),
        cursor_event(sequence=10),
    )

    result = invoke_watch()

    assert result.exit_code == 0
    assert result.stdout.count("TIME") == 1
    assert result.stdout.index("SNAPSHOT") < result.stdout.index("completed")
    assert result.stdout.count("SNAPSHOT") == 2
    assert "cursor" not in result.stdout
    assert result.stderr == "Performance stream closed\n"
    assert_closed_once()


def test_watch_json_emits_reset_ordinary_cursor_as_ndjson() -> None:
    FakeWatchClient.events = (reset_event(), ordinary_event(), cursor_event())

    result = invoke_watch("--format", "json")

    payloads = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.exit_code == 0
    assert [payload["type"] for payload in payloads] == [
        "reset", "completed", "cursor"
    ]
    assert result.stderr == "Performance stream closed\n"
    assert_closed_once()


def test_watch_forwards_filters_socket_and_loads_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        cli_common,
        "load_dotenv",
        lambda *, dotenv_path, override: calls.append((dotenv_path, override)),
    )
    socket_path = tmp_path / "control.sock"

    result = invoke_watch(
        "--socket", str(socket_path), "--filter", " state = idle "
    )

    client = assert_closed_once()
    assert result.exit_code == 0
    assert client.socket_path == socket_path
    assert client.filters == ("state=idle",)
    assert calls == [(tmp_path / ".env", False)]


def test_watch_keyboard_interrupt_closes_without_notice_or_abort() -> None:
    FakeWatchClient.stream_error = KeyboardInterrupt()

    result = invoke_watch()

    assert result.exit_code == 0
    assert result.stderr == ""
    assert "Aborted" not in result.output and "Traceback" not in result.output
    assert_closed_once()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ControlUnavailable(Path("/safe/control.sock"), "not found"),
            "Control API unavailable",
        ),
        (
            IncompatibleProtocol("required performance capability is disabled"),
            "required performance capability is disabled",
        ),
    ],
)
def test_watch_initial_unavailable_or_incompatible_has_collector_guidance(
    error: Exception, expected: str
) -> None:
    FakeWatchClient.open_error = error

    result = invoke_watch()

    assert result.exit_code == 1
    assert expected in result.stderr
    assert "--performance collector" in result.stderr
    assert "Traceback" not in result.output
    assert_closed_once()


def test_watch_control_error_is_generic_and_closes() -> None:
    FakeWatchClient.stream_error = ControlError(
        "Control API returned an invalid performance event stream"
    )

    result = invoke_watch()

    assert result.exit_code == 1
    assert result.stderr == (
        "Error: Control API returned an invalid performance event stream\n"
    )
    assert_closed_once()


def test_watch_consumer_output_error_is_generic_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_echo = typer.echo

    def broken_echo(message: object = None, *, err: bool = False, **kwargs: object) -> None:
        if not err:
            raise OSError("raw output device secret")
        original_echo(message, err=err, **kwargs)

    monkeypatch.setattr(performance_cli.typer, "echo", broken_echo)

    result = invoke_watch()

    assert result.exit_code == 1
    assert result.stderr == (
        "Error: Control API returned an invalid performance event stream\n"
    )
    assert "secret" not in result.stderr
    assert_closed_once()


def test_watch_invalid_usage_exits_two_before_network() -> None:
    result = invoke_watch("--format", "yaml")

    assert result.exit_code == 2
    assert FakeWatchClient.instances == []


def test_watch_help_and_dependencies() -> None:
    result = runner.invoke(app, ["perf", "--help"])
    sources = "\n".join(
        Path(module.__file__).read_text(encoding="utf-8")
        for module in (performance_cli, __import__(
            "claude_code_proxy.performance_watch_cli", fromlist=["*"]
        ))
    )

    assert result.exit_code == 0
    assert "--watch" in result.stdout
    assert ".providers" not in sources
    assert "Settings" not in sources
