from datetime import UTC, datetime
import logging
import re

from claude_code_proxy.logging import (
    AgentIdentity,
    RequestLogContext,
    SessionIdentity,
    log_performance,
)
from claude_code_proxy.performance import Measurement, RequestPerformanceSnapshot
from test.unit.logging_test_support import make_context


def performance_snapshot(**changes):
    values = {
        "id": "request-safe-123",
        "session_id": "session-safe-123",
        "operation": "messages",
        "outcome": "completed",
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
        "duration": Measurement.observed(0),
        "upstream_duration": Measurement.unavailable(),
        "ttft": Measurement.not_applicable(),
        "input_tokens": Measurement.observed(0),
        "output_tokens": Measurement.unavailable(),
        "cache_read_tokens": Measurement.not_applicable(),
        "cache_creation_tokens": Measurement.observed(0),
        "reasoning_tokens": Measurement.unavailable(),
        "tool_calls": Measurement.not_applicable(),
        "retries": Measurement.observed(0),
        "peak_concurrency": Measurement.observed(1),
        "reasoning_continuation": "unavailable",
        "failure": None,
    }
    values.update(changes)
    return RequestPerformanceSnapshot(**values)


def test_performance_log_distinguishes_zero_unavailable_and_not_applicable(caplog):
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        log_performance(performance_snapshot(), make_context())

    assert len(caplog.records) == 1
    rendered = caplog.records[0].getMessage()
    assert caplog.records[0].levelno == logging.INFO
    for expected in (
        "performance",
        "request=request-safe-123",
        "operation=messages",
        "outcome=completed",
        "duration_ms=0",
        "upstream_ms=unavailable",
        "ttft_ms=not_applicable",
        "input_tokens=0",
        "output_tokens=unavailable",
        "cache_read_tokens=not_applicable",
        "cache_creation_tokens=0",
        "reasoning_tokens=unavailable",
        "tools=not_applicable",
        "retries=0",
        "peak_concurrency=1",
        "reasoning_continuation=unavailable",
        "model=claude-sonnet",
        "upstream=openai/gpt-5.6-sol",
        "provider=fake",
        "effort=high",
    ):
        assert expected in rendered


def test_failed_performance_log_is_warning(caplog):
    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        log_performance(
            performance_snapshot(outcome="failed"),
            make_context(),
        )

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "performance" in caplog.records[0].getMessage()
    assert "outcome=failed" in caplog.records[0].getMessage()


def test_performance_log_encodes_and_bounds_every_string_field(caplog):
    hostile = "field outcome=forged\r\n\t=\\\"'‮" + "x" * 400
    context = RequestLogContext(
        session=SessionIdentity("safe", "[session safe]", False),
        agent=AgentIdentity("agent-safe", "[agent agent-safe]", hostile, False),
        method="POST",
        endpoint="/v1/messages",
        original_model=hostile,
        upstream_model=hostile,
        provider=hostile,
        effort=hostile,
    )
    snapshot = performance_snapshot(
        id=hostile,
        operation=hostile,
        outcome=hostile,
        reasoning_continuation=hostile,
    )

    with caplog.at_level(logging.INFO, logger="claude_code_proxy.logging"):
        log_performance(snapshot, context)

    rendered = caplog.records[0].getMessage()
    assert len(rendered.splitlines()) == 1
    assert "\r" not in rendered
    assert "\n" not in rendered
    assert "\t" not in rendered
    assert "‮" not in rendered
    for field in (
        "request",
        "operation",
        "outcome",
        "reasoning_continuation",
        "model",
        "upstream",
        "provider",
        "effort",
    ):
        [value] = re.findall(rf"(?:^| ){field}=(\S+)", rendered)
        assert len(value) == 128
        assert value.endswith("...")
        assert "=" not in value
        assert '"' not in value
        assert "'" not in value
    assert "field outcome=forged" not in rendered


def test_performance_logging_failure_isolated(monkeypatch):
    def fail_log(*_args, **_kwargs):
        raise RuntimeError("sink secret")

    monkeypatch.setattr("claude_code_proxy.logging.logger.log", fail_log)

    log_performance(performance_snapshot(), make_context())
