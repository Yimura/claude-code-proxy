import logging

import pytest

from claude_code_proxy.console_logging import LOG_FORMAT, SeverityFormatter
from claude_code_proxy.logging import agent_identity, session_identity


RESET = "\033[0m"


def make_record(level: int, message: str = "message") -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def make_formatter(environ: dict[str, str]) -> SeverityFormatter:
    formatter = SeverityFormatter(environ=environ)
    formatter.formatTime = lambda record, datefmt=None: "TIMESTAMP"
    return formatter


@pytest.mark.parametrize(
    ("level", "styled_token"),
    [
        (logging.DEBUG, f"\033[2mDEBUG{RESET}"),
        (logging.INFO, "INFO"),
        (logging.WARNING, f"\033[1;33mWARNING{RESET}"),
        (logging.ERROR, f"\033[1;31mERROR{RESET}"),
        (logging.CRITICAL, f"\033[1;31mCRITICAL{RESET}"),
    ],
)
def test_formatter_styles_only_supported_level_token(level, styled_token):
    formatter = make_formatter({})

    rendered = formatter.format(make_record(level))

    assert rendered == f"TIMESTAMP - {styled_token} - message"


def test_info_level_adds_no_ansi():
    rendered = make_formatter({}).format(make_record(logging.INFO))

    assert "\033[" not in rendered


def test_no_color_disables_formatter_and_identity_ansi():
    environ = {"NO_COLOR": ""}
    session = session_identity(
        "abcdef1234567890",
        request_scoped=False,
        is_new=False,
        environ=environ,
    )
    agent = agent_identity(
        "fedcba6543210987",
        None,
        is_new=False,
        environ=environ,
    )
    assert agent is not None
    message = f"{session.rendered} {agent.rendered} ready"

    rendered = make_formatter(environ).format(
        make_record(logging.WARNING, message)
    )

    assert rendered == f"TIMESTAMP - WARNING - {message}"
    assert "\033[" not in rendered


def test_formatter_preserves_existing_identity_ansi():
    session = session_identity(
        "abcdef1234567890",
        request_scoped=False,
        is_new=False,
        environ={},
    )
    agent = agent_identity(
        "fedcba6543210987",
        None,
        is_new=False,
        environ={},
    )
    assert agent is not None
    message = f"{session.rendered} {agent.rendered} ready"

    rendered = make_formatter({}).format(
        make_record(logging.WARNING, message)
    )

    assert rendered == (
        f"TIMESTAMP - \033[1;33mWARNING{RESET} - {message}"
    )


def test_warning_style_cannot_bleed_into_following_info_record():
    formatter = make_formatter({})

    warning = formatter.format(make_record(logging.WARNING, "first"))
    info = formatter.format(make_record(logging.INFO, "second"))

    assert warning == f"TIMESTAMP - \033[1;33mWARNING{RESET} - first"
    assert info == "TIMESTAMP - INFO - second"


def test_formatter_does_not_mutate_original_record():
    formatter = make_formatter({})
    record = make_record(logging.ERROR)

    formatter.format(record)

    assert record.levelname == "ERROR"


def test_formatter_preserves_args_and_custom_record_attributes():
    formatter = make_formatter({})
    record = make_record(logging.WARNING, "hello %s")
    record.args = ("world",)
    record.request_id = "request-123"

    rendered = formatter.format(record)

    assert rendered.endswith(" - hello world")
    assert not hasattr(record, "message")
    assert record.args == ("world",)
    assert record.request_id == "request-123"


def test_exception_record_can_be_formatted_twice_without_mutation():
    try:
        raise RuntimeError("safe failure")
    except RuntimeError as error:
        record = make_record(logging.ERROR, "failed %s")
        record.args = ("request",)
        record.request_id = "request-123"
        record.exc_info = (type(error), error, error.__traceback__)
    formatter = make_formatter({})

    first = formatter.format(record)
    second = formatter.format(record)

    assert first == second
    assert "RuntimeError: safe failure" in first
    assert not hasattr(record, "message")
    assert record.exc_text is None
    assert record.args == ("request",)
    assert record.request_id == "request-123"


def test_unknown_custom_level_is_plain():
    level = logging.CRITICAL + 5

    rendered = make_formatter({}).format(make_record(level))

    assert rendered == f"TIMESTAMP - {logging.getLevelName(level)} - message"
    assert "\033[" not in rendered


def test_formatter_captures_no_color_decision_at_construction():
    environ: dict[str, str] = {}
    formatter = make_formatter(environ)
    environ["NO_COLOR"] = "1"

    rendered = formatter.format(make_record(logging.WARNING))

    assert rendered == f"TIMESTAMP - \033[1;33mWARNING{RESET} - message"
    assert formatter._fmt == LOG_FORMAT
