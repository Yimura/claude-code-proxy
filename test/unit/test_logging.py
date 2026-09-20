import asyncio
import io
import json
import logging
import re

import pytest

from claude_code_proxy.console_logging import LOG_FORMAT, SeverityFormatter
from claude_code_proxy.domain.models import (
    ClientIdentity,
    StreamComplete,
    StreamError,
    TextDelta,
    TokenUsage,
)
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.logging import (
    AgentIdentity,
    MessageFilter,
    RequestLogContext,
    SessionIdentity,
    agent_identity,
    client_identity_from_headers,
    configure_logging,
    effective_effort,
    log_provider_failure,
    log_startup_summary,
    log_session_started,
    log_stream_failure,
    log_unexpected_failure,
    observe_stream,
    palette_index,
    session_identity,
)
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.providers.codex.auth import CodexAccountIdentity
from claude_code_proxy.reasoning import ReasoningPolicy


def make_context(identity: SessionIdentity | None = None) -> RequestLogContext:
    return RequestLogContext(
        session=identity
        or SessionIdentity("abcdef123456", "[session abcdef123456]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude-sonnet",
        upstream_model="openai/gpt-5.6-sol",
        provider="fake",
        effort="high",
    )




def test_agent_identity_hides_raw_values():
    identity = agent_identity(
        "a" * 64,
        "b" * 64,
        is_new=True,
        environ={"NO_COLOR": "1"},
    )

    assert identity == AgentIdentity(
        label="a" * 12,
        rendered=f"[agent {'a' * 12}]",
        parent_label="b" * 12,
        is_new=True,
    )

def test_client_identity_from_headers_reads_full_lineage():
    identity = client_identity_from_headers(
        {
            "x-claude-code-session-id": " session ",
            "x-claude-code-agent-id": " agent ",
            "x-claude-code-parent-agent-id": " parent ",
        }
    )

    assert identity.session_id == " session "
    assert identity.agent_id == " agent "
    assert identity.parent_agent_id == " parent "


def test_client_identity_from_headers_normalizes_blank_and_orphan_parent():
    blank = client_identity_from_headers(
        {
            "x-claude-code-session-id": "   ",
            "x-claude-code-agent-id": "\t",
            "x-claude-code-parent-agent-id": "parent",
        }
    )
    orphan = client_identity_from_headers(
        {"x-claude-code-parent-agent-id": "parent"}
    )

    assert blank == ClientIdentity()
    assert orphan == ClientIdentity()

def test_startup_summary_reports_litellm_transport(caplog):
    with caplog.at_level(
        logging.INFO, logger="claude_code_proxy.logging.readiness"
    ):
        log_startup_summary("litellm")

    assert [record.getMessage() for record in caplog.records] == [
        "OpenAI transport: litellm"
    ]


def test_startup_summary_reports_safe_codex_identity_and_switch_guidance(caplog):
    identity = CodexAccountIdentity(
        account_id="account-123",
        masked_email="j***@crimson7.io",
        source="opencode.db",
    )

    with caplog.at_level(
        logging.INFO, logger="claude_code_proxy.logging.readiness"
    ):
        log_startup_summary("codex", identity)

    assert [record.getMessage() for record in caplog.records] == [
        "OpenAI transport: codex",
        "OpenCode account: j***@crimson7.io [account-123] (opencode.db)",
        "To use another account, stop the proxy, switch the active OpenAI "
        "account in OpenCode, and restart.",
    ]


def test_startup_summary_falls_back_to_encoded_account_id(caplog):
    identity = CodexAccountIdentity(
        account_id="account\n\x1b\u202e",
        masked_email=None,
        source="auth.json",
    )

    with caplog.at_level(
        logging.INFO, logger="claude_code_proxy.logging.readiness"
    ):
        log_startup_summary("codex", identity)

    assert [record.getMessage() for record in caplog.records] == [
        "OpenAI transport: codex",
        "OpenCode account: account\\x0a\\x1b\\u202e (auth.json)",
        "To use another account, stop the proxy, switch the active OpenAI "
        "account in OpenCode, and restart.",
    ]


@pytest.fixture
def isolated_logging_state():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_filters = root.filters[:]
    original_level = root.level
    logger_names = (
        "claude_code_proxy.logging.session",
        "claude_code_proxy.logging.readiness",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
    )
    original_levels = {
        name: logging.getLogger(name).level for name in logger_names
    }
    for handler in original_handlers:
        root.removeHandler(handler)
    root.filters.clear()
    root.setLevel(logging.NOTSET)

    yield root

    for handler in root.handlers[:]:
        root.removeHandler(handler)
        if handler not in original_handlers:
            handler.close()
    for handler in original_handlers:
        root.addHandler(handler)
    root.filters[:] = original_filters
    root.setLevel(original_level)
    for name, level in original_levels.items():
        logging.getLogger(name).setLevel(level)


def assert_console_configuration(root):
    assert root.level == logging.WARN
    assert len(root.handlers) == 1
    [handler] = root.handlers
    assert isinstance(handler, logging.StreamHandler)
    assert isinstance(handler.formatter, SeverityFormatter)
    assert handler.formatter._fmt == LOG_FORMAT
    assert len(handler.filters) == 1
    assert isinstance(handler.filters[0], MessageFilter)
    assert not any(isinstance(item, MessageFilter) for item in root.filters)
    assert logging.getLogger(
        "claude_code_proxy.logging.readiness"
    ).level == logging.INFO
    assert logging.getLogger(
        "claude_code_proxy.logging.session"
    ).level == logging.INFO
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        assert logging.getLogger(name).level == logging.WARNING
    return handler


def test_configure_logging_installs_owned_console_handler(
    isolated_logging_state,
):
    configure_logging()

    assert_console_configuration(isolated_logging_state)


def test_console_handler_filters_propagated_child_records(
    isolated_logging_state,
):
    configure_logging()
    handler = assert_console_configuration(isolated_logging_state)
    output = io.StringIO()
    handler.setStream(output)
    child = logging.getLogger("task6.message_filter.child")
    child.handlers.clear()
    child.setLevel(logging.NOTSET)
    child.propagate = True

    child.warning("HTTP Request: hidden")
    child.warning("allowed message")

    rendered = output.getvalue()
    assert "HTTP Request: hidden" not in rendered
    assert "allowed message" in rendered


class TrackingHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(io.StringIO())
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


def test_configure_logging_replaces_preconfigured_root_without_losing_filters(
    isolated_logging_state,
):
    root = isolated_logging_state
    unrelated_filter = logging.Filter("unrelated")
    root.addFilter(unrelated_filter)
    root.addFilter(MessageFilter())
    previous_handler = TrackingHandler()
    root.addHandler(previous_handler)

    configure_logging()

    assert previous_handler.was_closed is True
    assert root.filters == [unrelated_filter]
    assert_console_configuration(root)


def test_configure_logging_is_idempotent(isolated_logging_state):
    root = isolated_logging_state

    configure_logging()
    first_handler = assert_console_configuration(root)
    configure_logging()
    second_handler = assert_console_configuration(root)

    assert second_handler is not first_handler
    assert first_handler._closed is True


def test_palette_index_is_stable_and_bounded():
    first = palette_index("safe-public-id")
    assert first == palette_index("safe-public-id")
    assert 0 <= first < 5


def test_session_identity_uses_twelve_character_safe_public_label():
    identity = session_identity(
        "abcdef1234567890", request_scoped=False, is_new=True, environ={}
    )

    assert identity.label == "abcdef123456"
    assert identity.rendered.endswith("abcdef123456\033[0m]")
    assert identity.is_new is True


def test_request_scoped_identity_is_clearly_marked():
    identity = session_identity(
        "abcdef1234567890", request_scoped=True, is_new=True, environ={}
    )

    assert identity.label == "abcdef123456"
    assert identity.rendered.startswith("[request \033[")
    assert identity.is_new is True


def test_no_color_disables_identity_color():
    identity = session_identity(
        "abcdef1234567890",
        request_scoped=False,
        is_new=False,
        environ={"NO_COLOR": "1"},
    )

    assert identity.rendered == "[session abcdef123456]"


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ReasoningPolicy(True, "minimal"), "minimal"),
        (ReasoningPolicy(True, "high"), "high"),
        (ReasoningPolicy(False, None), "none"),
        (ReasoningPolicy(None, None), "default"),
    ],
)
def test_effective_effort(policy, expected):
    assert effective_effort(policy) == expected



def test_root_lifecycle_log_excludes_agent_context(caplog):
    context = RequestLogContext(
        session=SessionIdentity("session-safe", "[session session-safe]", True),
        agent=AgentIdentity("agent-safe", "[agent agent-safe]", None, True),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude-sonnet",
        upstream_model="openai/gpt-5.6-sol",
        provider="codex",
        effort="high",
    )

    with caplog.at_level(
        logging.INFO,
        logger="claude_code_proxy.logging.session",
    ):
        log_session_started(context)

    assert "[session session-safe]" in caplog.text
    assert "[agent agent-safe]" not in caplog.text

def test_untrusted_log_context_escapes_record_and_terminal_controls(caplog):
    hostile = "field\n\r\t\x1b\x85\u2028\u2029\u202e\ud800"
    context = RequestLogContext(
        session=SessionIdentity("safe", "[session safe]", True),
        method=hostile,
        endpoint=hostile,
        original_model=hostile,
        upstream_model=hostile,
        provider=hostile,
        effort=hostile,
    )

    with caplog.at_level(logging.INFO):
        log_session_started(context)
        log_provider_failure(
            context,
            ProviderError(
                "Service unavailable", provider="fake", status_code=503
            ),
        )
        log_stream_failure(
            context,
            StreamError(error_type=hostile, provider=hostile),
        )
        log_unexpected_failure(context, RuntimeError("not logged"))

    rendered = caplog.text
    for token in (
        "\\x0a",
        "\\x0d",
        "\\x09",
        "\\x1b",
        "\\x85",
        "\\u2028",
        "\\u2029",
        "\\u202e",
        "\\ud800",
    ):
        assert token in rendered
    for control in ("\r", "\t", "\x1b", "\x85", "\u2028", "\u2029", "\u202e", "\ud800"):
        assert control not in rendered
    assert "field\n" not in rendered
    rendered.encode("utf-8", errors="strict")


def test_application_generated_session_ansi_is_preserved(caplog):
    context = make_context(
        SessionIdentity("abcdef123456", "[session \x1b[96mabcdef123456\x1b[0m]", True)
    )

    with caplog.at_level(
        logging.INFO, logger="claude_code_proxy.logging.session"
    ):
        log_session_started(context)

    assert "\x1b[96mabcdef123456\x1b[0m" in caplog.records[-1].getMessage()


def test_provider_failure_logs_structured_diagnostic_and_context(caplog):
    context = make_context()
    error = ProviderError(
        "Service unavailable",
        provider="upstream-provider",
        status_code=503,
        diagnostic=FailureDiagnostic(
            FailureCategory.TRANSPORT,
            FailureStage.REQUEST,
            "connection_error",
            provider_code="ECONNRESET",
        ),
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_provider_failure(context, error)

    assert len(caplog.records) == 1
    rendered = caplog.records[0].getMessage()
    for expected in (
        "[session abcdef123456]",
        "POST /v1/messages",
        "category=transport",
        "stage=request",
        "code=connection_error",
        "provider_code=ECONNRESET",
        "status=503",
        "model=claude-sonnet",
        "upstream=openai/gpt-5.6-sol",
        "provider=upstream-provider",
        "effort=high",
    ):
        assert expected in rendered
    assert "Service unavailable" not in rendered


def test_provider_failure_uses_request_fallback_diagnostic(caplog):
    error = ProviderError(
        "Service unavailable", provider="legacy", status_code=502
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_provider_failure(make_context(), error)

    rendered = caplog.records[0].getMessage()
    assert "category=upstream_http" in rendered
    assert "stage=request" in rendered
    assert "code=provider_error" in rendered
    assert "provider_code=" not in rendered
    assert "Service unavailable" not in rendered


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(403, False), (429, True), (503, True)],
)
def test_provider_failure_logs_retryability_from_status(
    caplog, status_code, retryable
):
    error = ProviderError(
        "Safe provider failure", provider="legacy", status_code=status_code
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_provider_failure(make_context(), error)

    assert f"retryable={retryable}" in caplog.records[0].getMessage()


def test_stream_failure_uses_stream_fallback_diagnostic(caplog):
    error = StreamError(status_code=None, retryable=False, provider="legacy")

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_stream_failure(make_context(), error)

    rendered = caplog.records[0].getMessage()
    assert "category=provider_protocol" in rendered
    assert "stage=stream" in rendered
    assert "code=stream_error" in rendered
    assert "provider_code=" not in rendered


def test_failure_log_provider_fallback_is_encoded_and_bounded(caplog):
    hostile_provider = "provider\r\n\t\x1b" + "x" * 400
    context = RequestLogContext(
        session=SessionIdentity("safe", "[session safe]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="model",
        upstream_model="upstream",
        provider=hostile_provider,
        effort="default",
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_stream_failure(context, StreamError())
        log_unexpected_failure(context, RuntimeError("not logged"))

    assert len(caplog.records) == 2
    for record in caplog.records:
        rendered = record.getMessage()
        match = re.search(r"(?:^| )provider=(\S+)", rendered)
        assert match is not None
        provider = match.group(1)
        assert len(provider) == 128
        assert provider.endswith("...")
        assert "\\x0d\\x0a\\x09\\x1b" in provider
        assert "\r" not in rendered
        assert "\n" not in rendered
        assert "\t" not in rendered
        assert "\x1b" not in rendered


def test_provider_failure_tokens_resist_field_injection(caplog):
    hostile = (
        "x status=200 retryable=False category=forged =\\'\"\u00a0"
        + "z" * 400
    )
    error = ProviderError(
        "Safe provider failure",
        provider=hostile,
        status_code=503,
        diagnostic=FailureDiagnostic(
            FailureCategory.UPSTREAM_HTTP,
            FailureStage.RESPONSE,
            "http_error",
            provider_code=hostile,
        ),
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_provider_failure(make_context(), error)

    rendered = caplog.records[0].getMessage()
    assert rendered.count(" status=") == 1
    assert rendered.count(" retryable=") == 1
    assert rendered.count(" category=") == 1
    assert "status=503" in rendered
    assert "retryable=True" in rendered
    for field in ("provider_code", "provider"):
        matches = re.findall(rf"(?:^| ){field}=(\S+)", rendered)
        assert len(matches) == 1
        value = matches[0]
        assert len(value) == 128
        assert value.endswith("...")
        for escaped in ("\\x20", "\\x3d", "\\x5c", "\\x27", "\\x22", "\\u00a0"):
            assert escaped in value
        assert "=" not in value
        assert "'" not in value
        assert '"' not in value
        assert "\u00a0" not in value


def test_stream_failure_tokens_bound_all_untrusted_structured_values(caplog):
    hostile = (
        "x status=200 retryable=False category=forged =\\'\"\u00a0"
        + "z" * 400
    )
    context = RequestLogContext(
        session=SessionIdentity("safe", "[session safe]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model=hostile,
        upstream_model=hostile,
        provider=hostile,
        effort=hostile,
    )
    error = StreamError(
        error_type=hostile,
        diagnostic=FailureDiagnostic(
            hostile,
            hostile,
            hostile,
            exception_type=hostile,
            location=hostile,
        ),
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_stream_failure(context, error)

    rendered = caplog.records[0].getMessage()
    for field in (
        "category",
        "stage",
        "code",
        "exception",
        "location",
        "error",
        "model",
        "upstream",
        "provider",
        "effort",
    ):
        matches = re.findall(rf"(?:^| ){field}=(\S+)", rendered)
        assert len(matches) == 1
        value = matches[0]
        assert len(value) == 128
        assert value.endswith("...")
        assert "\\x20status\\x3d200" in value
        assert "=" not in value
        assert "'" not in value
        assert '"' not in value
        assert "\u00a0" not in value
    assert rendered.count(" status=") == 1
    assert rendered.count(" retryable=") == 1


def test_diagnostic_text_is_encoded_and_bounded_to_128_rendered_chars(caplog):
    hostile = "prefix\r\n\t\x1b\x85\u2028\u2029\u202e" + "x" * 300
    diagnostic = FailureDiagnostic(
        hostile,
        hostile,
        hostile,
        provider_code=hostile,
    )
    error = ProviderError(
        "Service unavailable",
        provider="fake",
        status_code=503,
        diagnostic=diagnostic,
    )

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_provider_failure(make_context(), error)

    assert len(caplog.records) == 1
    rendered = caplog.records[0].getMessage()
    for field in ("category", "stage", "code", "provider_code"):
        match = re.search(rf"(?:^| ){field}=(\S+)", rendered)
        assert match is not None
        value = match.group(1)
        assert len(value) == 128
        assert value.endswith("...")
        assert "\\x0d\\x0a\\x09\\x1b\\x85\\u2028\\u2029\\u202e" in value
    assert "\r" not in rendered
    assert "\n" not in rendered
    assert "\t" not in rendered
    assert "\x1b" not in rendered
    assert len(rendered.splitlines()) == 1


def test_unexpected_failure_logs_safe_traceback_location(caplog):
    namespace = {
        "__name__": "claude_code_proxy.synthetic",
        "json": json,
    }
    exec(
        "def parse_invalid_json():\n"
        "    local_secret = 'LOCAL_SECRET_MUST_NOT_LEAK'\n"
        "    return json.loads('EXCEPTION_SECRET_MUST_NOT_LEAK')\n",
        namespace,
    )
    try:
        namespace["parse_invalid_json"]()
    except json.JSONDecodeError as error:
        with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
            log_unexpected_failure(make_context(), error)

    assert len(caplog.records) == 1
    rendered = caplog.records[0].getMessage()
    assert "category=internal" in rendered
    assert "stage=route" in rendered
    assert "code=unexpected_exception" in rendered
    assert "exception=JSONDecodeError" in rendered
    assert re.search(
        r"location=claude_code_proxy\.synthetic:parse_invalid_json:\d+",
        rendered,
    )
    assert "json.decoder" not in rendered
    assert "EXCEPTION_SECRET_MUST_NOT_LEAK" not in rendered
    assert "LOCAL_SECRET_MUST_NOT_LEAK" not in rendered
    assert "/home/" not in rendered
    assert "test_logging.py" not in rendered


def test_unexpected_failure_without_traceback_uses_stable_location(caplog):
    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        log_unexpected_failure(make_context(), RuntimeError("secret"))

    assert "location=unknown:unknown:0" in caplog.records[0].getMessage()


@pytest.mark.asyncio
async def test_observe_stream_logs_unexpected_exception_at_stream_stage(caplog):
    async def failing_events():
        raise RuntimeError("STREAM_SECRET_MUST_NOT_LEAK")
        yield

    with caplog.at_level(logging.ERROR, logger="claude_code_proxy.logging"):
        with pytest.raises(RuntimeError, match="STREAM_SECRET_MUST_NOT_LEAK"):
            await anext(observe_stream(failing_events(), make_context()))

    rendered = caplog.records[0].getMessage()
    assert "category=internal" in rendered
    assert "stage=stream" in rendered
    assert "code=unexpected_exception" in rendered
    assert "exception=RuntimeError" in rendered
    assert "location=claude_code_proxy.logging:observe_stream:" in rendered
    assert "STREAM_SECRET_MUST_NOT_LEAK" not in rendered


async def iter_events(events):
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_observe_stream_preserves_terminal_event():
    complete = StreamComplete("end_turn", TokenUsage(2, 1))

    observed = [
        event
        async for event in observe_stream(
            iter_events([TextDelta("hello"), complete]), make_context()
        )
    ]

    assert observed == [TextDelta("hello"), complete]


@pytest.mark.asyncio
async def test_observe_stream_logs_semantic_error_once_and_preserves_events(caplog):
    error = StreamError(
        error_type="api_error",
        message="Internal server error",
        status_code=502,
        retryable=True,
        provider="upstream-provider",
        diagnostic=FailureDiagnostic(
            FailureCategory.UPSTREAM_HTTP,
            FailureStage.STREAM,
            "stream_http_error",
            provider_code="overloaded",
        ),
    )
    events = [TextDelta("hello"), error]
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        observed = [
            event
            async for event in observe_stream(iter_events(events), make_context())
        ]

    assert observed == events
    assert caplog.text.count("provider stream failed") == 1
    assert "category=upstream_http" in caplog.text
    assert "stage=stream" in caplog.text
    assert "code=stream_http_error" in caplog.text
    assert "provider_code=overloaded" in caplog.text
    assert "error=api_error" in caplog.text
    assert "status=502" in caplog.text
    assert "retryable=True" in caplog.text
    assert "provider=upstream-provider" in caplog.text
    assert "Internal server error" not in caplog.text


class ClosableEvents:
    def __init__(self):
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return TextDelta("pending")

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_closing_observer_closes_upstream_iterator():
    events = ClosableEvents()
    observed = observe_stream(events, make_context())

    await anext(observed)
    await observed.aclose()

    assert events.closed is True


class CancelledEvents:
    def __init__(self):
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise asyncio.CancelledError

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_observe_stream_does_not_swallow_cancellation():
    events = CancelledEvents()

    with pytest.raises(asyncio.CancelledError):
        await anext(observe_stream(events, make_context()))

    assert events.closed is True
