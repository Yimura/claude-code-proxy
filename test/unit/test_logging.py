import asyncio
import logging

import pytest

from claude_code_proxy.domain.models import (
    ClientIdentity,
    StreamComplete,
    StreamError,
    TextDelta,
    TokenUsage,
)
from claude_code_proxy.logging import (
    AgentIdentity,
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


def test_configure_logging_exposes_readiness_info_and_keeps_uvicorn_quiet():
    configure_logging()

    assert logging.getLogger(
        "claude_code_proxy.logging.readiness"
    ).isEnabledFor(logging.INFO)
    assert not logging.getLogger("uvicorn").isEnabledFor(logging.INFO)
    assert not logging.getLogger("uvicorn.access").isEnabledFor(logging.INFO)
    assert not logging.getLogger("uvicorn.error").isEnabledFor(logging.INFO)


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
        log_provider_failure(context, 503)
        log_stream_failure(
            context,
            StreamError(error_type=hostile, provider=hostile),
        )
        log_unexpected_failure(context, hostile)

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
        retryable=True,
        diagnostic="upstream failed",
    )
    events = [TextDelta("hello"), error]
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        observed = [
            event
            async for event in observe_stream(iter_events(events), make_context())
        ]

    assert observed == events
    assert caplog.text.count("provider stream failed") == 1
    assert "error=api_error" in caplog.text
    assert "retryable=True" in caplog.text
    assert "upstream failed" not in caplog.text


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
