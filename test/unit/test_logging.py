import asyncio
import logging

import pytest

from claude_code_proxy.domain.models import (
    StreamComplete,
    StreamError,
    TextDelta,
    TokenUsage,
)
from claude_code_proxy.logging import (
    RequestLogContext,
    SessionIdentity,
    effective_effort,
    observe_stream,
    palette_index,
    session_identity,
)
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
