import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from claude_code_proxy.domain.models import StreamError, TextDelta
from claude_code_proxy.logging import (
    RequestLogContext,
    SessionIdentity,
    SessionTracker,
    effective_effort,
    observe_stream,
    palette_index,
)
from claude_code_proxy.reasoning import ReasoningPolicy


class Stream:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def make_context() -> RequestLogContext:
    return RequestLogContext(
        session=SessionIdentity("abcdef12", "[session abcdef12]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude-sonnet",
        upstream_model="openai/gpt-5.6-sol",
        provider="fake",
        effort="high",
    )


def test_palette_index_is_stable_and_bounded():
    first = palette_index("session-123")
    assert first == palette_index("session-123")
    assert 0 <= first < 5


def test_session_label_is_colored_for_non_tty_log_pipe():
    identity = SessionTracker(Stream(False), environ={}).observe("abcdef123456")
    assert identity.rendered.startswith("[session \033[")
    assert identity.rendered.endswith("abcdef12\033[0m]")


def test_session_label_is_colored_for_tty():
    identity = SessionTracker(Stream(True), environ={}).observe("abcdef123456")
    assert identity.rendered.startswith("[session \033[")
    assert identity.rendered.endswith("abcdef12\033[0m]")


def test_no_color_disables_color_for_non_tty_log_pipe():
    identity = SessionTracker(Stream(False), environ={"NO_COLOR": "1"}).observe(
        "abcdef123456"
    )
    assert identity.rendered == "[session abcdef12]"


def test_first_observation_marks_session_new_once():
    tracker = SessionTracker(Stream(False), environ={})
    assert tracker.observe("session-123").is_new is True
    assert tracker.observe("session-123").is_new is False


def test_concurrent_observation_marks_session_new_once():
    tracker = SessionTracker(Stream(False), environ={})
    with ThreadPoolExecutor(max_workers=8) as pool:
        observations = list(pool.map(tracker.observe, ["session-123"] * 32))
    assert sum(item.is_new for item in observations) == 1


def test_missing_session_uses_unique_unregistered_request_labels():
    tracker = SessionTracker(Stream(False), environ={})
    first = tracker.observe(None)
    second = tracker.observe("")
    assert first.label.startswith("req-")
    assert second.label.startswith("req-")
    assert first.label != second.label
    assert first.is_new is False
    assert second.is_new is False


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
async def test_observe_stream_logs_semantic_error_and_preserves_events(caplog):
    events = [TextDelta("hello"), StreamError("upstream failed")]
    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        observed = [
            event async for event in observe_stream(iter_events(events), make_context())
        ]
    assert observed == events
    assert caplog.text.count("provider stream failed") == 1
    assert "upstream failed" not in caplog.text
