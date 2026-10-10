"""Recognize an upstream context-length rejection without forwarding its text.

Anthropic reports an oversized prompt as ``prompt is too long: N tokens > M
maximum``, and Claude Code keys its prompt-too-long recovery on that phrase.
OpenAI-protocol upstreams word the same failure in many ways, and LiteLLM often
drops their structured ``context_length_exceeded`` code. This module matches the
wording and rebuilds the Anthropic message from the two token counts alone, so
no other upstream text reaches the client or the logs.
"""

import re

import litellm

PROMPT_TOO_LONG = "prompt is too long"
CONTEXT_LENGTH_CODE = "context_length_exceeded"

_CONTEXT_STATUSES = frozenset({400, 413})
_MARKERS = re.compile(
    r"prompt is too long|context[ _]length|context window|maximum context",
    re.IGNORECASE,
)
_REQUESTED = re.compile(
    r"(?:request(?:ed)?\s+is|you\s+requested|resulted\s+in|prompt\s+(?:is|has|contains))"
    r"\s+(\d+)\s+(?:prompt\s+|input\s+)?tokens",
    re.IGNORECASE,
)
_MAXIMUM = re.compile(
    r"(?:at\s+most|maximum\s+context\s+length\s+is|maximum\s+of|limit\s+of)"
    r"\s+(\d+)\s+(?:prompt\s+|input\s+)?tokens",
    re.IGNORECASE,
)


def is_context_length_error(error: Exception, status_code: int | None) -> bool:
    if isinstance(error, litellm.ContextWindowExceededError):
        return True
    if status_code not in _CONTEXT_STATUSES:
        return False
    if getattr(error, "code", None) == CONTEXT_LENGTH_CODE:
        return True
    return _MARKERS.search(str(error)) is not None


def prompt_too_long_message(error: Exception) -> str:
    """Return Anthropic's wording, carrying only the upstream token counts."""
    text = str(error)
    requested = _REQUESTED.search(text)
    maximum = _MAXIMUM.search(text)
    if requested is None or maximum is None:
        return PROMPT_TOO_LONG
    return (
        f"{PROMPT_TOO_LONG}: {int(requested.group(1))} tokens > "
        f"{int(maximum.group(1))} maximum"
    )
