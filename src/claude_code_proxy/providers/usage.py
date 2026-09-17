"""Normalize inclusive provider token counts into Anthropic usage semantics."""

from collections.abc import Mapping
from typing import Any

from ..domain.models import TokenUsage

_MISSING = object()


def _value(source: object, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, _MISSING)
    try:
        return getattr(source, name)
    except (AttributeError, TypeError):
        return _MISSING


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _first_count(source: object, names: tuple[str, ...]) -> int | None:
    for name in names:
        count = _count(_value(source, name))
        if count is not None:
            return count
    return None


def _first_value(source: object, names: tuple[str, ...]) -> object | None:
    for name in names:
        value = _value(source, name)
        if value is not _MISSING and value is not None:
            return value
    return None


def normalize_usage(usage: object) -> TokenUsage:
    inclusive_input = _first_count(usage, ("input_tokens", "prompt_tokens")) or 0
    inclusive_output = _first_count(
        usage, ("output_tokens", "completion_tokens")
    ) or 0
    input_details = _first_value(
        usage, ("input_tokens_details", "prompt_tokens_details")
    )
    output_details = _first_value(
        usage, ("output_tokens_details", "completion_tokens_details")
    )

    cache_read = _first_count(usage, ("cache_read_input_tokens",))
    if cache_read is None:
        cache_read = _first_count(input_details, ("cached_tokens",))
    if cache_read is None:
        private_read = _first_count(usage, ("_cache_read_input_tokens",))
        cache_read = private_read if private_read and private_read > 0 else 0

    cache_creation = _first_count(usage, ("cache_creation_input_tokens",))
    if cache_creation is None:
        cache_creation = _first_count(
            input_details,
            (
                "cache_write_tokens",
                "cache_creation_tokens",
                "cache_creation_input_tokens",
            ),
        )
    if cache_creation is None:
        private_creation = _first_count(
            usage, ("_cache_creation_input_tokens",)
        )
        cache_creation = (
            private_creation if private_creation and private_creation > 0 else 0
        )

    thinking_tokens = _first_count(output_details, ("reasoning_tokens",))
    if thinking_tokens is None:
        thinking_tokens = _first_count(usage, ("reasoning_tokens",))

    cache_read = min(cache_read, inclusive_input)
    remaining = inclusive_input - cache_read
    cache_creation = min(cache_creation, remaining)
    uncached_input = remaining - cache_creation
    if thinking_tokens is not None:
        thinking_tokens = min(thinking_tokens, inclusive_output)

    return TokenUsage(
        input_tokens=uncached_input,
        output_tokens=inclusive_output,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        thinking_tokens=thinking_tokens,
    )
