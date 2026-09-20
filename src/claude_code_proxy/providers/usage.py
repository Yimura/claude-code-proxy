"""Normalize inclusive provider token counts into Anthropic usage semantics."""

from collections.abc import Mapping
from typing import Any

from ..domain.models import TokenUsage, UsageField

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


def _first_observed_count(
    source: object, names: tuple[str, ...]
) -> tuple[int | None, bool]:
    for name in names:
        count = _count(_value(source, name))
        if count is not None:
            return count, True
    return None, False


def _first_value(source: object, names: tuple[str, ...]) -> object | None:
    for name in names:
        value = _value(source, name)
        if value is not _MISSING and value is not None:
            return value
    return None


def normalize_usage(usage: object) -> TokenUsage:
    observed_fields: set[UsageField] = set()

    inclusive_input, input_observed = _first_observed_count(
        usage, ("input_tokens", "prompt_tokens")
    )
    if input_observed:
        observed_fields.add("input_tokens")

    inclusive_output, output_observed = _first_observed_count(
        usage, ("output_tokens", "completion_tokens")
    )
    if output_observed:
        observed_fields.add("output_tokens")

    input_details = _first_value(
        usage, ("input_tokens_details", "prompt_tokens_details")
    )
    output_details = _first_value(
        usage, ("output_tokens_details", "completion_tokens_details")
    )

    cache_read, cache_read_observed = _first_observed_count(
        usage, ("cache_read_input_tokens",)
    )
    if not cache_read_observed:
        cache_read, cache_read_observed = _first_observed_count(
            input_details, ("cached_tokens",)
        )
    if not cache_read_observed:
        cache_read, cache_read_observed = _first_observed_count(
            usage, ("_cache_read_input_tokens",)
        )
        cache_read_observed = bool(
            cache_read_observed and cache_read is not None and cache_read > 0
        )
    if cache_read_observed and input_observed:
        observed_fields.add("cache_read_input_tokens")

    cache_creation, cache_creation_observed = _first_observed_count(
        usage, ("cache_creation_input_tokens",)
    )
    if not cache_creation_observed:
        cache_creation, cache_creation_observed = _first_observed_count(
            input_details,
            (
                "cache_write_tokens",
                "cache_creation_tokens",
                "cache_creation_input_tokens",
            ),
        )
    if not cache_creation_observed:
        cache_creation, cache_creation_observed = _first_observed_count(
            usage, ("_cache_creation_input_tokens",)
        )
        cache_creation_observed = bool(
            cache_creation_observed
            and cache_creation is not None
            and cache_creation > 0
        )
    if cache_creation_observed and input_observed:
        observed_fields.add("cache_creation_input_tokens")

    thinking_tokens, thinking_observed = _first_observed_count(
        output_details, ("reasoning_tokens",)
    )
    if not thinking_observed:
        thinking_tokens, thinking_observed = _first_observed_count(
            usage, ("reasoning_tokens",)
        )
    if thinking_observed and output_observed:
        observed_fields.add("thinking_tokens")

    inclusive_input = inclusive_input or 0
    inclusive_output = inclusive_output or 0
    cache_read = cache_read or 0
    cache_creation = cache_creation or 0

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
        observed_fields=frozenset(observed_fields),
    )
