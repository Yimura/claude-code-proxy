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


def _first_count(source: object, names: tuple[str, ...]) -> int | None:
    for name in names:
        count = _count(_value(source, name))
        if count is not None:
            return count
    return None


def _resolve_count(
    candidates: tuple[tuple[object, tuple[str, ...], bool], ...]
) -> int | None:
    for source, names, positive_only in candidates:
        count = _first_count(source, names)
        if count is not None and (not positive_only or count > 0):
            return count
    return None


def _first_value(source: object, names: tuple[str, ...]) -> object | None:
    for name in names:
        value = _value(source, name)
        if value is not _MISSING and value is not None:
            return value
    return None


def _normalize_input_usage(
    usage: object,
) -> tuple[int, int, int, set[UsageField]]:
    inclusive_input = _first_count(usage, ("input_tokens", "prompt_tokens"))
    input_details = _first_value(
        usage, ("input_tokens_details", "prompt_tokens_details")
    )
    cache_read = _resolve_count((
        (usage, ("cache_read_input_tokens",), False),
        (input_details, ("cached_tokens",), False),
        (usage, ("_cache_read_input_tokens",), True),
    ))
    cache_creation = _resolve_count((
        (usage, ("cache_creation_input_tokens",), False),
        (
            input_details,
            (
                "cache_write_tokens",
                "cache_creation_tokens",
                "cache_creation_input_tokens",
            ),
            False,
        ),
        (usage, ("_cache_creation_input_tokens",), True),
    ))

    input_tokens = inclusive_input or 0
    cache_read_tokens = min(cache_read or 0, input_tokens)
    remaining = input_tokens - cache_read_tokens
    cache_creation_tokens = min(cache_creation or 0, remaining)
    observed_fields: set[UsageField] = set()
    if inclusive_input is not None:
        observed_fields.add("input_tokens")
        if cache_read is not None:
            observed_fields.add("cache_read_input_tokens")
        if cache_creation is not None:
            observed_fields.add("cache_creation_input_tokens")

    return (
        remaining - cache_creation_tokens,
        cache_creation_tokens,
        cache_read_tokens,
        observed_fields,
    )


def _normalize_output_usage(
    usage: object,
) -> tuple[int, int | None, set[UsageField]]:
    inclusive_output = _first_count(
        usage, ("output_tokens", "completion_tokens")
    )
    output_details = _first_value(
        usage, ("output_tokens_details", "completion_tokens_details")
    )
    thinking_tokens = _resolve_count((
        (output_details, ("reasoning_tokens",), False),
        (usage, ("reasoning_tokens",), False),
    ))

    output_tokens = inclusive_output or 0
    if thinking_tokens is not None:
        thinking_tokens = min(thinking_tokens, output_tokens)
    observed_fields: set[UsageField] = set()
    if inclusive_output is not None:
        observed_fields.add("output_tokens")
        if thinking_tokens is not None:
            observed_fields.add("thinking_tokens")

    return output_tokens, thinking_tokens, observed_fields


def normalize_usage(usage: object) -> TokenUsage:
    (
        input_tokens,
        cache_creation_tokens,
        cache_read_tokens,
        input_fields,
    ) = _normalize_input_usage(usage)
    output_tokens, thinking_tokens, output_fields = _normalize_output_usage(usage)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_tokens,
        cache_read_input_tokens=cache_read_tokens,
        thinking_tokens=thinking_tokens,
        observed_fields=frozenset(input_fields | output_fields),
    )
