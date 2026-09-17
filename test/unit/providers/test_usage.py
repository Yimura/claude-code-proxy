from types import SimpleNamespace

import pytest
from litellm.types.utils import Usage

from claude_code_proxy.domain.models import TokenUsage
from claude_code_proxy.providers.usage import normalize_usage


def test_normalizes_codex_usage():
    raw = {
        "input_tokens": 100,
        "input_tokens_details": {
            "cached_tokens": 60,
            "cache_write_tokens": 10,
        },
        "output_tokens": 20,
        "output_tokens_details": {"reasoning_tokens": 7},
    }

    assert normalize_usage(raw) == TokenUsage(30, 20, 10, 60, 7)


def test_normalizes_litellm_usage_object():
    raw = Usage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details={
            "cached_tokens": 60,
            "cache_write_tokens": 10,
        },
        completion_tokens_details={"reasoning_tokens": 7},
    )

    assert normalize_usage(raw) == TokenUsage(30, 20, 10, 60, 7)


def test_normalizes_public_top_level_cache_aliases():
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cache_read_input_tokens": 60,
        "cache_creation_input_tokens": 10,
        "reasoning_tokens": 7,
    }

    assert normalize_usage(raw) == TokenUsage(30, 20, 10, 60, 7)


def test_normalizes_positive_private_cache_aliases():
    raw = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        _cache_read_input_tokens=60,
        _cache_creation_input_tokens=10,
    )

    assert normalize_usage(raw) == TokenUsage(30, 20, 10, 60)


def test_explicit_zero_reasoning_is_preserved():
    raw = {
        "input_tokens": 4,
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
    }

    assert normalize_usage(raw) == TokenUsage(4, 2, thinking_tokens=0)


def test_missing_reasoning_remains_unreported():
    assert normalize_usage({"input_tokens": 4, "output_tokens": 2}) == TokenUsage(4, 2)


def test_clamps_inconsistent_subsets_and_conserves_input():
    usage = normalize_usage({
        "input_tokens": 10,
        "input_tokens_details": {
            "cached_tokens": 8,
            "cache_write_tokens": 7,
        },
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 9},
    })

    assert usage == TokenUsage(0, 5, 2, 8, 5)
    assert (
        usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    ) == 10


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "2", None])
def test_invalid_counts_are_ignored(invalid):
    usage = normalize_usage({
        "input_tokens": invalid,
        "prompt_tokens": 9,
        "output_tokens": invalid,
        "completion_tokens": 3,
        "input_tokens_details": {
            "cached_tokens": invalid,
            "cache_write_tokens": invalid,
        },
        "output_tokens_details": {"reasoning_tokens": invalid},
    })

    assert usage == TokenUsage(9, 3)


def test_missing_usage_returns_zero_totals():
    assert normalize_usage(None) == TokenUsage(0, 0)
