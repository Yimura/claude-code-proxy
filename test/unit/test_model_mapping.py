import pytest

from claude_code_proxy.config import ModelConfig
from claude_code_proxy.model_mapping import ModelResolver, ResolvedModel
from claude_code_proxy.reasoning import MappingEntry


def resolver(mappings=None, tiers=None):
    return ModelResolver(ModelConfig(tiers or {}, mappings or {}))


def test_tier_uses_file_defined_target():
    result = resolver(
        {"sonnet": MappingEntry(tier="big", effort="high")},
        {"big": "gemini/gemini-2.5-pro"},
    ).resolve("claude-sonnet")

    assert result == ResolvedModel(
        "claude-sonnet", "gemini/gemini-2.5-pro", True, "high"
    )


def test_unprefixed_tier_target_defaults_to_openai():
    result = resolver(
        {"sonnet": MappingEntry(tier="big")}, {"big": "gpt-5.6-sol"}
    ).resolve("claude-sonnet")

    assert result.model == "openai/gpt-5.6-sol"


def test_unprefixed_exact_target_defaults_to_openai():
    result = resolver({
        "fable": MappingEntry(model="gpt-daybreak-blue-latest", effort="high")
    }).resolve("claude-fable-latest")

    assert result == ResolvedModel(
        "claude-fable-latest", "openai/gpt-daybreak-blue-latest", True, "high"
    )


@pytest.mark.parametrize(
    "target",
    ["openai/gpt-5", "gemini/gemini-2.5-pro", "anthropic/claude-opus-5"],
)
def test_exact_target_preserves_supported_prefix(target):
    assert resolver({"custom": MappingEntry(model=target)}).resolve(
        "custom-model"
    ).model == target


@pytest.mark.parametrize(
    ("submitted", "expected"),
    [
        ("gpt-5.6-sol", "openai/gpt-5.6-sol"),
        ("gemini-2.5-pro", "gemini/gemini-2.5-pro"),
        ("openai/gpt-5.6-sol", "openai/gpt-5.6-sol"),
        ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-pro"),
    ],
)
def test_known_models_infer_or_preserve_prefix(submitted, expected):
    assert resolver().resolve(submitted).model == expected


def test_unknown_model_is_unchanged():
    assert resolver().resolve("unknown") == ResolvedModel(
        "unknown", "unknown", False, None
    )
