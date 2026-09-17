import pytest

from claude_code_proxy.config import ModelConfig, ModelDefinition
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.reasoning import MappingEntry


def model(target, context_window=None):
    return ModelDefinition(target=target, context_window=context_window)


def resolver(mappings=None, tiers=None, models=None):
    return ModelResolver(
        ModelConfig(models or {}, tiers or {}, mappings or {})
    )


def test_tier_uses_model_definition_target_and_capability():
    result = resolver(
        {"sonnet": MappingEntry(tier="big", effort="high")},
        {"big": "gemini-big"},
        {"gemini-big": model("gemini/gemini-big", 1_000_000)},
    ).resolve("claude-sonnet")

    assert result.original == "claude-sonnet"
    assert result.model == "gemini/gemini-big"
    assert result.response_model == "claude-sonnet[1m]"
    assert result.mapped is True
    assert result.effort == "high"


def test_unprefixed_definition_target_defaults_to_openai():
    result = resolver(
        {"sonnet": MappingEntry(tier="big")},
        {"big": "sol"},
        {"sol": model("gpt-5.6-sol", 1_000_000)},
    ).resolve("claude-sonnet")

    assert result.model == "openai/gpt-5.6-sol"
    assert result.response_model == "claude-sonnet[1m]"


def test_direct_mapping_uses_model_definition():
    result = resolver(
        {"fable": MappingEntry(model="custom", effort="high")},
        models={"custom": model("gpt-custom-large", 1_000_000)},
    ).resolve("claude-fable-latest")

    assert result.original == "claude-fable-latest"
    assert result.model == "openai/gpt-custom-large"
    assert result.response_model == "claude-fable-latest[1m]"
    assert result.mapped is True
    assert result.effort == "high"


@pytest.mark.parametrize(
    "target",
    ["openai/gpt-5", "gemini/gemini-2.5-pro", "anthropic/claude-opus-5"],
)
def test_definition_target_preserves_supported_prefix(target):
    result = resolver(
        {"custom": MappingEntry(model="target")},
        models={"target": model(target)},
    ).resolve("custom-model")

    assert result.model == target
    assert result.response_model == target


def test_existing_1m_suffix_is_not_duplicated():
    result = resolver(
        {"opus": MappingEntry(model="sol")},
        models={"sol": model("openai/gpt-5.6-sol", 1_000_000)},
    ).resolve("claude-opus-5[1m]")

    assert result.response_model == "claude-opus-5[1m]"


def test_smaller_context_removes_1m_suffix():
    result = resolver(
        {"opus": MappingEntry(model="small")},
        models={"small": model("openai/gpt-small", 200_000)},
    ).resolve("claude-opus-5[1m]")

    assert result.response_model == "claude-opus-5"


def test_unknown_context_preserves_current_upstream_response_identity():
    result = resolver(
        {"opus": MappingEntry(model="unknown")},
        models={"unknown": model("openai/gpt-unknown")},
    ).resolve("claude-opus-5")

    assert result.response_model == "openai/gpt-unknown"


@pytest.mark.parametrize(
    ("submitted", "expected"),
    [
        ("gpt-5.6-sol", "openai/gpt-5.6-sol"),
        ("gemini-2.5-pro", "gemini/gemini-2.5-pro"),
        ("openai/gpt-5.6-sol", "openai/gpt-5.6-sol"),
        ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-pro"),
    ],
)
def test_known_direct_models_infer_or_preserve_prefix(submitted, expected):
    result = resolver().resolve(submitted)

    assert result.model == expected
    assert result.response_model == expected


def test_unknown_model_is_unchanged():
    result = resolver().resolve("unknown")

    assert result.original == "unknown"
    assert result.model == "unknown"
    assert result.response_model == "unknown"
    assert result.mapped is False
    assert result.effort is None
