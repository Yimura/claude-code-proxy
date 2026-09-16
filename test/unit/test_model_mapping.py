import pytest
from claude_code_proxy.model_mapping import ModelResolver, ResolvedModel
from claude_code_proxy.reasoning import MappingEntry


def resolver(mappings=None, preferred="openai", big="gpt-big", small="gpt-small"):
    return ModelResolver(mappings or {}, preferred, big, small)


def test_unprefixed_exact_target_defaults_to_openai():
    result = resolver({"fable": MappingEntry(model="gpt-daybreak-blue-latest", effort="high")}).resolve("claude-fable-latest")
    assert result == ResolvedModel("claude-fable-latest", "openai/gpt-daybreak-blue-latest", True, "high")


@pytest.mark.parametrize("target", ["openai/gpt-5", "gemini/gemini-2.5-pro", "anthropic/claude-opus-5"])
def test_exact_target_preserves_supported_prefix(target):
    assert resolver({"custom": MappingEntry(model=target)}).resolve("custom-model").model == target


def test_google_tier_uses_known_gemini_model():
    result = resolver({"sonnet": MappingEntry(tier="big")}, preferred="google", big="gemini-2.5-pro").resolve("claude-sonnet")
    assert result.model == "gemini/gemini-2.5-pro"


def test_google_tier_falls_back_to_openai_for_unknown_model():
    result = resolver({"sonnet": MappingEntry(tier="big")}, preferred="google", big="custom").resolve("claude-sonnet")
    assert result.model == "openai/custom"


def test_anthropic_preference_bypasses_mappings_and_strips_prefix():
    result = resolver({"sonnet": MappingEntry(tier="big", effort="high")}, preferred="anthropic").resolve("openai/claude-sonnet")
    assert result == ResolvedModel("openai/claude-sonnet", "anthropic/claude-sonnet", True, None)


@pytest.mark.parametrize(("submitted", "expected"), [("gpt-5.6-sol", "openai/gpt-5.6-sol"), ("gemini-2.5-pro", "gemini/gemini-2.5-pro"), ("openai/gpt-5.6-sol", "openai/gpt-5.6-sol"), ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-pro")])
def test_known_models_infer_or_preserve_prefix(submitted, expected):
    assert resolver().resolve(submitted).model == expected


def test_unknown_model_is_unchanged():
    assert resolver().resolve("unknown") == ResolvedModel("unknown", "unknown", False, None)
