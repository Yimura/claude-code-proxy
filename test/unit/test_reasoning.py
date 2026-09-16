from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from claude_code_proxy.reasoning import (
    MappingEntry,
    OutputConfig,
    ReasoningPolicy,
    ThinkingConfig,
    parse_model_mappings,
    resolve_reasoning_policy,
)


def test_thinking_requires_type_or_legacy_enabled():
    with pytest.raises(ValidationError):
        ThinkingConfig()


def test_thinking_rejects_type_and_legacy_enabled_together():
    with pytest.raises(ValidationError):
        ThinkingConfig(type="adaptive", enabled=True)


def test_enabled_thinking_requires_budget():
    with pytest.raises(ValidationError):
        ThinkingConfig(type="enabled")


@pytest.mark.parametrize("budget_tokens", [0, -1, 1, 1023, True, False])
def test_budget_tokens_reject_invalid_values(budget_tokens):
    with pytest.raises(ValidationError):
        ThinkingConfig(type="enabled", budget_tokens=budget_tokens)


def test_budget_tokens_accept_minimum():
    assert ThinkingConfig(type="enabled", budget_tokens=1024).budget_tokens == 1024


@pytest.mark.parametrize("thinking_type", ["adaptive", "disabled"])
def test_budget_is_invalid_for_non_enabled_type(thinking_type):
    with pytest.raises(ValidationError):
        ThinkingConfig(type=thinking_type, budget_tokens=1024)


@pytest.mark.parametrize("enabled", [True, False])
def test_budget_is_invalid_with_legacy_enabled(enabled):
    with pytest.raises(ValidationError):
        ThinkingConfig(enabled=enabled, budget_tokens=1024)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (ThinkingConfig(type="enabled", budget_tokens=1024), True),
        (ThinkingConfig(type="adaptive"), True),
        (ThinkingConfig(type="disabled"), False),
        (ThinkingConfig(enabled=True), True),
        (ThinkingConfig(enabled=False), False),
    ],
)
def test_is_enabled_for_current_and_legacy_forms(config, expected):
    assert config.is_enabled() is expected


def test_invalid_thinking_type_is_rejected():
    with pytest.raises(ValidationError):
        ThinkingConfig(type="automatic")


def test_output_config_preserves_extra_fields():
    config = OutputConfig(effort="high", verbosity="concise")
    assert config.effort == "high"
    assert config.model_dump()["verbosity"] == "concise"


def test_invalid_request_effort_is_rejected():
    with pytest.raises(ValidationError):
        OutputConfig(effort="xhigh")


def test_explicit_effort_overrides_mapping():
    assert resolve_reasoning_policy(
        output_config=OutputConfig(effort="low"),
        thinking=ThinkingConfig(type="disabled"),
        mapping_effort="xhigh",
    ) == ReasoningPolicy(enabled=True, effort="low")


def test_max_request_effort_maps_to_provider_high():
    assert resolve_reasoning_policy(
        output_config=OutputConfig(effort="max")
    ) == ReasoningPolicy(enabled=True, effort="high")


def test_enabled_without_mapping_defaults_to_medium():
    assert resolve_reasoning_policy(
        thinking=ThinkingConfig(type="enabled", budget_tokens=1024)
    ) == ReasoningPolicy(enabled=True, effort="medium")


def test_adaptive_uses_mapping():
    assert resolve_reasoning_policy(
        thinking=ThinkingConfig(type="adaptive"), mapping_effort="xhigh"
    ) == ReasoningPolicy(enabled=True, effort="xhigh")


@pytest.mark.parametrize(
    "thinking",
    [ThinkingConfig(type="enabled", budget_tokens=1024), ThinkingConfig(type="adaptive")],
)
def test_enabled_thinking_with_none_mapping_defaults_to_medium(thinking):
    assert resolve_reasoning_policy(
        thinking=thinking, mapping_effort="none"
    ) == ReasoningPolicy(enabled=True, effort="medium")


def test_disabled_suppresses_mapping():
    assert resolve_reasoning_policy(
        thinking=ThinkingConfig(type="disabled"), mapping_effort="high"
    ) == ReasoningPolicy(enabled=False, effort=None)


@pytest.mark.parametrize(
    ("thinking", "mapping_effort", "expected"),
    [
        (ThinkingConfig(enabled=True), None, ReasoningPolicy(True, "medium")),
        (ThinkingConfig(enabled=False), "high", ReasoningPolicy(False, None)),
        (None, "minimal", ReasoningPolicy(True, "minimal")),
        (None, None, ReasoningPolicy(None, None)),
        (None, "none", ReasoningPolicy(False, None)),
    ],
)
def test_reasoning_policy_variants(thinking, mapping_effort, expected):
    assert resolve_reasoning_policy(
        thinking=thinking, mapping_effort=mapping_effort
    ) == expected


def test_reasoning_policy_is_frozen():
    policy = ReasoningPolicy(enabled=True, effort="high")
    with pytest.raises(FrozenInstanceError):
        policy.enabled = False


def test_mapping_rejects_legacy_tier_strings():
    with pytest.raises(ValueError, match="haiku"):
        parse_model_mappings({"haiku": "small"})


def test_structured_mapping_supports_tier_and_exact_model():
    assert parse_model_mappings(
        {
            "sonnet": {"tier": "big", "effort": "medium"},
            "fable": {"model": "gpt-daybreak-blue-latest", "effort": "high"},
        }
    ) == {
        "sonnet": MappingEntry(tier="big", effort="medium"),
        "fable": MappingEntry(model="gpt-daybreak-blue-latest", effort="high"),
    }


@pytest.mark.parametrize(
    "value", [{"tier": "big", "model": "gpt-5"}, {"effort": "high"}]
)
def test_mapping_rejects_both_or_neither_selector(value):
    with pytest.raises(ValueError, match="fable") as error:
        parse_model_mappings({"fable": value})
    assert repr(value) in str(error.value)


@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        ("opus", {"tier": "big", "effort": "max"}),
    ],
)
def test_mapping_rejects_invalid_tier_or_effort(pattern, value):
    with pytest.raises(ValueError, match=pattern) as error:
        parse_model_mappings({pattern: value})
    assert repr(value) in str(error.value)
