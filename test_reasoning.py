import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from pydantic import ValidationError

from reasoning import (
    MappingEntry,
    OutputConfig,
    ReasoningPolicy,
    ThinkingConfig,
    parse_model_mappings,
    resolve_reasoning_policy,
)


class ThinkingConfigTests(unittest.TestCase):
    def test_requires_type_or_legacy_enabled(self):
        with self.assertRaises(ValidationError):
            ThinkingConfig()

    def test_rejects_type_and_legacy_enabled_together(self):
        with self.assertRaises(ValidationError):
            ThinkingConfig(type="adaptive", enabled=True)

    def test_enabled_type_requires_budget(self):
        with self.assertRaises(ValidationError):
            ThinkingConfig(type="enabled")

    def test_budget_tokens_rejects_values_below_minimum_and_booleans(self):
        for budget_tokens in (0, -1, 1, 1023, True, False):
            with self.subTest(budget_tokens=budget_tokens):
                with self.assertRaises(ValidationError):
                    ThinkingConfig(type="enabled", budget_tokens=budget_tokens)

    def test_budget_tokens_accepts_minimum(self):
        config = ThinkingConfig(type="enabled", budget_tokens=1024)
        self.assertEqual(config.budget_tokens, 1024)

    def test_budget_is_invalid_for_non_enabled_type(self):
        for thinking_type in ("adaptive", "disabled"):
            with self.subTest(thinking_type=thinking_type):
                with self.assertRaises(ValidationError):
                    ThinkingConfig(type=thinking_type, budget_tokens=1024)

    def test_budget_is_invalid_with_legacy_enabled(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                with self.assertRaises(ValidationError):
                    ThinkingConfig(enabled=enabled, budget_tokens=1024)

    def test_is_enabled_for_current_and_legacy_forms(self):
        cases = [
            (ThinkingConfig(type="enabled", budget_tokens=1024), True),
            (ThinkingConfig(type="adaptive"), True),
            (ThinkingConfig(type="disabled"), False),
            (ThinkingConfig(enabled=True), True),
            (ThinkingConfig(enabled=False), False),
        ]
        for config, expected in cases:
            with self.subTest(config=config):
                self.assertEqual(config.is_enabled(), expected)

    def test_invalid_thinking_type_is_rejected(self):
        with self.assertRaises(ValidationError):
            ThinkingConfig(type="automatic")


class OutputConfigTests(unittest.TestCase):
    def test_preserves_extra_fields(self):
        config = OutputConfig(effort="high", verbosity="concise")
        self.assertEqual(config.effort, "high")
        self.assertEqual(config.model_dump()["verbosity"], "concise")

    def test_invalid_request_effort_is_rejected(self):
        with self.assertRaises(ValidationError):
            OutputConfig(effort="xhigh")


class ResolveReasoningPolicyTests(unittest.TestCase):
    def test_explicit_effort_overrides_mapping(self):
        self.assertEqual(
            resolve_reasoning_policy(
                output_config=OutputConfig(effort="low"),
                thinking=ThinkingConfig(type="disabled"),
                mapping_effort="xhigh",
            ),
            ReasoningPolicy(enabled=True, effort="low"),
        )

    def test_max_request_effort_maps_to_provider_high(self):
        self.assertEqual(
            resolve_reasoning_policy(output_config=OutputConfig(effort="max")),
            ReasoningPolicy(enabled=True, effort="high"),
        )

    def test_enabled_without_mapping_defaults_to_medium(self):
        self.assertEqual(
            resolve_reasoning_policy(
                thinking=ThinkingConfig(type="enabled", budget_tokens=1024)
            ),
            ReasoningPolicy(enabled=True, effort="medium"),
        )

    def test_adaptive_uses_mapping(self):
        self.assertEqual(
            resolve_reasoning_policy(
                thinking=ThinkingConfig(type="adaptive"), mapping_effort="xhigh"
            ),
            ReasoningPolicy(enabled=True, effort="xhigh"),
        )

    def test_enabled_and_adaptive_with_none_mapping_default_to_medium(self):
        configs = (
            ThinkingConfig(type="enabled", budget_tokens=1024),
            ThinkingConfig(type="adaptive"),
        )
        for thinking in configs:
            with self.subTest(thinking=thinking):
                self.assertEqual(
                    resolve_reasoning_policy(
                        thinking=thinking, mapping_effort="none"
                    ),
                    ReasoningPolicy(enabled=True, effort="medium"),
                )

    def test_adaptive_without_mapping_defaults_to_medium(self):
        self.assertEqual(
            resolve_reasoning_policy(thinking=ThinkingConfig(type="adaptive")),
            ReasoningPolicy(enabled=True, effort="medium"),
        )

    def test_disabled_suppresses_mapping(self):
        self.assertEqual(
            resolve_reasoning_policy(
                thinking=ThinkingConfig(type="disabled"), mapping_effort="high"
            ),
            ReasoningPolicy(enabled=False, effort=None),
        )

    def test_legacy_true_and_false(self):
        self.assertEqual(
            resolve_reasoning_policy(thinking=ThinkingConfig(enabled=True)),
            ReasoningPolicy(enabled=True, effort="medium"),
        )
        self.assertEqual(
            resolve_reasoning_policy(
                thinking=ThinkingConfig(enabled=False), mapping_effort="high"
            ),
            ReasoningPolicy(enabled=False, effort=None),
        )

    def test_mapping_applies_without_toggle(self):
        self.assertEqual(
            resolve_reasoning_policy(mapping_effort="minimal"),
            ReasoningPolicy(enabled=True, effort="minimal"),
        )

    def test_absent_inputs_leave_provider_default_unspecified(self):
        self.assertEqual(resolve_reasoning_policy(), ReasoningPolicy(None, None))

    def test_mapping_none_disables_reasoning(self):
        self.assertEqual(
            resolve_reasoning_policy(mapping_effort="none"),
            ReasoningPolicy(enabled=False, effort=None),
        )

    def test_reasoning_policy_is_frozen(self):
        policy = ReasoningPolicy(enabled=True, effort="high")
        with self.assertRaises(FrozenInstanceError):
            policy.enabled = False


class ModelMappingParsingTests(unittest.TestCase):
    def test_legacy_string_values_are_tiers(self):
        mappings = parse_model_mappings({"haiku": "small", "opus": "big"})
        self.assertEqual(mappings["haiku"], MappingEntry(tier="small"))
        self.assertEqual(mappings["opus"], MappingEntry(tier="big"))

    def test_structured_tier_effort_and_exact_model(self):
        mappings = parse_model_mappings(
            {
                "sonnet": {"tier": "big", "effort": "medium"},
                "fable": {
                    "model": "gpt-daybreak-blue-latest",
                    "effort": "high",
                },
            }
        )
        self.assertEqual(
            mappings["sonnet"], MappingEntry(tier="big", effort="medium")
        )
        self.assertEqual(
            mappings["fable"],
            MappingEntry(model="gpt-daybreak-blue-latest", effort="high"),
        )

    def test_rejects_both_or_neither_selector_and_names_pattern(self):
        invalid_values = (
            {"tier": "big", "model": "gpt-5"},
            {"effort": "high"},
        )
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "fable") as context:
                    parse_model_mappings({"fable": invalid_value})
                self.assertIn(repr(invalid_value), str(context.exception))

    def test_rejects_invalid_tier_and_names_pattern_and_value(self):
        invalid_value = {"tier": "large", "effort": "medium"}
        with self.assertRaisesRegex(ValueError, "sonnet") as context:
            parse_model_mappings({"sonnet": invalid_value})
        self.assertIn(repr(invalid_value), str(context.exception))

    def test_rejects_invalid_effort_and_names_pattern_and_value(self):
        invalid_value = {"tier": "big", "effort": "max"}
        with self.assertRaisesRegex(ValueError, "opus") as context:
            parse_model_mappings({"opus": invalid_value})
        self.assertIn(repr(invalid_value), str(context.exception))


class ModelMappingConfigTests(unittest.TestCase):
    def test_requires_top_level_mappings_field(self):
        from server import _parse_model_mapping_config

        for invalid_config in ({}, {"models": {}}, [], None):
            with self.subTest(invalid_config=invalid_config):
                with self.assertRaisesRegex(ValueError, "top-level 'mappings'"):
                    _parse_model_mapping_config(invalid_config)

    def test_requires_mappings_to_be_an_object(self):
        from server import _parse_model_mapping_config

        with self.assertRaisesRegex(ValueError, "model mappings must be an object"):
            _parse_model_mapping_config({"mappings": []})


class ServerModelMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server

        cls.server = server

    def test_unprefixed_exact_model_defaults_to_openai_and_ignores_big_model(self):
        mapping = {
            "fable": MappingEntry(
                model="gpt-daybreak-blue-latest", effort="high"
            )
        }
        with (
            patch.object(self.server, "MODEL_MAPPING", mapping),
            patch.object(self.server, "BIG_MODEL", "must-not-be-used"),
        ):
            self.assertEqual(
                self.server.resolve_mapped_model("claude-fable-latest"),
                ("openai/gpt-daybreak-blue-latest", True, "high"),
            )

    def test_exact_model_preserves_supported_provider_prefixes(self):
        for explicit_model in (
            "openai/gpt-5",
            "gemini/gemini-2.5-pro",
            "anthropic/claude-opus-4-1",
        ):
            with self.subTest(explicit_model=explicit_model):
                mapping = {"custom": MappingEntry(model=explicit_model)}
                with patch.object(self.server, "MODEL_MAPPING", mapping):
                    self.assertEqual(
                        self.server.resolve_mapped_model("custom-model"),
                        (explicit_model, True, None),
                    )

    def test_default_mappings_include_effort_and_fable_exact_target(self):
        defaults = self.server.DEFAULT_MODEL_MAPPING
        self.assertEqual(
            defaults["haiku"], MappingEntry(tier="small", effort="medium")
        )
        self.assertEqual(
            defaults["sonnet"], MappingEntry(tier="big", effort="medium")
        )
        self.assertEqual(
            defaults["opus"], MappingEntry(tier="big", effort="high")
        )
        self.assertEqual(
            defaults["fable"],
            MappingEntry(model="gpt-daybreak-blue-latest", effort="high"),
        )


class MessagesRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server import MessagesRequest

        cls.MessagesRequest = MessagesRequest

    def make_request(self, **extra):
        return self.MessagesRequest(
            model="claude-sonnet-test",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            **extra,
        )

    def test_current_anthropic_fields_are_accepted(self):
        request = self.make_request(
            thinking={"type": "enabled", "budget_tokens": 4096},
            output_config={
                "effort": "high",
                "format": {"type": "json_schema"},
            },
        )
        self.assertEqual(request.thinking.type, "enabled")
        self.assertEqual(request.output_config.effort, "high")
        self.assertEqual(request.mapped_effort, "medium")
        self.assertEqual(request.original_model, "claude-sonnet-test")

    def test_adaptive_disabled_and_legacy_forms_are_accepted(self):
        for thinking in (
            {"type": "adaptive"},
            {"type": "disabled"},
            {"enabled": True},
            {"enabled": False},
        ):
            with self.subTest(thinking=thinking):
                self.make_request(thinking=thinking)

    def test_invalid_effort_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.make_request(output_config={"effort": "extreme"})

    def test_internal_mapping_fields_are_excluded_from_payload(self):
        dumped = self.make_request().model_dump()
        self.assertNotIn("original_model", dumped)
        self.assertNotIn("mapped_effort", dumped)


class ProviderReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import codex_provider
        import server

        cls.codex_provider = codex_provider
        cls.server = server

    def make_request(self, **extra):
        return self.server.MessagesRequest(
            model="claude-sonnet-test",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            **extra,
        )

    def test_mapping_effort_reaches_litellm_and_codex(self):
        request = self.make_request()
        converted = self.server.convert_anthropic_to_litellm(request)
        body = self.codex_provider.build_request(request)
        self.assertEqual(converted["reasoning_effort"], "medium")
        self.assertEqual(body["reasoning"], {"effort": "medium"})

    def test_explicit_effort_overrides_mapping_in_both_paths(self):
        request = self.make_request(output_config={"effort": "high"})
        converted = self.server.convert_anthropic_to_litellm(request)
        body = self.codex_provider.build_request(request)
        self.assertEqual(converted["reasoning_effort"], "high")
        self.assertEqual(body["reasoning"], {"effort": "high"})

    def test_explicit_max_downgrades_to_high_in_both_paths(self):
        request = self.make_request(output_config={"effort": "max"})
        converted = self.server.convert_anthropic_to_litellm(request)
        body = self.codex_provider.build_request(request)
        self.assertEqual(converted["reasoning_effort"], "high")
        self.assertEqual(body["reasoning"], {"effort": "high"})

    def test_disabled_thinking_omits_reasoning_in_both_paths(self):
        request = self.make_request(thinking={"type": "disabled"})
        self.assertNotIn(
            "reasoning_effort",
            self.server.convert_anthropic_to_litellm(request),
        )
        self.assertNotIn("reasoning", self.codex_provider.build_request(request))

    def test_anthropic_passthrough_preserves_original_fields(self):
        with patch.object(self.server, "PREFERRED_PROVIDER", "anthropic"):
            request = self.make_request(
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "high",
                    "format": {"type": "json_schema"},
                },
            )
        converted = self.server.convert_anthropic_to_litellm(request)
        self.assertEqual(converted["thinking"], {"type": "adaptive"})
        self.assertEqual(
            converted["output_config"],
            {"effort": "high", "format": {"type": "json_schema"}},
        )
        self.assertNotIn("reasoning_effort", converted)


if __name__ == "__main__":
    unittest.main()
