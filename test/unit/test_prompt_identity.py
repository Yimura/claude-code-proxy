from claude_code_proxy.domain.models import TextBlock
from claude_code_proxy.prompt_identity import reconcile_system_identity

HARNESS_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
MODEL_IDENTITY = (
    "You are powered by the model named Opus 5. "
    "The exact model ID is claude-opus-5."
)
HARNESS_REPLACEMENT = (
    "You are running inside Claude Code, Anthropic's coding-agent CLI harness."
)


def reconcile(target: str, *, system=None, original="claude-opus-5", mapped=True):
    blocks = system or (
        TextBlock(
            f"{HARNESS_IDENTITY}\n{MODEL_IDENTITY} "
            "Assistant knowledge cutoff is May 2026."
        ),
    )
    return reconcile_system_identity(blocks, original, target, mapped)


def test_openai_mapping_names_actual_model_and_preserves_suffix():
    result = reconcile("openai/gpt-5.6-sol")

    assert result == (
        TextBlock(
            f"{HARNESS_REPLACEMENT}\n"
            "The model generating this response is openai/gpt-5.6-sol, "
            "not an Anthropic Claude model. "
            "Assistant knowledge cutoff is May 2026."
        ),
    )


def test_gemini_mapping_names_actual_non_anthropic_model():
    result = reconcile("gemini/gemini-2.5-pro")

    assert (
        "The model generating this response is gemini/gemini-2.5-pro, "
        "not an Anthropic Claude model."
    ) in result[0].text


def test_anthropic_mapping_names_actual_claude_model():
    result = reconcile("anthropic/claude-sonnet-5")

    assert (
        "The model generating this response is anthropic/claude-sonnet-5."
        in result[0].text
    )
    assert "not an Anthropic Claude model" not in result[0].text


def test_unmapped_request_returns_original_tuple():
    system = (TextBlock(f"{HARNESS_IDENTITY}\n{MODEL_IDENTITY}"),)

    assert reconcile("claude-opus-5", system=system, mapped=False) is system


def test_unrecognized_system_text_returns_original_tuple():
    system = (TextBlock("Compare Claude Opus with Sonnet, Haiku, and Fable."),)

    assert reconcile("openai/gpt-5.6-sol", system=system) is system


def test_mismatched_embedded_model_id_returns_original_tuple():
    system = (
        TextBlock(
            f"{HARNESS_IDENTITY}\nYou are powered by the model named Sonnet 5. "
            "The exact model ID is claude-sonnet-5."
        ),
    )

    assert reconcile("openai/gpt-5.6-sol", system=system) is system


def test_model_identity_without_harness_does_not_invent_harness_context():
    system = (TextBlock(f"Policy.\n{MODEL_IDENTITY}\nRemain concise."),)

    result = reconcile("openai/gpt-5.6-sol", system=system)

    assert result == (
        TextBlock(
            "Policy.\nThe model generating this response is "
            "openai/gpt-5.6-sol, not an Anthropic Claude model.\n"
            "Remain concise."
        ),
    )
    assert "running inside Claude Code" not in result[0].text


def test_multiple_blocks_preserve_boundaries_and_unrelated_text():
    system = (
        TextBlock(f"prefix\n{HARNESS_IDENTITY}\nsuffix"),
        TextBlock(f"{MODEL_IDENTITY} Assistant knowledge cutoff is May 2026."),
        TextBlock("Keep Opus examples unchanged."),
    )

    result = reconcile("openai/gpt-5.6-sol", system=system)

    assert len(result) == 3
    assert result[0] == TextBlock(f"prefix\n{HARNESS_REPLACEMENT}\nsuffix")
    assert result[1] == TextBlock(
        "The model generating this response is openai/gpt-5.6-sol, "
        "not an Anthropic Claude model. Assistant knowledge cutoff is May 2026."
    )
    assert result[2] is system[2]


def test_model_identity_split_across_blocks_is_unchanged():
    system = (
        TextBlock("You are powered by the model named Opus 5."),
        TextBlock("The exact model ID is claude-opus-5."),
    )

    assert reconcile("openai/gpt-5.6-sol", system=system) is system
