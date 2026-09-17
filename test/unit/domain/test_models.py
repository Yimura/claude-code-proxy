from dataclasses import FrozenInstanceError
import pytest
from claude_code_proxy.domain.models import (
    CompletionRequest,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    TokenUsage,
)
from claude_code_proxy.reasoning import ReasoningPolicy


def test_completion_request_is_immutable():
    request = CompletionRequest(
        original_model="claude-sonnet",
        model="openai/gpt-5.6-sol",
        response_model="claude-sonnet[1m]",
        max_tokens=100,
        messages=(Message(role="user", content=(TextBlock("hello"),)),),
        reasoning=ReasoningPolicy(None, None),
        session_id="session-1",
    )
    assert request.original_model == "claude-sonnet"
    assert request.model == "openai/gpt-5.6-sol"
    assert request.response_model == "claude-sonnet[1m]"
    assert request.session_id == "session-1"
    with pytest.raises(FrozenInstanceError):
        request.model = "openai/gpt-5"


def test_token_usage_defaults_cache_counts_to_zero():
    usage = TokenUsage(input_tokens=4, output_tokens=2)
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_read_input_tokens == 0


def test_redacted_thinking_block_is_immutable():
    block = RedactedThinkingBlock("codex-reasoning-v1:data")
    assert block.data == "codex-reasoning-v1:data"
    with pytest.raises(FrozenInstanceError):
        block.data = "changed"


def test_token_usage_preserves_reported_thinking_tokens():
    usage = TokenUsage(4, 10, 1, 3, thinking_tokens=6)
    assert usage.thinking_tokens == 6
