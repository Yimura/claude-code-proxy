from dataclasses import FrozenInstanceError
import pytest
from claude_code_proxy.domain.models import CompletionRequest, Message, TextBlock, TokenUsage
from claude_code_proxy.reasoning import ReasoningPolicy


def test_completion_request_is_immutable():
    request = CompletionRequest(
        original_model="claude-sonnet",
        model="claude-sonnet",
        max_tokens=100,
        messages=(Message(role="user", content=(TextBlock("hello"),)),),
        reasoning=ReasoningPolicy(None, None),
    )
    with pytest.raises(FrozenInstanceError):
        request.model = "openai/gpt-5"


def test_token_usage_defaults_cache_counts_to_zero():
    usage = TokenUsage(input_tokens=4, output_tokens=2)
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_read_input_tokens == 0
