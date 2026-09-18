from dataclasses import FrozenInstanceError
import pytest
from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionRequest,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    TokenUsage,
)
from claude_code_proxy.reasoning import ReasoningPolicy


def test_completion_request_is_immutable():
    identity = ClientIdentity(
        session_id="session-1",
        agent_id="agent-1",
        parent_agent_id="parent-1",
    )
    request = CompletionRequest(
        original_model="claude-sonnet",
        model="openai/gpt-5.6-sol",
        response_model="claude-sonnet[1m]",
        max_tokens=100,
        messages=(Message(role="user", content=(TextBlock("hello"),)),),
        reasoning=ReasoningPolicy(None, None),
        client_identity=identity,
    )
    assert request.original_model == "claude-sonnet"
    assert request.model == "openai/gpt-5.6-sol"
    assert request.response_model == "claude-sonnet[1m]"
    assert request.client_identity is identity
    with pytest.raises(FrozenInstanceError):
        request.model = "openai/gpt-5"


def test_client_identity_is_frozen_and_hides_raw_values():
    identity = ClientIdentity(
        session_id="raw-session",
        agent_id="raw-agent",
        parent_agent_id="raw-parent",
    )

    assert "raw-session" not in repr(identity)
    assert "raw-agent" not in repr(identity)
    assert "raw-parent" not in repr(identity)
    with pytest.raises(FrozenInstanceError):
        identity.agent_id = "changed"


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
