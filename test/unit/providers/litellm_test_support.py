from dataclasses import replace
from pathlib import Path

import pytest

from claude_code_proxy.config import Settings
from claude_code_proxy.domain.models import CompletionRequest, Message, TextBlock
from claude_code_proxy.logging import RequestLogContext, SessionIdentity
from claude_code_proxy.reasoning import ReasoningPolicy


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="anthropic-key",
        openai_api_key="openai-key",
        gemini_api_key="gemini-key",
        vertex_project="project",
        vertex_location="region",
        use_vertex_auth=False,
        openai_base_url=None,
        openai_transport="litellm",
        opencode_data_dir=Path("/auth"),
        model_mapping_path=Path("mapping.json"),
    )


def log_context():
    return RequestLogContext(
        session=SessionIdentity("session", "[session session]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude",
        upstream_model="openai/gpt-5.6-sol",
        provider="litellm",
        effort="default",
    )


def request(model="openai/gpt-5.6-sol", **changes):
    base = CompletionRequest(
        original_model=model,
        model=model,
        response_model=model,
        max_tokens=20000,
        messages=(Message("user", (TextBlock("hello"),)),),
        reasoning=ReasoningPolicy(True, "high"),
    )
    return replace(base, **changes)


class FakeClient:
    def __init__(self, response=None, chunks=(), token_count=9):
        self.response = response
        self.chunks = chunks
        self.token_count = token_count
        self.counter_args = None

    def completion(self, **kwargs):
        return self.response

    async def acompletion(self, **kwargs):
        async def generate():
            for chunk in self.chunks:
                yield chunk
        return generate()

    def token_counter(self, **kwargs):
        self.counter_args = kwargs
        return self.token_count
