"""Tests for the zai backend's built-in WebSearch tool translation."""

from dataclasses import replace
from pathlib import Path

from claude_code_proxy.config import Settings
from claude_code_proxy.domain.models import (
    CompletionRequest,
    Message,
    TextBlock,
    ToolDefinition,
)
from claude_code_proxy.providers.litellm import LiteLLMProvider
from claude_code_proxy.reasoning import ReasoningPolicy


def settings():
    return Settings(
        anthropic_api_key="anthropic-key",
        openai_api_key="openai-key",
        gemini_api_key="gemini-key",
        vertex_project="project",
        vertex_location="region",
        use_vertex_auth=False,
        openai_base_url="https://api.z.ai/api/coding/paas/v4",
        openai_transport="litellm",
        opencode_data_dir=Path("/auth"),
        model_mapping_path=Path("mapping.json"),
    )


WEB_SEARCH_TOOL = ToolDefinition(
    "WebSearch", "search the web", {"type": "object", "properties": {}}
)
FETCH_TOOL = ToolDefinition(
    "WebFetch", "fetch a page", {"type": "object", "properties": {}}
)


def request(backend="zai", tools=(WEB_SEARCH_TOOL,), **changes):
    base = CompletionRequest(
        original_model="openai/glm-5.3",
        model="openai/glm-5.3",
        response_model="openai/glm-5.3",
        max_tokens=2000,
        messages=(Message("user", (TextBlock("price?"),)),),
        reasoning=ReasoningPolicy(True, "medium"),
        backend=backend,
        tools=tuple(tools),
    )
    return replace(base, **changes)


def test_zai_replaces_client_web_search_with_builtin_tool():
    payload = LiteLLMProvider(settings()).build_request(request(), stream=False)

    assert payload["tools"] == [
        {"type": "web_search", "web_search": {"enable": True, "search_engine": "search-prime"}}
    ]


def test_zai_keeps_other_tools_alongside_builtin_search():
    payload = LiteLLMProvider(settings()).build_request(
        request(tools=(WEB_SEARCH_TOOL, FETCH_TOOL)), stream=False
    )

    assert payload["tools"][0] == {
        "type": "web_search",
        "web_search": {"enable": True, "search_engine": "search-prime"},
    }
    assert payload["tools"][1]["function"]["name"] == "WebFetch"


def test_non_zai_backend_keeps_client_web_search():
    payload = LiteLLMProvider(settings()).build_request(
        request(backend="vllm"), stream=False
    )

    assert payload["tools"][0]["function"]["name"] == "WebSearch"


def test_zai_without_web_search_tool_adds_no_builtin():
    payload = LiteLLMProvider(settings()).build_request(
        request(tools=(FETCH_TOOL,)), stream=False
    )

    assert [tool["function"]["name"] for tool in payload["tools"]] == ["WebFetch"]


def test_zai_keeps_tool_choice_pointing_at_client_tool():
    from claude_code_proxy.domain.models import ToolChoice

    payload = LiteLLMProvider(settings()).build_request(
        request(
            tools=(WEB_SEARCH_TOOL, FETCH_TOOL),
            tool_choice=ToolChoice(type="tool", name="WebFetch"),
        ),
        stream=False,
    )

    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "WebFetch"},
    }
