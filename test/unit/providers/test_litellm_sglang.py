from dataclasses import replace
import json

import httpx
import pytest

from claude_code_proxy.domain.models import Message, TextBlock
from claude_code_proxy.providers import litellm as litellm_module
from claude_code_proxy.providers.litellm import LiteLLMProvider

from .litellm_test_support import FakeClient, request, settings  # noqa: F401


@pytest.fixture
def sglang_settings(settings):  # noqa: F811
    return replace(
        settings, openai_base_url="http://engine:30001/v1", openai_api_key="engine-key"
    )


@pytest.fixture
def tokenize_calls(monkeypatch):
    class Calls(list):
        answers: list

    calls = Calls()
    answers = []
    real_client = httpx.AsyncClient

    def handler(http_request):
        calls.append(http_request)
        return answers.pop(0) if answers else httpx.Response(200, json={"count": 362})

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(litellm_module.httpx, "AsyncClient", client_factory)
    calls.answers = answers
    return calls


def sglang_request(**changes):
    return request(model="openai/qwen3.8-27B", backend="sglang", **changes)


@pytest.mark.asyncio
async def test_sglang_count_uses_server_tokenizer(sglang_settings, tokenize_calls):
    client = FakeClient(token_count=9)

    count = await LiteLLMProvider(sglang_settings, client).count_tokens(sglang_request())

    assert count == 362
    assert client.counter_args is None
    [call] = tokenize_calls
    assert str(call.url) == "http://engine:30001/tokenize"
    assert call.headers["Authorization"] == "Bearer engine-key"
    body = json.loads(call.content)
    assert body["model"] == "qwen3.8-27B"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["chat_template_kwargs"] == {"reasoning_effort": "high"}


@pytest.mark.asyncio
async def test_sglang_count_forwards_tools(sglang_settings, tokenize_calls):
    from claude_code_proxy.domain.models import ToolDefinition

    tool = ToolDefinition(name="t", description="d", input_schema={"type": "object"})
    await LiteLLMProvider(sglang_settings, FakeClient()).count_tokens(
        sglang_request(tools=(tool,))
    )

    [call] = tokenize_calls
    assert json.loads(call.content)["tools"][0]["function"]["name"] == "t"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(500, json={"error": "down"}),
        httpx.Response(200, json={"tokens": [1, 2]}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"count": True}),
    ],
)
async def test_sglang_count_falls_back_to_local_counter(
    sglang_settings, tokenize_calls, answer
):
    tokenize_calls.answers.append(answer)
    client = FakeClient(token_count=9)

    count = await LiteLLMProvider(sglang_settings, client).count_tokens(sglang_request())

    assert count == 9
    assert len(tokenize_calls) == 1


@pytest.mark.asyncio
async def test_sglang_count_without_base_url_stays_local(settings, tokenize_calls):  # noqa: F811
    client = FakeClient(token_count=9)

    assert await LiteLLMProvider(settings, client).count_tokens(sglang_request()) == 9
    assert tokenize_calls == []


@pytest.mark.asyncio
async def test_vllm_count_never_calls_the_server(sglang_settings, tokenize_calls):
    client = FakeClient(token_count=9)
    vllm = request(model="openai/qwen3.8-27B", backend="vllm")

    assert await LiteLLMProvider(sglang_settings, client).count_tokens(vllm) == 9
    assert tokenize_calls == []


def test_sglang_demotes_mid_conversation_system_message(settings):  # noqa: F811
    payload = LiteLLMProvider(settings, object()).build_request(
        sglang_request(
            system=(TextBlock("be helpful"),),
            messages=(
                Message("user", (TextBlock("hello"),)),
                Message("system", (TextBlock("hook context"),)),
            ),
        ),
        stream=False,
    )

    assert [m["role"] for m in payload["messages"]] == ["system", "user", "user"]
