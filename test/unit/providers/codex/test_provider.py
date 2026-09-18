from dataclasses import replace
import json

import pytest

from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionRequest,
    Message,
    StreamComplete,
    StreamError,
    StreamStart,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolDefinition,
)
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.providers.codex.orchestration import (
    AGENT_COMPLETION_POLICY,
    AGENT_GUIDANCE,
    TASK_OUTPUT_GUIDANCE,
)
from claude_code_proxy.providers.codex.provider import (
    CODEX_RESPONSES_URL,
    CodexProvider,
)
from claude_code_proxy.reasoning import ReasoningPolicy


class Auth:
    def __init__(
        self,
        current=("secret-access", "account"),
        recovered=None,
        get_failure=None,
        recovery_failure=None,
        on_recover=None,
    ):
        self.current = current
        self.recovered = recovered or current
        self.get_failure = get_failure
        self.recovery_failure = recovery_failure
        self.on_recover = on_recover
        self.rejected = []

    async def get_auth(self):
        if self.get_failure is not None:
            raise self.get_failure
        return self.current

    async def recover_rejected(self, access_token):
        self.rejected.append(access_token)
        if self.on_recover is not None:
            self.on_recover()
        if self.recovery_failure is not None:
            raise self.recovery_failure
        return self.recovered


class Response:
    def __init__(self, status=200, lines=(), text=()):
        self.status_code = status
        self.lines = lines
        self.text = text
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.exited = True

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aiter_text(self):
        for part in self.text:
            yield part


class Client:
    responses = []
    requests = []

    def __init__(self, **kwargs):
        self._responses = iter(self.responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return next(self._responses)


@pytest.fixture(autouse=True)
def reset_client():
    Client.responses = []
    Client.requests = []


def request(session_id=None, **changes):
    base = CompletionRequest(
        "claude",
        "openai/gpt-5",
        "claude",
        100,
        (Message("user", (TextBlock("hi"),)),),
        ReasoningPolicy(None, None),
        client_identity=ClientIdentity(session_id=session_id),
    )
    return replace(base, **changes)


def orchestration_request():
    return request(
        system=(TextBlock("base system"),),
        tools=(
            ToolDefinition(
                "Agent",
                "Launch worker.",
                {"type": "object", "properties": {"prompt": {"type": "string"}}},
            ),
            ToolDefinition(
                "TaskOutput",
                "Retrieve output.",
                {"type": "object", "properties": {"task_id": {"type": "string"}}},
            ),
        ),
    )


def completed_response(input_tokens=0, output_tokens=0):
    return Response(
        lines=[
            "event: response.completed",
            (
                'data: {"usage":{"input_tokens":'
                f"{input_tokens},\"output_tokens\":{output_tokens}"
                '},"status":"completed"}'
            ),
            "data: [DONE]",
        ]
    )


async def collect(provider, completion_request=None):
    completion_request = completion_request or request()
    return [event async for event in provider.stream(completion_request)]


async def test_stream_posts_headers_and_returns_semantic_events():
    Client.responses = [
        Response(
            lines=[
                "event: response.output_text.delta",
                'data: {"delta":"hello"}',
                "event: response.completed",
                (
                    'data: {"usage":{"input_tokens":2,'
                    '"output_tokens":1},"status":"completed"}'
                ),
                "data: [DONE]",
            ]
        )
    ]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamStart(),
        TextDelta("hello"),
        StreamComplete("end_turn", TokenUsage(2, 1)),
    ]
    assert Client.requests[0][1] == CODEX_RESPONSES_URL
    assert (
        Client.requests[0][2]["headers"]["Authorization"]
        == "Bearer secret-access"
    )


async def test_stream_forwards_client_session_id_unchanged():
    Client.responses = [completed_response()]

    await collect(CodexProvider(Auth(), Client), request("session-1"))

    assert Client.requests[0][2]["headers"]["session-id"] == "session-1"


async def test_root_and_agents_share_session_but_use_distinct_threads():
    Client.responses = [
        completed_response(),
        completed_response(),
        completed_response(),
    ]
    provider = CodexProvider(Auth(), Client)

    await collect(
        provider,
        request(client_identity=ClientIdentity("session")),
    )
    await collect(
        provider,
        request(client_identity=ClientIdentity("session", "first")),
    )
    await collect(
        provider,
        request(client_identity=ClientIdentity("session", "second")),
    )

    headers = [call[2]["headers"] for call in Client.requests]
    payloads = [call[2]["json"] for call in Client.requests]
    assert {item["session-id"] for item in headers} == {"session"}
    assert {body["prompt_cache_key"] for body in payloads} == {"session"}
    assert len({item["thread-id"] for item in headers}) == 3
    assert all(
        item["x-client-request-id"] == item["thread-id"]
        for item in headers
    )


async def test_nested_agent_forwards_matching_parent_thread_metadata():
    Client.responses = [completed_response()]
    client_identity = ClientIdentity("session", "child", "parent")

    await collect(
        CodexProvider(Auth(), Client),
        request(client_identity=client_identity),
    )

    outbound = Client.requests[0][2]
    headers = outbound["headers"]
    metadata = json.loads(
        outbound["json"]["client_metadata"]["x-codex-turn-metadata"]
    )
    assert headers["x-codex-parent-thread-id"] == metadata["parent_thread_id"]


async def test_headerless_requests_receive_distinct_fallback_sessions():
    Client.responses = [completed_response(), completed_response()]
    provider = CodexProvider(Auth(), Client)

    await collect(provider)
    await collect(provider)

    first, second = [call[2] for call in Client.requests]
    assert first["headers"]["session-id"] != second["headers"]["session-id"]
    assert first["headers"]["thread-id"] != second["headers"]["thread-id"]
    assert first["json"]["prompt_cache_key"] != second["json"]["prompt_cache_key"]


async def test_401_retry_reuses_client_session_id():
    Client.responses = [Response(status=401), completed_response()]

    await collect(CodexProvider(Auth(), Client), request("session-1"))

    first, second = [call[2] for call in Client.requests]
    assert first["headers"]["session-id"] == second["headers"]["session-id"] == "session-1"
    assert first["headers"]["thread-id"] == second["headers"]["thread-id"]
    assert first["headers"]["x-client-request-id"] == second["headers"]["x-client-request-id"]
    assert first["json"]["prompt_cache_key"] == second["json"]["prompt_cache_key"]
    assert first["json"]["client_metadata"] == second["json"]["client_metadata"]


async def test_401_retry_reuses_generated_fallback_session_id():
    Client.responses = [Response(status=401), completed_response()]

    await collect(CodexProvider(Auth(), Client))

    first, second = [call[2] for call in Client.requests]
    assert first["headers"]["session-id"] == second["headers"]["session-id"]
    assert first["headers"]["thread-id"] == second["headers"]["thread-id"]
    assert first["headers"]["x-client-request-id"] == second["headers"]["x-client-request-id"]
    assert first["json"]["prompt_cache_key"] == second["json"]["prompt_cache_key"]
    assert first["json"]["client_metadata"] == second["json"]["client_metadata"]


async def test_401_recovers_credentials_after_closing_response_and_retries_once():
    rejected = Response(status=401)
    auth = Auth(
        recovered=("new-access", "account"),
        on_recover=lambda: rejected.exited
        or pytest.fail("response must close before credential recovery"),
    )
    Client.responses = [rejected, completed_response()]

    events = await collect(CodexProvider(auth, Client))

    assert events == [
        StreamStart(),
        StreamComplete("end_turn", TokenUsage(0, 0)),
    ]
    assert auth.rejected == ["secret-access"]
    assert len(Client.requests) == 2
    assert (
        Client.requests[0][2]["headers"]["Authorization"]
        == "Bearer secret-access"
    )
    assert (
        Client.requests[1][2]["headers"]["Authorization"]
        == "Bearer new-access"
    )


async def test_second_401_returns_authentication_error_without_stream_start():
    auth = Auth(recovered=("new-access", "account"))
    Client.responses = [Response(status=401), Response(status=401)]

    events = await collect(CodexProvider(auth, Client))

    assert len(Client.requests) == 2
    assert auth.rejected == ["secret-access"]
    assert events == [
        StreamError(
            error_type="authentication_error",
            message="Codex authentication failed",
            status_code=401,
            retryable=False,
            provider="codex",
            diagnostic="Codex authentication failed",
        )
    ]


async def test_403_does_not_reload_or_retry_credentials():
    auth = Auth(recovered=("new-access", "account"))
    Client.responses = [Response(status=403, text=["forbidden"])]

    events = await collect(CodexProvider(auth, Client))

    assert len(Client.requests) == 1
    assert auth.rejected == []
    assert events == [
        StreamError(
            error_type="permission_error",
            message="Codex API error 403: forbidden",
            status_code=403,
            retryable=False,
            provider="codex",
            diagnostic="Codex API error 403: forbidden",
        )
    ]


async def test_non_200_stream_returns_error_without_stream_start():
    Client.responses = [Response(status=429, text=["quota resets in 30 seconds"])]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamError(
            error_type="rate_limit_error",
            message="Codex API error 429: quota resets in 30 seconds",
            status_code=429,
            retryable=True,
            provider="codex",
            diagnostic="Codex API error 429: quota resets in 30 seconds",
        )
    ]


async def test_initial_credential_failure_returns_safe_authentication_error():
    auth = Auth(
        get_failure=RuntimeError(
            "access sample-access-token refresh sample-refresh-token"
        )
    )

    events = await collect(CodexProvider(auth, Client))

    assert Client.requests == []
    assert events == [
        StreamError(
            error_type="authentication_error",
            message="Codex authentication failed",
            status_code=401,
            retryable=False,
            provider="codex",
            diagnostic="Codex authentication failed",
        )
    ]
    assert "sample-access-token" not in events[0].diagnostic
    assert "sample-refresh-token" not in events[0].diagnostic


async def test_recovery_failure_returns_safe_error_without_second_request():
    rejected = Response(status=401)
    auth = Auth(
        recovery_failure=RuntimeError("sample-refresh-token was rejected"),
        on_recover=lambda: rejected.exited
        or pytest.fail("response must close before credential recovery"),
    )
    Client.responses = [rejected]

    events = await collect(CodexProvider(auth, Client))

    assert len(Client.requests) == 1
    assert events == [
        StreamError(
            error_type="authentication_error",
            message="Codex authentication failed",
            status_code=401,
            retryable=False,
            provider="codex",
            diagnostic="Codex authentication failed",
        )
    ]
    assert "sample-refresh-token" not in events[0].diagnostic


async def test_stream_forwards_explicit_provider_failure_message():
    Client.responses = [
        Response(
            lines=[
                "event: response.failed",
                (
                    'data: {"response":{"error":{"code":"server_error",'
                    '"message":"Model backend unavailable"}}}'
                ),
                "data: [DONE]",
            ]
        )
    ]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamStart(),
        StreamError(
            error_type="api_error",
            message="Model backend unavailable",
            retryable=True,
            provider="codex",
            diagnostic="server_error: Model backend unavailable",
        ),
    ]


async def test_stream_requires_explicit_completed_event():
    Client.responses = [
        Response(
            lines=[
                "event: response.output_text.delta",
                'data: {"delta":"partial"}',
                "data: [DONE]",
            ]
        )
    ]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[:2] == [StreamStart(), TextDelta("partial")]
    assert isinstance(events[-1], StreamError)
    assert events[-1].message == "Internal server error"


async def test_complete_preserves_stream_error_status():
    Client.responses = [Response(status=429, text=["quota exceeded"])]

    with pytest.raises(ProviderError) as caught:
        await CodexProvider(Auth(), Client).complete(request())

    assert caught.value.status_code == 429
    assert str(caught.value) == "Codex API error 429: quota exceeded"


async def test_complete_buffers_stream():
    Client.responses = [
        Response(
            lines=[
                "event: response.output_text.delta",
                'data: {"delta":"hello"}',
                "event: response.completed",
                'data: {"usage":{},"status":"completed"}',
            ]
        )
    ]

    response = await CodexProvider(Auth(), Client).complete(request())

    assert response.content == (TextBlock("hello"),)


async def test_count_tokens_uses_local_counter_only():
    calls = []

    async def local_counter(completion_request):
        calls.append(completion_request)
        return 8

    provider = CodexProvider(Auth(), Client, local_counter)

    assert await provider.count_tokens(request()) == 8
    assert calls[0].model == "openai/gpt-5"


async def test_stream_applies_codex_agent_completion_guidance():
    Client.responses = [completed_response()]

    await collect(CodexProvider(Auth(), Client), orchestration_request())

    payload = Client.requests[0][2]["json"]
    assert payload["instructions"] == f"base system\n\n{AGENT_COMPLETION_POLICY}"
    assert payload["tools"][0]["description"].endswith(AGENT_GUIDANCE)
    assert payload["tools"][1]["description"].endswith(TASK_OUTPUT_GUIDANCE)
    source_tools = orchestration_request().tools
    assert payload["tools"][0]["parameters"] == source_tools[0].input_schema
    assert payload["tools"][1]["parameters"] == source_tools[1].input_schema


async def test_count_tokens_applies_codex_agent_completion_guidance():
    captured = []

    async def count_tokens(completion_request):
        captured.append(completion_request)
        return 17

    provider = CodexProvider(Auth(), Client, token_counter=count_tokens)

    assert await provider.count_tokens(orchestration_request()) == 17
    assert captured[0].system[-1] == TextBlock(AGENT_COMPLETION_POLICY)
    assert captured[0].tools[0].description.endswith(AGENT_GUIDANCE)
    assert captured[0].tools[1].description.endswith(TASK_OUTPUT_GUIDANCE)
