import asyncio
from dataclasses import replace

import httpx
import pytest

from claude_code_proxy.domain.models import (
    ClientIdentity,
    CompletionRequest,
    Message,
    TextBlock,
    ToolDefinition,
)
from claude_code_proxy.logging import RequestLogContext, SessionIdentity
from claude_code_proxy.providers.codex.provider import CODEX_RESPONSES_URL
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


class RecordingTelemetry:
    def __init__(self):
        self.calls = []

    def mark_retries_supported(self):
        self.calls.append(("mark_retries_supported",))

    def record_retry(self):
        self.calls.append(("record_retry",))

    def set_reasoning_continuation(self, value):
        self.calls.append(("set_reasoning_continuation", value))


class Response:
    def __init__(
        self,
        status=200,
        lines=(),
        text=(),
        headers=None,
        line_error=None,
        exit_error=None,
        block_lines=False,
    ):
        self.status_code = status
        self.headers = httpx.Headers(headers or {})
        self.lines = lines
        self.text = text
        self.line_error = line_error
        self.exit_error = exit_error
        self.block_lines = block_lines
        self.line_waiting = asyncio.Event()
        self.text_reads = 0
        self.text_chunk_size = None
        self.entered = False
        self.exited = False
        self.exit_count = 0

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.exited = True
        self.exit_count += 1
        if self.exit_error is not None:
            raise self.exit_error

    async def aiter_lines(self):
        for line in self.lines:
            yield line
        if self.block_lines:
            self.line_waiting.set()
            await asyncio.Event().wait()
        if self.line_error is not None:
            raise self.line_error

    async def aiter_text(self, chunk_size=None):
        self.text_chunk_size = chunk_size
        for part in self.text:
            self.text_reads += 1
            yield part

    async def aiter_raw(self):
        for part in self.text:
            self.text_reads += 1
            yield part.encode()


class RawBodyStream(httpx.AsyncByteStream):
    def __init__(self, chunks=(), error=None):
        self.chunks = chunks
        self.error = error
        self.attempted = False
        self.iterations = 0
        self.closed = False

    async def __aiter__(self):
        self.attempted = True
        for chunk in self.chunks:
            self.iterations += 1
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True


class EnterFailureContext:
    def __init__(self, error):
        self.error = error
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        raise self.error

    async def __aexit__(self, *args):
        pass


class BlockingEnterContext:
    def __init__(self):
        self.waiting = asyncio.Event()

    async def __aenter__(self):
        self.waiting.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *args):
        pass


class RealResponseContext:
    def __init__(self, response):
        self.response = response
        self.exited = False

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        self.exited = True
        await self.response.aclose()


def raw_response(status, stream, headers=None):
    return httpx.Response(
        status,
        headers=headers,
        stream=stream,
        request=httpx.Request("POST", CODEX_RESPONSES_URL),
    )


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
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response


class ExitClient(Client):
    exit_error = None
    exited = False
    exit_count = 0

    async def __aexit__(self, *args):
        type(self).exited = True
        type(self).exit_count += 1
        if self.exit_error is not None:
            raise self.exit_error


@pytest.fixture(autouse=True)
def reset_client():
    Client.responses = []
    Client.requests = []
    ExitClient.responses = []
    ExitClient.requests = []
    ExitClient.exit_error = None
    ExitClient.exited = False
    ExitClient.exit_count = 0


def log_context(provider="codex"):
    return RequestLogContext(
        session=SessionIdentity("session", "[session session]", False),
        method="POST",
        endpoint="/v1/messages",
        original_model="claude",
        upstream_model="openai/gpt-5",
        provider=provider,
        effort="default",
    )


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


def orchestration_request(*, agent_id=None):
    return request(
        system=(TextBlock("base system"),),
        client_identity=ClientIdentity(
            session_id="session-1",
            agent_id=agent_id,
            parent_agent_id="parent" if agent_id else None,
        ),
        tools=(
            ToolDefinition(
                "Agent",
                "Launch worker.",
                {"type": "object", "properties": {"prompt": {"type": "string"}}},
            ),
            ToolDefinition(
                "SendMessage",
                "Message worker.",
                {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
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


async def collect(provider, completion_request=None, telemetry=None):
    completion_request = completion_request or request()
    if telemetry is None:
        stream = provider.stream(completion_request)
    else:
        stream = provider.stream(completion_request, telemetry=telemetry)
    return [event async for event in stream]
