import asyncio
import json
import logging
import re

import httpx
import pytest

from claude_code_proxy.domain.models import (
    ClientIdentity,
    StreamComplete,
    StreamError,
    StreamStart,
    TextBlock,
    TextDelta,
    TokenUsage,
)
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.logging import log_stream_failure
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
from claude_code_proxy.providers.codex.translation import CodexEventTranslator
from claude_code_proxy.service import _validated_stream

from test.unit.providers.codex.provider_test_support import (
    Auth,
    Client,
    EnterFailureContext,
    ExitClient,
    RawBodyStream,
    RealResponseContext,
    Response,
    collect,
    completed_response,
    log_context,
    orchestration_request,
    raw_response,
    request,
    reset_client as reset_client,
)


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


async def test_early_close_immediately_closes_response_and_client_once():
    response = Response()
    ExitClient.responses = [response]
    stream = CodexProvider(Auth(), ExitClient).stream(request())

    assert await anext(stream) == StreamStart()
    await stream.aclose()

    assert response.exited is True
    assert response.exit_count == 1
    assert ExitClient.exited is True
    assert ExitClient.exit_count == 1


async def test_validated_stream_early_close_closes_codex_contexts_once():
    response = Response()
    ExitClient.responses = [response]
    provider = CodexProvider(Auth(), ExitClient)
    stream = _validated_stream(provider.stream(request()), provider.name)

    assert await anext(stream) == StreamStart()
    await stream.aclose()

    assert response.exited is True
    assert response.exit_count == 1
    assert ExitClient.exited is True
    assert ExitClient.exit_count == 1


@pytest.mark.parametrize("validated", [False, True])
@pytest.mark.parametrize("exit_owner", ["response", "client"])
@pytest.mark.parametrize("exit_error_type", ["runtime", "cancelled"])
async def test_external_close_ignores_context_cleanup_failure(
    validated, exit_owner, exit_error_type
):
    exit_error = (
        RuntimeError("secret cleanup failure")
        if exit_error_type == "runtime"
        else asyncio.CancelledError()
    )
    response = Response()
    client_factory = ExitClient
    ExitClient.responses = [response]
    if exit_owner == "response":
        response.exit_error = exit_error
    else:
        ExitClient.exit_error = exit_error
    provider = CodexProvider(Auth(), client_factory)
    provider_stream = provider.stream(request())
    stream = (
        _validated_stream(provider_stream, provider.name)
        if validated
        else provider_stream
    )

    assert await anext(stream) == StreamStart()
    await stream.aclose()

    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert response.exit_count == 1
    assert ExitClient.exit_count == 1


@pytest.mark.parametrize("validated", [False, True])
@pytest.mark.parametrize("exit_owner", ["response", "client"])
@pytest.mark.parametrize("exit_error_type", ["runtime", "cancelled"])
async def test_external_cancellation_survives_context_cleanup_failure(
    validated, exit_owner, exit_error_type
):
    exit_error = (
        RuntimeError("secret cleanup failure")
        if exit_error_type == "runtime"
        else asyncio.CancelledError()
    )
    response = Response(block_lines=True)
    ExitClient.responses = [response]
    if exit_owner == "response":
        response.exit_error = exit_error
    else:
        ExitClient.exit_error = exit_error
    provider = CodexProvider(Auth(), ExitClient)
    provider_stream = provider.stream(request())
    stream = (
        _validated_stream(provider_stream, provider.name)
        if validated
        else provider_stream
    )

    assert await anext(stream) == StreamStart()
    pending = asyncio.create_task(anext(stream))
    await response.line_waiting.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert response.exit_count == 1
    assert ExitClient.exit_count == 1


async def test_terminal_event_is_emitted_after_response_and_client_cleanup():
    response = completed_response()
    ExitClient.responses = [response]
    stream = CodexProvider(Auth(), ExitClient).stream(request())

    assert await anext(stream) == StreamStart()
    terminal = await anext(stream)

    assert terminal == StreamComplete("end_turn", TokenUsage(0, 0))
    assert response.exited is True
    assert ExitClient.exited is True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            httpx.ReadTimeout(
                "secret exit timeout",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            StreamError(
                error_type="timeout_error",
                message="Request timed out",
                status_code=504,
                provider="codex",
                diagnostic=FailureDiagnostic(
                    FailureCategory.TRANSPORT,
                    FailureStage.STREAM,
                    "timeout",
                ),
            ),
        ),
        (
            httpx.ConnectError(
                "secret exit connection",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            StreamError(
                status_code=503,
                provider="codex",
                diagnostic=FailureDiagnostic(
                    FailureCategory.TRANSPORT,
                    FailureStage.STREAM,
                    "transport_error",
                ),
            ),
        ),
        (
            RuntimeError("secret exit failure"),
            StreamError(
                status_code=500,
                provider="codex",
                diagnostic=FailureDiagnostic(
                    FailureCategory.INTERNAL,
                    FailureStage.STREAM,
                    "stream_cleanup_failed",
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize("exit_owner", ["response", "client"])
async def test_context_exit_failure_replaces_success_with_one_stream_error(
    error, expected, exit_owner
):
    response = completed_response()
    client_factory = Client
    if exit_owner == "response":
        response.exit_error = error
        Client.responses = [response]
    else:
        ExitClient.responses = [response]
        ExitClient.exit_error = error
        client_factory = ExitClient

    events = await collect(CodexProvider(Auth(), client_factory))

    assert events[0] == StreamStart()
    assert len(events) == 2
    actual = events[1]
    assert actual.error_type == expected.error_type
    assert actual.message == expected.message
    assert actual.status_code == expected.status_code
    assert actual.provider == expected.provider
    assert actual.diagnostic.category == expected.diagnostic.category
    assert actual.diagnostic.stage == expected.diagnostic.stage
    assert actual.diagnostic.code == expected.diagnostic.code
    if isinstance(error, RuntimeError):
        assert actual.diagnostic.exception_type == "RuntimeError"
        assert re.fullmatch(
            r"claude_code_proxy\.providers\.codex\.provider:"
            r"(?:_stream_with_client|stream):\d+",
            actual.diagnostic.location,
        )
    else:
        assert actual.diagnostic.exception_type is None
        assert actual.diagnostic.location is None
    assert response.exited is True
    if exit_owner == "client":
        assert ExitClient.exited is True
    assert "secret" not in repr(events)


async def test_response_exit_failure_does_not_replace_http_status_failure():
    body = '{"error":{"code":"quota"}}'
    response = Response(
        status=429,
        text=[body],
        headers={"Content-Length": str(len(body))},
        exit_error=RuntimeError("secret exit failure"),
    )
    Client.responses = [response]

    events = await collect(CodexProvider(Auth(), Client))

    assert len(events) == 1
    assert events[0].status_code == 429
    assert events[0].diagnostic == FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.RESPONSE,
        "http_error",
        "quota",
    )
    assert response.exited is True


async def test_response_exit_failure_does_not_replace_stream_failure():
    response = Response(
        lines=[
            "event: response.failed",
            'data: {"error":{"code":"server_error"}}',
        ],
        exit_error=RuntimeError("secret exit failure"),
    )
    Client.responses = [response]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamStart(),
        StreamError(
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.PROVIDER_PROTOCOL,
                FailureStage.STREAM,
                "response_failed",
                "server_error",
            ),
        ),
    ]
    assert response.exited is True


@pytest.mark.parametrize("exit_owner", ["response", "client"])
async def test_context_exit_cancellation_propagates_without_terminal(exit_owner):
    response = completed_response()
    client_factory = Client
    if exit_owner == "response":
        response.exit_error = asyncio.CancelledError()
        Client.responses = [response]
    else:
        ExitClient.responses = [response]
        ExitClient.exit_error = asyncio.CancelledError()
        client_factory = ExitClient
    stream = CodexProvider(Auth(), client_factory).stream(request())

    assert await anext(stream) == StreamStart()
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert response.exited is True


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


async def test_403_does_not_reload_or_expose_response_body():
    auth = Auth(recovered=("new-access", "account"))
    Client.responses = [Response(status=403, text=["forbidden secret"])]

    events = await collect(CodexProvider(auth, Client))

    assert len(Client.requests) == 1
    assert auth.rejected == []
    assert events == [
        StreamError(
            error_type="permission_error",
            message="Permission denied",
            status_code=403,
            retryable=False,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.AUTHENTICATION,
                FailureStage.RESPONSE,
                "http_error",
            ),
        )
    ]
    assert "forbidden secret" not in repr(events)


@pytest.mark.parametrize(
    ("body", "provider_code"),
    [
        ('{"error":{"code":"rate_limit_exceeded","message":"secret"}}', "rate_limit_exceeded"),
        ('{"error":{"type":"quota_error","message":"secret"}}', None),
        ('{"error":{"code":"","type":"must_not_replace"}}', None),
        ('{"error":{"code":{"nested":"not allowed"}}}', None),
        ('{"message":"not a recognized envelope"}', None),
        ("plain secret body", None),
    ],
)
async def test_non_200_extracts_only_allowlisted_json_error_code(
    body, provider_code
):
    Client.responses = [
        Response(
            status=429,
            text=[body],
            headers={"Content-Length": str(len(body.encode()))},
        )
    ]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamError(
            error_type="rate_limit_error",
            message="Rate limit exceeded",
            status_code=429,
            retryable=True,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.UPSTREAM_HTTP,
                FailureStage.RESPONSE,
                "http_error",
                provider_code,
            ),
        )
    ]
    assert "secret" not in repr(events)


async def test_non_200_stops_reading_large_body_and_does_not_parse_truncated_json():
    response = Response(
        status=500,
        text=['{"error":{"code":"must_not_survive"},"padding":"']
        + ["x" * 1024] * 100
        + ['"}'],
    )
    Client.responses = [response]

    events = await collect(CodexProvider(Auth(), Client))

    assert response.text_reads < len(response.text)
    assert events[-1].diagnostic == FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.RESPONSE,
        "http_error",
    )
    assert "must_not_survive" not in repr(events)


async def test_non_200_without_content_length_does_not_read_body():
    stream = RawBodyStream(
        [b'{"error":{"code":"must_not_survive"}}' + b"x" * 100_000]
    )
    response = raw_response(429, stream)

    async def reject_text_decoding(*args, **kwargs):
        raise AssertionError("optional parsing must not decode text")

    response.aiter_text = reject_text_decoding
    context = RealResponseContext(response)
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[-1].status_code == 429
    assert events[-1].message == "Rate limit exceeded"
    assert events[-1].diagnostic == FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.RESPONSE,
        "http_error",
    )
    assert stream.iterations == 0
    assert context.exited is True
    assert stream.closed is True


async def test_non_200_valid_bounded_content_length_omits_unknown_code():
    body = b'{"error":{"code":"bounded_code"}}'
    stream = RawBodyStream([body])
    context = RealResponseContext(
        raw_response(
            429,
            stream,
            {"Content-Length": str(len(body))},
        )
    )
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[-1].diagnostic.provider_code is None
    assert stream.iterations == 1
    assert context.exited is True
    assert stream.closed is True


@pytest.mark.parametrize(
    "content_length",
    ["+2", " 2", "2 ", "-1", "2,2"],
)
async def test_non_200_rejects_invalid_content_length_without_reading(
    content_length,
):
    stream = RawBodyStream([b"{}"])
    context = RealResponseContext(
        raw_response(429, stream, {"Content-Length": content_length})
    )
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[-1].diagnostic.provider_code is None
    assert stream.iterations == 0
    assert context.exited is True
    assert stream.closed is True


@pytest.mark.parametrize("declared_delta", [-1, 1])
async def test_non_200_rejects_content_length_mismatch(declared_delta):
    body = b'{"error":{"code":"must_not_survive"}}'
    stream = RawBodyStream([body])
    context = RealResponseContext(
        raw_response(
            429,
            stream,
            {"Content-Length": str(len(body) + declared_delta)},
        )
    )
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[-1].diagnostic.provider_code is None
    assert context.exited is True
    assert stream.closed is True


async def test_non_200_lying_small_content_length_rejects_oversized_chunk():
    stream = RawBodyStream(
        [b'{"error":{"code":"must_not_survive"}}' + b"x" * 100_000]
    )
    context = RealResponseContext(
        raw_response(429, stream, {"Content-Length": "8"})
    )
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[-1].diagnostic.provider_code is None
    assert stream.iterations == 1
    assert context.exited is True
    assert stream.closed is True


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Length": "100000"},
        {"Content-Encoding": "gzip"},
    ],
)
async def test_non_200_skips_optional_parsing_for_large_or_encoded_body(headers):
    stream = RawBodyStream([b'{"error":{"code":"must_not_survive"}}'])
    context = RealResponseContext(raw_response(429, stream, headers))
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[-1].status_code == 429
    assert events[-1].diagnostic.provider_code is None
    assert stream.iterations == 0
    assert context.exited is True
    assert stream.closed is True


@pytest.mark.parametrize(
    ("status_code", "error", "message", "category", "retryable"),
    [
        (
            403,
            httpx.ReadTimeout(
                "secret timeout",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            "Permission denied",
            FailureCategory.AUTHENTICATION,
            False,
        ),
        (
            429,
            httpx.ConnectError(
                "secret connection",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            "Rate limit exceeded",
            FailureCategory.UPSTREAM_HTTP,
            True,
        ),
        (
            429,
            RuntimeError("secret parser failure"),
            "Rate limit exceeded",
            FailureCategory.UPSTREAM_HTTP,
            True,
        ),
    ],
)
async def test_optional_body_extraction_failure_preserves_http_status_error(
    status_code, error, message, category, retryable
):
    body = b"{}"
    stream = RawBodyStream(chunks=[body], error=error)
    context = RealResponseContext(
        raw_response(
            status_code,
            stream,
            {"Content-Length": str(len(body))},
        )
    )
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamError(
            error_type=(
                "permission_error"
                if status_code == 403
                else "rate_limit_error"
            ),
            message=message,
            status_code=status_code,
            retryable=retryable,
            provider="codex",
            diagnostic=FailureDiagnostic(
                category,
                FailureStage.RESPONSE,
                "http_error",
            ),
        )
    ]
    assert "secret" not in repr(events)
    assert stream.attempted is True
    assert stream.iterations == 1
    assert context.exited is True
    assert stream.closed is True


async def test_optional_body_extraction_does_not_swallow_cancellation():
    stream = RawBodyStream(error=asyncio.CancelledError())
    context = RealResponseContext(
        raw_response(429, stream, {"Content-Length": "1"})
    )
    Client.responses = [context]

    with pytest.raises(asyncio.CancelledError):
        await collect(CodexProvider(Auth(), Client))

    assert context.exited is True
    assert stream.closed is True


@pytest.mark.parametrize(
    ("error", "status_code", "message", "code"),
    [
        (
            httpx.ReadTimeout(
                "secret timeout detail",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            504,
            "Request timed out",
            "timeout",
        ),
        (
            httpx.ConnectError(
                "secret connection detail",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            503,
            "Internal server error",
            "transport_error",
        ),
    ],
)
async def test_transport_failures_are_safe_and_structured(
    error, status_code, message, code
):
    Client.responses = [error]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamError(
            error_type="timeout_error" if status_code == 504 else "api_error",
            message=message,
            status_code=status_code,
            retryable=True,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.TRANSPORT,
                FailureStage.REQUEST,
                code,
            ),
        )
    ]
    assert "secret" not in repr(events)


@pytest.mark.parametrize(
    "target",
    ["reconcile_codex_request", "build_request"],
)
async def test_request_preparation_failure_is_translation_error(
    monkeypatch, target
):
    def fail_preparation(*args, **kwargs):
        raise RuntimeError("secret request detail")

    monkeypatch.setattr(
        f"claude_code_proxy.providers.codex.provider.{target}",
        fail_preparation,
    )

    events = await collect(CodexProvider(Auth(), Client))

    assert len(events) == 1
    event = events[0]
    assert event.status_code == 500
    assert event.provider == "codex"
    assert event.diagnostic.category == FailureCategory.TRANSLATION
    assert event.diagnostic.stage == FailureStage.PROVIDER_TRANSLATION
    assert event.diagnostic.code == "request_translation_failed"
    assert event.diagnostic.exception_type == "RuntimeError"
    assert re.fullmatch(
        r"claude_code_proxy\.providers\.codex\.provider:_prepare_request:\d+",
        event.diagnostic.location,
    )
    assert "secret request detail" not in repr(events)
    assert Client.requests == []


@pytest.mark.parametrize(
    ("error", "status_code", "message", "code"),
    [
        (
            httpx.ReadTimeout(
                "secret timeout",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            504,
            "Request timed out",
            "timeout",
        ),
        (
            httpx.ConnectError(
                "secret connection",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            503,
            "Internal server error",
            "transport_error",
        ),
    ],
)
async def test_response_context_entry_transport_failure_is_request_stage(
    error, status_code, message, code
):
    context = EnterFailureContext(error)
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert context.entered is True
    assert events == [
        StreamError(
            error_type=("timeout_error" if status_code == 504 else "api_error"),
            message=message,
            status_code=status_code,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.TRANSPORT,
                FailureStage.REQUEST,
                code,
            ),
        )
    ]


async def test_response_context_entry_unknown_failure_is_request_stage(caplog):
    context = EnterFailureContext(RuntimeError("secret context failure"))
    Client.responses = [context]

    events = await collect(CodexProvider(Auth(), Client))

    assert len(events) == 1
    event = events[0]
    assert event.status_code == 500
    assert event.provider == "codex"
    assert event.diagnostic.category == FailureCategory.INTERNAL
    assert event.diagnostic.stage == FailureStage.REQUEST
    assert event.diagnostic.code == "request_failed"
    assert event.diagnostic.exception_type == "RuntimeError"
    assert re.fullmatch(
        r"claude_code_proxy\.providers\.codex\.provider:_stream_with_client:\d+",
        event.diagnostic.location,
    )
    assert context.entered is True
    assert "secret context failure" not in repr(events)

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_stream_failure(log_context(), event)

    rendered = caplog.records[-1].getMessage()
    assert "exception=RuntimeError" in rendered
    assert re.search(
        r"location=claude_code_proxy\.providers\.codex\.provider:"
        r"_stream_with_client:\d+",
        rendered,
    )
    assert "secret context failure" not in rendered


@pytest.mark.parametrize(
    ("error", "status_code", "message", "code"),
    [
        (
            httpx.ReadTimeout(
                "secret timeout",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            504,
            "Request timed out",
            "timeout",
        ),
        (
            httpx.ConnectError(
                "secret connection",
                request=httpx.Request("POST", CODEX_RESPONSES_URL),
            ),
            503,
            "Internal server error",
            "transport_error",
        ),
    ],
)
async def test_sse_iteration_transport_failure_is_stream_stage(
    error, status_code, message, code
):
    response = Response(
        lines=[
            "event: response.output_text.delta",
            'data: {"delta":"partial"}',
        ],
        line_error=error,
    )
    Client.responses = [response]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamStart(),
        TextDelta("partial"),
        StreamError(
            error_type=("timeout_error" if status_code == 504 else "api_error"),
            message=message,
            status_code=status_code,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.TRANSPORT,
                FailureStage.STREAM,
                code,
            ),
        ),
    ]
    assert response.exited is True
    assert "secret" not in repr(events)


async def test_sse_iteration_unknown_failure_is_internal_stream_error():
    response = Response(
        line_error=RuntimeError("secret stream failure")
    )
    Client.responses = [response]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[0] == StreamStart()
    assert len(events) == 2
    event = events[1]
    assert event.status_code == 500
    assert event.provider == "codex"
    assert event.diagnostic.category == FailureCategory.INTERNAL
    assert event.diagnostic.stage == FailureStage.STREAM
    assert event.diagnostic.code == "stream_failed"
    assert event.diagnostic.exception_type == "RuntimeError"
    assert re.fullmatch(
        r"claude_code_proxy\.providers\.codex\.provider:_response_events:\d+",
        event.diagnostic.location,
    )
    assert response.exited is True
    assert "secret stream failure" not in repr(events)


@pytest.mark.parametrize(
    "lines",
    [
        ["event: response.output_text.delta", "data: []"],
        [
            "event: response.output_text.delta",
            'data: {"delta":7}',
        ],
        [
            "event: response.function_call_arguments.delta",
            'data: {"output_index":0,"delta":7}',
        ],
        [
            "event: response.output_item.added",
            'data: {"output_index":0,"item":"not-an-object"}',
        ],
    ],
)
async def test_valid_json_with_invalid_event_shape_is_protocol_error(lines):
    response = Response(lines=lines)
    Client.responses = [response]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamStart(),
        StreamError(
            status_code=500,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.PROVIDER_PROTOCOL,
                FailureStage.STREAM,
                "invalid_event_payload",
            ),
        ),
    ]
    assert response.exited is True


async def test_translator_feed_failure_is_translation_error(monkeypatch):
    def fail_translation(self, event_type, data):
        raise RuntimeError("secret translator detail")

    monkeypatch.setattr(CodexEventTranslator, "feed", fail_translation)
    response = Response(
        lines=[
            "event: response.output_text.delta",
            'data: {"delta":"hello"}',
        ]
    )
    Client.responses = [response]

    events = await collect(CodexProvider(Auth(), Client))

    assert events[0] == StreamStart()
    assert len(events) == 2
    event = events[1]
    assert event.status_code == 500
    assert event.provider == "codex"
    assert event.diagnostic.category == FailureCategory.TRANSLATION
    assert event.diagnostic.stage == FailureStage.PROVIDER_TRANSLATION
    assert event.diagnostic.code == "stream_chunk_translation_failed"
    assert event.diagnostic.exception_type == "RuntimeError"
    assert re.fullmatch(
        r"claude_code_proxy\.providers\.codex\.provider:_consume_response:\d+",
        event.diagnostic.location,
    )
    assert response.exited is True
    assert "secret translator detail" not in repr(events)


async def test_malformed_sse_data_is_controlled_protocol_error(caplog):
    Client.responses = [
        Response(lines=["event: response.output_text.delta", "data: {secret"])
    ]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamStart(),
        StreamError(
            status_code=500,
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.PROVIDER_PROTOCOL,
                FailureStage.STREAM,
                "malformed_sse_data",
            ),
        ),
    ]
    assert events[-1].diagnostic.exception_type is None
    assert events[-1].diagnostic.location is None
    assert "secret" not in repr(events)

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.logging"):
        log_stream_failure(log_context(), events[-1])

    rendered = caplog.records[-1].getMessage()
    assert "exception=" not in rendered
    assert "location=" not in rendered
    assert "secret" not in rendered


async def test_stream_response_failed_uses_generic_message_and_code_only():
    Client.responses = [
        Response(
            lines=[
                "event: response.failed",
                (
                    'data: {"response":{"error":{"code":"server_error",'
                    '"message":"Model backend secret"}}}'
                ),
                "data: [DONE]",
            ]
        )
    ]

    events = await collect(CodexProvider(Auth(), Client))

    assert events == [
        StreamStart(),
        StreamError(
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.PROVIDER_PROTOCOL,
                FailureStage.STREAM,
                "response_failed",
                "server_error",
            ),
        ),
    ]
    assert "Model backend secret" not in repr(events)


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

    assert events == [
        StreamStart(),
        TextDelta("partial"),
        StreamError(
            provider="codex",
            diagnostic=FailureDiagnostic(
                FailureCategory.PROVIDER_PROTOCOL,
                FailureStage.STREAM,
                "missing_response_completed",
            ),
        ),
    ]


async def test_complete_preserves_stream_error_status_and_diagnostic():
    body = '{"error":{"code":"quota_exhausted","message":"secret"}}'
    Client.responses = [
        Response(
            status=429,
            text=[body],
            headers={"Content-Length": str(len(body))},
        )
    ]

    with pytest.raises(ProviderError) as caught:
        await CodexProvider(Auth(), Client).complete(request())

    assert caught.value.status_code == 429
    assert str(caught.value) == "Rate limit exceeded"
    assert caught.value.diagnostic == FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.RESPONSE,
        "http_error",
    )
    assert caught.value.diagnostic.provider_code is None
    assert "secret" not in repr(caught.value)


async def test_complete_normalization_failure_is_safe_translation_error(
    monkeypatch,
):
    Client.responses = [completed_response()]

    def fail_normalization(request, events):
        raise RuntimeError("secret response detail")

    monkeypatch.setattr(
        "claude_code_proxy.providers.codex.provider.response_from_events",
        fail_normalization,
    )

    with pytest.raises(ProviderError) as caught:
        await CodexProvider(Auth(), Client).complete(request())

    assert str(caught.value) == "Internal server error"
    assert caught.value.status_code == 500
    diagnostic = caught.value.diagnostic
    assert diagnostic.category == FailureCategory.TRANSLATION
    assert diagnostic.stage == FailureStage.PROVIDER_TRANSLATION
    assert diagnostic.code == "response_translation_failed"
    assert diagnostic.exception_type == "RuntimeError"
    assert re.fullmatch(
        r"claude_code_proxy\.providers\.codex\.provider:complete:\d+",
        diagnostic.location,
    )
    assert "secret response detail" not in repr(caught.value)


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

    async def local_counter(completion_request, telemetry=None):
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

    async def count_tokens(completion_request, telemetry=None):
        captured.append(completion_request)
        return 17

    provider = CodexProvider(Auth(), Client, token_counter=count_tokens)

    assert await provider.count_tokens(orchestration_request()) == 17
    assert captured[0].system[-1] == TextBlock(AGENT_COMPLETION_POLICY)
    assert captured[0].tools[0].description.endswith(AGENT_GUIDANCE)
    assert captured[0].tools[1].description.endswith(TASK_OUTPUT_GUIDANCE)
