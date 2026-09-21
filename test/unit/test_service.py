import asyncio
import logging
from dataclasses import replace

import pytest
from claude_code_proxy.config import ModelConfig, ModelDefinition
from claude_code_proxy.domain.models import ClientIdentity, CompletionRequest, CompletionResponse, Message, StreamComplete, StreamError, TextBlock, TextDelta, TokenUsage
from claude_code_proxy.failures import FailureCategory, FailureDiagnostic, FailureStage
from claude_code_proxy.model_mapping import ModelResolver
from claude_code_proxy.performance import notify_telemetry
from claude_code_proxy.providers.base import ProviderError
from claude_code_proxy.reasoning import MappingEntry, ReasoningPolicy
from claude_code_proxy.service import ProxyService


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.last_request = None
        self.last_telemetry = None

    async def complete(self, request, telemetry=None):
        self.last_request = request
        self.last_telemetry = telemetry
        return CompletionResponse("id", request.response_model, (TextBlock("ok"),), "end_turn", TokenUsage(1, 1))
    async def stream(self, request, telemetry=None):
        self.last_request = request
        self.last_telemetry = telemetry
        yield StreamComplete("end_turn", TokenUsage(1, 1))
    async def count_tokens(self, request, telemetry=None):
        self.last_request = request
        self.last_telemetry = telemetry
        return 7


class LegacyProvider:
    name = "legacy"

    def __init__(self):
        self.response = CompletionResponse(
            "legacy-id",
            "claude-sonnet",
            (TextBlock("ok"),),
            "end_turn",
            TokenUsage(1, 1),
        )
        self.event = StreamComplete("end_turn", TokenUsage(1, 1))

    async def complete(self, request):
        return self.response

    async def stream(self, request):
        yield self.event

    async def count_tokens(self, request):
        return 11


def make_request(model="claude-sonnet", **changes):
    request = CompletionRequest(
        original_model=model,
        model=model,
        response_model=model,
        max_tokens=100,
        messages=(Message("user", (TextBlock("hi"),)),),
        reasoning=ReasoningPolicy(None, None),
    )
    return replace(request, **changes)


def test_prepare_preserves_client_identity():
    provider = FakeProvider()
    service = ProxyService(
        ModelResolver(ModelConfig({}, {}, {})), "codex", provider, provider
    )
    identity = ClientIdentity("session-1", "agent-1", "parent-1")

    prepared = service.prepare(make_request(client_identity=identity))

    assert prepared.client_identity is identity


@pytest.mark.asyncio
async def test_codex_transport_selects_codex_for_resolved_openai_model():
    lite, codex = FakeProvider(), FakeProvider()
    service = ProxyService(ModelResolver(ModelConfig(
        {"sol": ModelDefinition(target="openai/gpt-5.6-sol", context_window=1_000_000)},
        {"big": "sol"},
        {"sonnet": MappingEntry(tier="big", effort="high")},
    )), "codex", lite, codex)
    await service.complete(make_request())
    assert codex.last_request.model == "openai/gpt-5.6-sol"
    assert codex.last_request.reasoning == ReasoningPolicy(True, "high")
    assert lite.last_request is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini/gemini-2.5-pro", "anthropic/claude-opus-5"])
async def test_non_openai_models_use_litellm(model):
    lite, codex = FakeProvider(), FakeProvider()
    service = ProxyService(ModelResolver(ModelConfig({}, {}, {})), "codex", lite, codex)
    await service.complete(make_request(model))
    assert lite.last_request.model == model
    assert codex.last_request is None


@pytest.mark.asyncio
async def test_count_tokens_uses_same_selection():
    lite, codex = FakeProvider(), FakeProvider()
    service = ProxyService(ModelResolver(ModelConfig({}, {}, {})), "codex", lite, codex)
    assert await service.count_tokens(make_request("openai/gpt-5.6-sol")) == 7
    assert codex.last_request is not None


@pytest.mark.asyncio
async def test_stream_uses_same_selection():
    lite, codex = FakeProvider(), FakeProvider()
    service = ProxyService(ModelResolver(ModelConfig({}, {}, {})), "litellm", lite, codex)
    assert [event async for event in service.stream(make_request())]
    assert lite.last_request is not None


class EventProvider(FakeProvider):
    def __init__(self, events=(), error=None):
        super().__init__()
        self.events = events
        self.error = error

    async def stream(self, request, telemetry=None):
        self.last_request = request
        self.last_telemetry = telemetry
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event


async def service_events(provider):
    service = ProxyService(
        ModelResolver(ModelConfig({}, {}, {})), "litellm", provider, provider
    )
    return [event async for event in service.stream(make_request())]


@pytest.mark.asyncio
async def test_stream_converts_eof_without_terminal_to_protocol_error():
    events = await service_events(EventProvider([TextDelta("partial")]))

    assert events[0] == TextDelta("partial")
    assert events[-1] == StreamError(
        provider="fake",
        diagnostic=FailureDiagnostic(
            FailureCategory.PROVIDER_PROTOCOL,
            FailureStage.STREAM,
            "missing_terminal_event",
        ),
    )


@pytest.mark.asyncio
async def test_stream_converts_raised_exception_to_safe_error():
    events = await service_events(EventProvider(error=RuntimeError("secret body")))

    assert len(events) == 1
    stream_error = events[0]
    assert stream_error.error_type == "api_error"
    assert stream_error.message == "Internal server error"
    assert stream_error.retryable is True
    assert stream_error.provider == "fake"
    assert stream_error.diagnostic is not None
    assert stream_error.diagnostic.category == FailureCategory.INTERNAL
    assert stream_error.diagnostic.stage == FailureStage.STREAM
    assert stream_error.diagnostic.code == "unexpected_exception"
    assert stream_error.diagnostic.exception_type == "RuntimeError"
    assert stream_error.diagnostic.location.startswith(
        "claude_code_proxy.service:_validated_stream:"
    )
    assert "secret body" not in repr(events)


@pytest.mark.asyncio
async def test_stream_preserves_safe_provider_error_and_diagnostic():
    diagnostic = FailureDiagnostic(
        FailureCategory.TRANSPORT,
        FailureStage.REQUEST,
        "transport_error",
    )
    events = await service_events(
        EventProvider(
            error=ProviderError(
                "Provider temporarily unavailable",
                provider="upstream",
                status_code=503,
                diagnostic=diagnostic,
            )
        )
    )

    assert events == [
        StreamError(
            message="Provider temporarily unavailable",
            status_code=503,
            retryable=True,
            provider="upstream",
            diagnostic=diagnostic,
        )
    ]


class TerminalThenBlocksProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.closed = False
        self.waiting = asyncio.Event()

    async def stream(self, request, telemetry=None):
        self.last_telemetry = telemetry
        try:
            yield StreamComplete("end_turn", TokenUsage(1, 1))
            self.waiting.set()
            await asyncio.Event().wait()
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_stream_forwards_terminal_without_waiting_for_provider_eof():
    provider = TerminalThenBlocksProvider()
    service = ProxyService(
        ModelResolver(ModelConfig({}, {}, {})), "litellm", provider, provider
    )
    stream = service.stream(make_request())

    event = await asyncio.wait_for(anext(stream), timeout=0.1)

    assert event == StreamComplete("end_turn", TokenUsage(1, 1))
    assert provider.waiting.is_set() is False

    await stream.aclose()

    assert provider.closed is True


IDENTITY_SYSTEM = (
    TextBlock("You are Claude Code, Anthropic's official CLI for Claude."),
    TextBlock(
        "You are powered by the model named Opus 5. "
        "The exact model ID is claude-opus-5."
    ),
)
EXPECTED_MAPPED_IDENTITY = (
    TextBlock(
        "You are running inside Claude Code, Anthropic's coding-agent CLI harness."
    ),
    TextBlock(
        "The model generating this response is openai/gpt-5.6-sol, "
        "not an Anthropic Claude model."
    ),
)


def mapped_service(transport="codex"):
    lite, codex = FakeProvider(), FakeProvider()
    resolver = ModelResolver(
        ModelConfig(
            {
                "sol": ModelDefinition(
                    target="openai/gpt-5.6-sol", context_window=1_000_000
                )
            },
            {},
            {"opus": MappingEntry(model="sol", effort="high")},
        )
    )
    return ProxyService(resolver, transport, lite, codex), lite, codex


def identity_request():
    return make_request("claude-opus-5", system=IDENTITY_SYSTEM)


def test_prepare_reconciles_mapped_model_identity():
    service, _, _ = mapped_service()

    prepared = service.prepare(identity_request())

    assert prepared.original_model == "claude-opus-5"
    assert prepared.model == "openai/gpt-5.6-sol"
    assert prepared.response_model == "claude-opus-5[1m]"
    assert prepared.system == EXPECTED_MAPPED_IDENTITY


def test_prepare_carries_mapped_context_window():
    service, _, _ = mapped_service()

    prepared = service.prepare(identity_request())

    assert prepared.context_window == 1_000_000


@pytest.mark.asyncio
async def test_complete_dispatches_reconciled_identity_to_codex():
    service, _, codex = mapped_service()

    await service.complete(identity_request())

    assert codex.last_request.system == EXPECTED_MAPPED_IDENTITY


@pytest.mark.asyncio
async def test_stream_dispatches_reconciled_identity_to_litellm():
    service, lite, _ = mapped_service("litellm")

    assert [event async for event in service.stream(identity_request())]

    assert lite.last_request.system == EXPECTED_MAPPED_IDENTITY


@pytest.mark.asyncio
async def test_count_tokens_dispatches_reconciled_identity():
    service, _, codex = mapped_service()

    assert await service.count_tokens(identity_request()) == 7

    assert codex.last_request.system == EXPECTED_MAPPED_IDENTITY
    assert codex.last_request.response_model == "claude-opus-5[1m]"


def test_prepare_reconciles_identity_in_system_role_message():
    service, _, _ = mapped_service()
    request = make_request(
        "claude-opus-5",
        messages=(
            Message("user", (TextBlock("Who are you?"),)),
            Message("system", (IDENTITY_SYSTEM[1],)),
        ),
    )

    prepared = service.prepare(request)

    assert prepared.messages == (
        request.messages[0],
        Message("system", (EXPECTED_MAPPED_IDENTITY[1],)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["complete", "stream", "count_tokens"])
async def test_dispatches_reconciled_system_role_identity(operation):
    service, _, codex = mapped_service()
    request = make_request(
        "claude-opus-5",
        messages=(Message("system", (IDENTITY_SYSTEM[1],)),),
    )

    result = getattr(service, operation)(request)
    if operation == "stream":
        assert [event async for event in result]
    else:
        await result

    assert codex.last_request.messages == (
        Message("system", (EXPECTED_MAPPED_IDENTITY[1],)),
    )


class RecordingTelemetry:
    def __init__(self, raising=()):
        self.calls = []
        self._raising = set(raising)

    def _record(self, name, value=None):
        self.calls.append((name, value))
        if name in self._raising:
            raise RuntimeError("telemetry-secret-marker")

    def upstream_started(self):
        self._record("upstream_started")

    def upstream_finished(self):
        self._record("upstream_finished")

    def stream_event(self, event):
        self._record("stream_event", event)

    def response(self, response):
        self._record("response", response)

    def count_tokens(self, value):
        self._record("count_tokens", value)

    def mark_retries_supported(self):
        self._record("mark_retries_supported")

    def record_retry(self):
        self._record("record_retry")

    def set_reasoning_continuation(self, value):
        self._record("set_reasoning_continuation", value)


class LifecycleProvider(FakeProvider):
    def __init__(
        self,
        *,
        response=None,
        complete_error=None,
        count=7,
        count_error=None,
        events=(),
        stream_error=None,
    ):
        super().__init__()
        self.response_value = response or CompletionResponse(
            "response-id",
            "claude-sonnet",
            (TextBlock("ok"),),
            "end_turn",
            TokenUsage(1, 1),
        )
        self.complete_error = complete_error
        self.count = count
        self.count_error = count_error
        self.events = events
        self.stream_error = stream_error
        self.close_count = 0

    async def complete(self, request, telemetry=None):
        self.last_request = request
        self.last_telemetry = telemetry
        if self.complete_error is not None:
            raise self.complete_error
        return self.response_value

    async def count_tokens(self, request, telemetry=None):
        self.last_request = request
        self.last_telemetry = telemetry
        if self.count_error is not None:
            raise self.count_error
        return self.count

    async def stream(self, request, telemetry=None):
        self.last_request = request
        self.last_telemetry = telemetry
        try:
            if self.stream_error is not None:
                raise self.stream_error
            for event in self.events:
                yield event
        finally:
            self.close_count += 1


def lifecycle_service(provider):
    return ProxyService(
        ModelResolver(ModelConfig({}, {}, {})), "litellm", provider, provider
    )


@pytest.mark.asyncio
async def test_complete_without_telemetry_supports_legacy_provider_arity():
    provider = LegacyProvider()

    response = await lifecycle_service(provider).complete_prepared(make_request())

    assert response is provider.response


@pytest.mark.asyncio
async def test_count_tokens_without_telemetry_supports_legacy_provider_arity():
    provider = LegacyProvider()

    result = await lifecycle_service(provider).count_tokens_prepared(make_request())

    assert result == 11


@pytest.mark.asyncio
async def test_stream_without_telemetry_supports_legacy_provider_arity():
    provider = LegacyProvider()

    events = [
        event
        async for event in lifecycle_service(provider).stream_prepared(make_request())
    ]

    assert events == [provider.event]
    assert events[0] is provider.event


@pytest.mark.asyncio
async def test_complete_observes_lifecycle_in_order_and_forwards_telemetry():
    provider = LifecycleProvider()
    telemetry = RecordingTelemetry()

    response = await lifecycle_service(provider).complete_prepared(
        make_request(), telemetry=telemetry
    )

    assert response is provider.response_value
    assert provider.last_telemetry is not telemetry
    assert provider.last_telemetry is not None
    assert telemetry.calls == [
        ("upstream_started", None),
        ("response", response),
        ("upstream_finished", None),
    ]


@pytest.mark.asyncio
async def test_complete_provider_failure_preserves_error_and_finishes():
    error = ProviderError("safe", provider="fake", status_code=503)
    provider = LifecycleProvider(complete_error=error)
    telemetry = RecordingTelemetry()

    with pytest.raises(ProviderError) as caught:
        await lifecycle_service(provider).complete_prepared(
            make_request(), telemetry=telemetry
        )

    assert caught.value is error
    assert caught.value.status_code == 503
    assert telemetry.calls == [
        ("upstream_started", None),
        ("upstream_finished", None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_count_tokens_observes_lifecycle_and_preserves_outcome(fails):
    error = ProviderError("safe", provider="fake", status_code=429)
    provider = LifecycleProvider(count=23, count_error=error if fails else None)
    telemetry = RecordingTelemetry()
    operation = lifecycle_service(provider).count_tokens_prepared(
        make_request(), telemetry=telemetry
    )

    if fails:
        with pytest.raises(ProviderError) as caught:
            await operation
        assert caught.value is error
    else:
        assert await operation == 23

    assert provider.last_telemetry is not telemetry
    assert provider.last_telemetry is not None
    assert telemetry.calls == [
        ("upstream_started", None),
        ("upstream_finished", None),
    ]


@pytest.mark.asyncio
async def test_stream_observes_events_before_yield_without_replacing_them():
    first = TextDelta("first")
    second = StreamComplete("end_turn", TokenUsage(1, 2))
    provider = LifecycleProvider(events=(first, second))
    telemetry = RecordingTelemetry()

    events = [
        event
        async for event in lifecycle_service(provider).stream_prepared(
            make_request(), telemetry=telemetry
        )
    ]

    assert events[0] is first
    assert events[1] is second
    assert provider.last_telemetry is not telemetry
    assert provider.last_telemetry is not None
    assert provider.close_count == 1
    assert telemetry.calls == [
        ("upstream_started", None),
        ("stream_event", first),
        ("stream_event", second),
        ("upstream_finished", None),
    ]


@pytest.mark.asyncio
async def test_stream_observes_safe_error_created_from_provider_exception():
    provider = LifecycleProvider(stream_error=RuntimeError("provider secret"))
    telemetry = RecordingTelemetry()

    events = [
        event
        async for event in lifecycle_service(provider).stream_prepared(
            make_request(), telemetry=telemetry
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].message == "Internal server error"
    assert telemetry.calls == [
        ("upstream_started", None),
        ("stream_event", events[0]),
        ("upstream_finished", None),
    ]


@pytest.mark.asyncio
async def test_stream_early_close_closes_provider_and_finishes_once():
    provider = LifecycleProvider(
        events=(TextDelta("first"), TextDelta("unconsumed"))
    )
    telemetry = RecordingTelemetry()
    stream = lifecycle_service(provider).stream_prepared(
        make_request(), telemetry=telemetry
    )

    first = await anext(stream)
    await stream.aclose()
    await stream.aclose()

    assert first is provider.events[0]
    assert provider.close_count == 1
    assert telemetry.calls == [
        ("upstream_started", None),
        ("stream_event", first),
        ("upstream_finished", None),
    ]


class CancellingProvider(LifecycleProvider):
    async def stream(self, request, telemetry=None):
        self.last_telemetry = telemetry
        try:
            yield TextDelta("first")
            raise asyncio.CancelledError
        finally:
            self.close_count += 1


@pytest.mark.asyncio
async def test_stream_cancellation_propagates_and_finishes_once():
    provider = CancellingProvider()
    telemetry = RecordingTelemetry()
    stream = lifecycle_service(provider).stream_prepared(
        make_request(), telemetry=telemetry
    )

    first = await anext(stream)
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert provider.close_count == 1
    assert telemetry.calls == [
        ("upstream_started", None),
        ("stream_event", first),
        ("upstream_finished", None),
    ]


PROVIDER_CALLBACKS = (
    "mark_retries_supported",
    "record_retry",
    "set_reasoning_continuation",
)


class ProviderCallbackProvider(LifecycleProvider):
    @staticmethod
    def notify_provider_callbacks(telemetry):
        telemetry.mark_retries_supported()
        telemetry.record_retry()
        telemetry.set_reasoning_continuation("expected")

    async def complete(self, request, telemetry=None):
        self.notify_provider_callbacks(telemetry)
        return await super().complete(request, telemetry)

    async def count_tokens(self, request, telemetry=None):
        self.notify_provider_callbacks(telemetry)
        return await super().count_tokens(request, telemetry)

    async def stream(self, request, telemetry=None):
        self.notify_provider_callbacks(telemetry)
        inner = super().stream(request, telemetry)
        try:
            async for event in inner:
                yield event
        finally:
            await inner.aclose()


def assert_provider_callbacks_isolated(provider, telemetry, caplog):
    assert provider.last_telemetry is not telemetry
    assert provider.last_telemetry is not None
    adapter_calls = [
        name for name, _ in telemetry.calls if name in PROVIDER_CALLBACKS
    ]
    assert adapter_calls == list(PROVIDER_CALLBACKS)
    assert caplog.messages == ["telemetry callback failed"] * 3


@pytest.mark.asyncio
async def test_complete_isolates_provider_telemetry_callbacks(caplog):
    provider = ProviderCallbackProvider()
    telemetry = RecordingTelemetry(PROVIDER_CALLBACKS)

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.performance"):
        response = await lifecycle_service(provider).complete_prepared(
            make_request(), telemetry=telemetry
        )

    assert response is provider.response_value
    assert_provider_callbacks_isolated(provider, telemetry, caplog)


@pytest.mark.asyncio
async def test_count_tokens_isolates_provider_telemetry_callbacks(caplog):
    provider = ProviderCallbackProvider(count=23)
    telemetry = RecordingTelemetry(PROVIDER_CALLBACKS)

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.performance"):
        result = await lifecycle_service(provider).count_tokens_prepared(
            make_request(), telemetry=telemetry
        )

    assert result == 23
    assert_provider_callbacks_isolated(provider, telemetry, caplog)


@pytest.mark.asyncio
async def test_stream_isolates_provider_telemetry_callbacks(caplog):
    event = StreamComplete("end_turn", TokenUsage(1, 1))
    provider = ProviderCallbackProvider(events=(event,))
    telemetry = RecordingTelemetry(PROVIDER_CALLBACKS)

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.performance"):
        events = [
            item
            async for item in lifecycle_service(provider).stream_prepared(
                make_request(), telemetry=telemetry
            )
        ]

    assert events == [event]
    assert events[0] is event
    assert provider.close_count == 1
    assert_provider_callbacks_isolated(provider, telemetry, caplog)


@pytest.mark.asyncio
async def test_provider_callback_failures_preserve_provider_error(caplog):
    error = ProviderError("safe", provider="fake", status_code=503)
    provider = ProviderCallbackProvider(complete_error=error)
    telemetry = RecordingTelemetry(PROVIDER_CALLBACKS)

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.performance"):
        with pytest.raises(ProviderError) as caught:
            await lifecycle_service(provider).complete_prepared(
                make_request(), telemetry=telemetry
            )

    assert caught.value is error
    assert_provider_callbacks_isolated(provider, telemetry, caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_callback",
    ["upstream_started", "response", "upstream_finished"],
)
async def test_complete_callback_failures_are_isolated_and_safely_logged(
    failing_callback, caplog
):
    provider = LifecycleProvider()
    telemetry = RecordingTelemetry((failing_callback,))

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.performance"):
        response = await lifecycle_service(provider).complete_prepared(
            make_request(), telemetry=telemetry
        )

    assert response is provider.response_value
    assert [name for name, _ in telemetry.calls] == [
        "upstream_started",
        "response",
        "upstream_finished",
    ]
    assert caplog.messages == ["telemetry callback failed"]
    assert "telemetry-secret-marker" not in caplog.text
    assert "response-id" not in caplog.text
    assert "hi" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_callback",
    ["upstream_started", "stream_event", "upstream_finished"],
)
async def test_stream_callback_failure_keeps_event_and_cleanup(
    failing_callback, caplog
):
    event = StreamComplete("end_turn", TokenUsage(1, 1))
    provider = LifecycleProvider(events=(event,))
    telemetry = RecordingTelemetry((failing_callback,))

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.performance"):
        events = [
            item
            async for item in lifecycle_service(provider).stream_prepared(
                make_request(), telemetry=telemetry
            )
        ]

    assert events == [event]
    assert events[0] is event
    assert provider.close_count == 1
    assert [name for name, _ in telemetry.calls] == [
        "upstream_started",
        "stream_event",
        "upstream_finished",
    ]
    assert caplog.messages == ["telemetry callback failed"]
    assert "telemetry-secret-marker" not in caplog.text


class FactoryFailureProvider(FakeProvider):
    def __init__(self, error):
        super().__init__()
        self.error = error
        self.stream_calls = 0

    def stream(self, request, telemetry=None):
        self.stream_calls += 1
        raise self.error


@pytest.mark.asyncio
async def test_stream_factory_failure_is_lazy_observed_and_finished():
    provider = FactoryFailureProvider(RuntimeError("provider secret"))
    telemetry = RecordingTelemetry()

    stream = lifecycle_service(provider).stream_prepared(
        make_request(), telemetry=telemetry
    )

    assert provider.stream_calls == 0
    events = [event async for event in stream]
    assert provider.stream_calls == 1
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].message == "Internal server error"
    assert telemetry.calls == [
        ("upstream_started", None),
        ("stream_event", events[0]),
        ("upstream_finished", None),
    ]


@pytest.mark.asyncio
async def test_stream_start_cancellation_does_not_construct_provider_stream():
    provider = FactoryFailureProvider(AssertionError("must not construct"))

    class CancellingTelemetry(RecordingTelemetry):
        def upstream_started(self):
            raise asyncio.CancelledError

    stream = lifecycle_service(provider).stream_prepared(
        make_request(), telemetry=CancellingTelemetry()
    )

    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert provider.stream_calls == 0


class CloseFailureEvents:
    def __init__(self):
        self.emitted = False
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.emitted:
            raise StopAsyncIteration
        self.emitted = True
        return StreamComplete("end_turn", TokenUsage(1, 1))

    async def aclose(self):
        self.close_count += 1
        raise RuntimeError("cleanup failed")


class CloseFailureProvider(FakeProvider):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def stream(self, request, telemetry=None):
        self.last_telemetry = telemetry
        return self.events


@pytest.mark.asyncio
async def test_stream_notifies_finish_when_owned_iterator_close_raises():
    events = CloseFailureEvents()
    telemetry = RecordingTelemetry()
    stream = lifecycle_service(CloseFailureProvider(events)).stream_prepared(
        make_request(), telemetry=telemetry
    )

    event = await anext(stream)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await anext(stream)

    assert events.close_count == 1
    assert telemetry.calls == [
        ("upstream_started", None),
        ("stream_event", event),
        ("upstream_finished", None),
    ]


@pytest.mark.asyncio
async def test_none_telemetry_preserves_provider_results_and_event_identity():
    response_provider = LifecycleProvider()
    stream_event = StreamComplete("end_turn", TokenUsage(1, 1))
    stream_provider = LifecycleProvider(events=(stream_event,))

    response = await lifecycle_service(response_provider).complete_prepared(
        make_request(), telemetry=None
    )
    events = [
        event
        async for event in lifecycle_service(stream_provider).stream_prepared(
            make_request(), telemetry=None
        )
    ]

    assert response is response_provider.response_value
    assert events[0] is stream_event
    assert response_provider.last_telemetry is None
    assert stream_provider.last_telemetry is None


def test_notify_telemetry_logs_only_fixed_warning(caplog):
    marker = "callback-secret\n\x1b[31m" + "x" * 500
    hostile_error = type(marker, (Exception,), {})

    class HostileTelemetry:
        def upstream_started(self):
            raise hostile_error

    with caplog.at_level(logging.WARNING, logger="claude_code_proxy.performance"):
        notify_telemetry(HostileTelemetry(), "upstream_started")

    assert caplog.messages == ["telemetry callback failed"]
    assert marker not in caplog.text
    assert "callback-secret" not in caplog.text
    assert "\x1b" not in caplog.text
    assert len(caplog.messages[0]) == len("telemetry callback failed")


def test_notify_telemetry_does_not_swallow_cancellation():
    class CancelTelemetry:
        def upstream_started(self):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        notify_telemetry(CancelTelemetry(), "upstream_started")
