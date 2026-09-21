"""Codex subscription HTTP provider."""

import asyncio
from dataclasses import dataclass
import json

import httpx

from ...domain.models import (
    CompletionRequest,
    StreamComplete,
    StreamError,
    StreamStart,
)
from ...performance import ProviderTelemetry
from ...failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
    unexpected_failure_diagnostic,
)
from ..base import (
    ProviderError,
    protocol_error,
    public_error,
    scalar_provider_code,
    stream_error_from_exception,
)
from .auth import CodexAuth
from .identity import CodexIdentity
from .orchestration import reconcile_codex_request
from .translation import CodexEventTranslator, build_request, response_from_events

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_USER_AGENT = "opencode/latest/2.0.3/cli"
_MAX_ERROR_BODY_BYTES = 4096


@dataclass
class _CodexStreamState:
    terminal: object | None = None
    response_started: bool = False

    def record_context_failure(self, error: StreamError) -> None:
        if isinstance(self.terminal, StreamError):
            return
        self.terminal = error


class CodexProvider:
    name = "codex"

    def __init__(
        self,
        auth: CodexAuth,
        client_factory=httpx.AsyncClient,
        token_counter=None,
    ) -> None:
        self._auth = auth
        self._client_factory = client_factory
        self._token_counter = token_counter

    async def complete(
        self,
        request: CompletionRequest,
        telemetry: ProviderTelemetry | None = None,
    ):
        events = [
            event async for event in self.stream(request, telemetry=telemetry)
        ]
        error = next((event for event in events if isinstance(event, StreamError)), None)
        if error:
            raise ProviderError(
                error.message,
                provider=error.provider or self.name,
                status_code=error.status_code or 500,
                diagnostic=error.diagnostic,
            )
        try:
            return response_from_events(request, events)
        except Exception as error:
            raise self._translation_error(
                error, "response_translation_failed"
            ) from error

    async def stream(self, request: CompletionRequest, telemetry: ProviderTelemetry | None = None):
        try:
            request, identity, payload = self._prepare_request(request)
        except ProviderError as error:
            yield stream_error_from_exception(error, provider=self.name)
            return

        try:
            access_token, account_id = await self._auth_credentials()
        except ProviderError as error:
            yield stream_error_from_exception(error, provider=self.name)
            return

        state = _CodexStreamState()
        external_signal = None
        try:
            async with self._client_factory(
                timeout=httpx.Timeout(300.0, connect=30.0)
            ) as client:
                inner = self._stream_with_client(
                    client,
                    identity,
                    payload,
                    access_token,
                    account_id,
                    state,
                )
                try:
                    async for event in inner:
                        yield event
                except (GeneratorExit, asyncio.CancelledError) as signal:
                    external_signal = signal
                    raise
                finally:
                    try:
                        await inner.aclose()
                    except (GeneratorExit, asyncio.CancelledError):
                        if external_signal is None:
                            raise
                    except Exception:
                        if external_signal is None:
                            raise
        except (GeneratorExit, asyncio.CancelledError) as signal:
            if external_signal is not None:
                raise external_signal
            raise signal
        except Exception as error:
            if external_signal is not None:
                raise external_signal
            stage = (
                FailureStage.STREAM
                if state.response_started
                else FailureStage.REQUEST
            )
            code = (
                "stream_cleanup_failed"
                if state.response_started
                else "request_failed"
            )
            state.record_context_failure(
                self._error_event(error, stage=stage, code=code)
            )

        if state.terminal is None:
            state.terminal = self._error_event(
                RuntimeError(),
                stage=FailureStage.STREAM,
                code="stream_failed",
            )
        yield state.terminal

    def _prepare_request(self, request: CompletionRequest):
        try:
            request = reconcile_codex_request(request)
            identity = CodexIdentity.from_client(request.client_identity)
            return request, identity, build_request(request, identity)
        except Exception as error:
            raise self._translation_error(
                error, "request_translation_failed"
            ) from error

    async def _stream_with_client(
        self,
        client,
        identity: CodexIdentity,
        payload,
        access_token: str,
        account_id: str,
        state: _CodexStreamState,
    ):
        for attempt in range(2):
            retry_rejected = False
            external_signal = None
            headers = self._build_headers(access_token, account_id, identity)
            try:
                response_context = client.stream(
                    "POST",
                    CODEX_RESPONSES_URL,
                    headers=headers,
                    json=payload,
                )
                async with response_context as response:
                    try:
                        if response.status_code == 401 and attempt == 0:
                            retry_rejected = True
                        elif response.status_code == 401:
                            state.terminal = self._error_event(
                                self._credentials_rejected_error(),
                                stage=FailureStage.CREDENTIALS,
                                code="credentials_rejected",
                            )
                        elif response.status_code != 200:
                            state.terminal = self._error_event(
                                await self._http_error(response),
                                stage=FailureStage.RESPONSE,
                                code="http_error",
                            )
                        else:
                            state.response_started = True
                            async for event in self._consume_response(response):
                                if isinstance(
                                    event, (StreamComplete, StreamError)
                                ):
                                    state.terminal = event
                                else:
                                    yield event
                    except (GeneratorExit, asyncio.CancelledError) as signal:
                        external_signal = signal
                        raise
            except (GeneratorExit, asyncio.CancelledError) as signal:
                if external_signal is not None:
                    raise external_signal
                raise signal
            except Exception as error:
                if external_signal is not None:
                    raise external_signal
                stage = (
                    FailureStage.STREAM
                    if state.response_started
                    else FailureStage.REQUEST
                )
                code = (
                    "stream_cleanup_failed"
                    if state.response_started
                    else "request_failed"
                )
                state.record_context_failure(
                    self._error_event(error, stage=stage, code=code)
                )

            if state.terminal is not None:
                return
            if retry_rejected:
                try:
                    access_token, account_id = await self._auth_credentials(
                        access_token
                    )
                except ProviderError as error:
                    state.terminal = stream_error_from_exception(
                        error, provider=self.name
                    )
                    return

    async def _consume_response(self, response):
        translator = CodexEventTranslator()
        yield StreamStart()
        try:
            async for event_type, data in self._response_events(response):
                try:
                    events = translator.feed(event_type, data)
                except Exception as error:
                    error = self._translation_error(
                        error, "stream_chunk_translation_failed"
                    )
                    yield stream_error_from_exception(error, provider=self.name)
                    return
                for event in events:
                    yield event
                    if isinstance(event, StreamError):
                        return
        except ProviderError as error:
            yield stream_error_from_exception(error, provider=self.name)
            return
        except httpx.TimeoutException:
            error = self._transport_error(FailureStage.STREAM, timeout=True)
            yield stream_error_from_exception(error, provider=self.name)
            return
        except httpx.TransportError:
            error = self._transport_error(FailureStage.STREAM, timeout=False)
            yield stream_error_from_exception(error, provider=self.name)
            return
        except Exception as error:
            yield self._error_event(
                error,
                stage=FailureStage.STREAM,
                code="stream_failed",
            )
            return

        if not translator.completed:
            yield protocol_error("missing_response_completed", provider=self.name)
            return
        yield translator.finish()

    def _credentials_rejected_error(self) -> ProviderError:
        return ProviderError(
            "Authentication failed",
            provider=self.name,
            status_code=401,
            diagnostic=FailureDiagnostic(
                FailureCategory.AUTHENTICATION,
                FailureStage.CREDENTIALS,
                "credentials_rejected",
            ),
        )

    def _translation_error(
        self, error: Exception, code: str
    ) -> ProviderError:
        return ProviderError(
            "Internal server error",
            provider=self.name,
            status_code=500,
            diagnostic=unexpected_failure_diagnostic(
                error,
                category=FailureCategory.TRANSLATION,
                stage=FailureStage.PROVIDER_TRANSLATION,
                code=code,
            ),
        )

    def _error_event(
        self,
        error: Exception,
        *,
        stage: FailureStage,
        code: str,
    ) -> StreamError:
        if isinstance(error, ProviderError):
            provider_error = error
        elif isinstance(error, httpx.TimeoutException):
            provider_error = self._transport_error(stage, timeout=True)
        elif isinstance(error, httpx.TransportError):
            provider_error = self._transport_error(stage, timeout=False)
        else:
            provider_error = ProviderError(
                "Internal server error",
                provider=self.name,
                status_code=500,
                diagnostic=unexpected_failure_diagnostic(
                    error,
                    stage=stage,
                    code=code,
                ),
            )
        return stream_error_from_exception(provider_error, provider=self.name)

    def _transport_error(
        self, stage: FailureStage, *, timeout: bool
    ) -> ProviderError:
        status_code = 504 if timeout else 503
        _, message = public_error(status_code)
        return ProviderError(
            message,
            provider=self.name,
            status_code=status_code,
            diagnostic=FailureDiagnostic(
                FailureCategory.TRANSPORT,
                stage,
                "timeout" if timeout else "transport_error",
            ),
        )

    async def _auth_credentials(
        self, rejected_access: str | None = None
    ) -> tuple[str, str]:
        try:
            if rejected_access is None:
                return await self._auth.get_auth()
            return await self._auth.recover_rejected(rejected_access)
        except Exception as error:
            code = (
                "credential_load_failed"
                if rejected_access is None
                else "credential_recovery_failed"
            )
            raise ProviderError(
                "Authentication failed",
                provider=self.name,
                status_code=401,
                diagnostic=FailureDiagnostic(
                    FailureCategory.AUTHENTICATION,
                    FailureStage.CREDENTIALS,
                    code,
                ),
            ) from error

    async def count_tokens(
        self,
        request: CompletionRequest,
        telemetry: ProviderTelemetry | None = None,
    ) -> int:
        if self._token_counter is None:
            return 1000
        return await self._token_counter(
            reconcile_codex_request(request), telemetry=telemetry
        )

    async def _http_error(self, response) -> ProviderError:
        status_code = response.status_code
        _, message = public_error(status_code)
        category = (
            FailureCategory.AUTHENTICATION
            if status_code in {401, 403}
            else FailureCategory.UPSTREAM_HTTP
        )
        status_error = ProviderError(
            message,
            provider=self.name,
            status_code=status_code,
            diagnostic=FailureDiagnostic(
                category,
                FailureStage.RESPONSE,
                "http_error",
            ),
        )
        try:
            provider_code = await self._provider_error_code(response)
        except Exception:
            return status_error
        if provider_code is None:
            return status_error
        return ProviderError(
            message,
            provider=self.name,
            status_code=status_code,
            diagnostic=FailureDiagnostic(
                category,
                FailureStage.RESPONSE,
                "http_error",
                provider_code,
            ),
        )

    async def _provider_error_code(self, response) -> str | None:
        """Optionally enrich errors from a declared, bounded raw body.

        Extraction requires an uncompressed response with a valid Content-Length
        no larger than the retained-byte limit. HTTPX may allocate an oversized
        transport chunk before yielding it; this helper does not copy such a chunk
        and never retains more than the declared bounded body.
        """
        content_encoding = response.headers.get("content-encoding")
        if content_encoding and content_encoding.strip().lower() != "identity":
            return None

        content_length = response.headers.get("content-length")
        if (
            content_length is None
            or not content_length.isascii()
            or not content_length.isdecimal()
        ):
            return None
        declared_length = int(content_length)
        if declared_length < 0 or declared_length > _MAX_ERROR_BODY_BYTES:
            return None

        body = bytearray()
        async for chunk in response.aiter_raw():
            remaining = declared_length - len(body)
            if len(chunk) > remaining:
                return None
            body.extend(chunk)
        if len(body) != declared_length:
            return None

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if not isinstance(error, dict):
            return None
        provider_code = scalar_provider_code(error.get("code"))
        if provider_code is not None:
            return provider_code
        return scalar_provider_code(error.get("type"))

    async def _response_events(self, response):
        event_type = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                event_type = line[7:].strip()
                continue
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                data = json.loads(raw)
            except ValueError as error:
                raise self._protocol_error("malformed_sse_data") from error
            if not _valid_event_payload(event_type or "", data):
                raise self._protocol_error("invalid_event_payload")
            yield event_type or "", data

    def _protocol_error(self, code: str) -> ProviderError:
        return ProviderError(
            "Internal server error",
            provider=self.name,
            status_code=500,
            diagnostic=FailureDiagnostic(
                FailureCategory.PROVIDER_PROTOCOL,
                FailureStage.STREAM,
                code,
            ),
        )

    def _build_headers(
        self,
        access_token: str,
        account_id: str,
        identity: CodexIdentity,
    ) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": CODEX_USER_AGENT,
            "originator": "opencode",
            "x-codex-beta-features": "remote_compaction_v2",
            "chatgpt-account-id": account_id,
            "session-id": identity.session_id,
            "thread-id": identity.thread_id,
            "x-client-request-id": identity.thread_id,
        }
        if identity.parent_thread_id is not None:
            headers["x-codex-parent-thread-id"] = identity.parent_thread_id
        return headers


def _valid_event_payload(event_type: str, data: object) -> bool:
    if not isinstance(data, dict):
        return False

    if event_type == "response.output_text.delta":
        return isinstance(data.get("delta", ""), str)

    if event_type == "response.content_part.delta":
        delta = data.get("delta", "")
        if isinstance(delta, str):
            return True
        return isinstance(delta, dict) and isinstance(delta.get("text", ""), str)

    if event_type in {
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    }:
        field = (
            "delta"
            if event_type.endswith(".delta")
            else "arguments"
        )
        return _valid_output_index(data) and isinstance(data.get(field, ""), str)

    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = data.get("item", {})
        if not _valid_output_index(data) or not isinstance(item, dict):
            return False
        if item.get("type") != "function_call":
            return True
        return all(
            value is None or isinstance(value, str)
            for value in (
                item.get("call_id"),
                item.get("id"),
                item.get("name"),
            )
        )

    if event_type in {
        "response.completed",
        "response.incomplete",
        "response.failed",
    }:
        return "response" not in data or isinstance(data["response"], dict)

    return True


def _valid_output_index(data: dict) -> bool:
    output_index = data.get("output_index", 0)
    return isinstance(output_index, int) and not isinstance(output_index, bool)
