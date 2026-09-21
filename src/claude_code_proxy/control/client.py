"""Synchronous client for the local control API."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import TypeAdapter, ValidationError

from ..limits import MAX_CONTROL_INTEGER
from ..text_safety import escaped_text_atom
from .schemas import (
    HealthResponse,
    PerformanceListResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
    ProcessIdentityResponse,
    SessionListResponse,
    SessionResponse,
)

_PROTOCOL_VERSION = 1
_DEFAULT_TIMEOUT_SECONDS = 2.0
_STREAM_TIMEOUT = httpx.Timeout(_DEFAULT_TIMEOUT_SECONDS, read=None)
_MAX_ERROR_DETAIL = 200
# Reset snapshots can be large; cap any single NDJSON record at 64 MiB.
_MAX_NDJSON_LINE_BYTES = 64 * 1024 * 1024
_PERFORMANCE_EVENT_ADAPTER = TypeAdapter(PerformanceStreamEvent)


class ControlError(Exception):
    """A control API failure safe to show to a CLI user."""


class ControlUnavailable(ControlError):
    """The local control API could not be reached."""

    def __init__(self, socket_path: Path, reason: str) -> None:
        self.socket_path = socket_path
        self.reason = _safe_text(reason)
        super().__init__(
            f"Control API unavailable at {_safe_text(str(socket_path))}: "
            f"{self.reason}"
        )


class IncompatibleProtocol(ControlError):
    """The endpoint does not implement the protocol required by this client."""


class PerformanceEventStream(Iterator[PerformanceStreamEvent]):
    """Iterate events; use as a context manager when stopping early."""

    def __init__(self, events: Generator[PerformanceStreamEvent, None, None]) -> None:
        self._events = events
        self._closed = False

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> PerformanceStreamEvent:
        if self._closed:
            raise StopIteration
        try:
            return next(self._events)
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the response-owning generator once."""
        if self._closed:
            return
        self._closed = True
        self._events.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ControlClient:
    """Own an HTTPX client connected to one control Unix socket."""

    def __init__(
        self,
        socket_path: Path,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.TimeoutTypes = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.socket_path = socket_path.expanduser().absolute()
        if transport is None:
            transport = httpx.HTTPTransport(uds=str(self.socket_path))
        self._client = httpx.Client(
            base_url="http://control",
            transport=transport,
            timeout=timeout,
        )
        self._closed = False
        self._active_streams: set[httpx.Response] = set()

    def health(self, *, _required_capability: str = "sessions") -> HealthResponse:
        """Return validated health and protocol information."""
        payload = self._request_json("/v1/health")
        version = payload.get("protocol_version")
        if type(version) is not int or version != _PROTOCOL_VERSION:
            displayed = "missing" if version is None else _safe_text(repr(version))
            raise IncompatibleProtocol(
                f"Control API protocol version {displayed} is incompatible; "
                f"expected {_PROTOCOL_VERSION}"
            )
        if "capabilities" not in payload:
            _raise_missing_capability(_required_capability)
        try:
            response = HealthResponse.model_validate(payload)
            _validate_health_numbers(response)
            return response
        except (ValidationError, ValueError) as error:
            raise ControlError("Control API returned an invalid health response") from error

    def sessions(self, filters: Sequence[str] = ()) -> SessionListResponse:
        """Negotiate capabilities and return validated session snapshots."""
        health = self.health()
        _require_capability(health, "sessions")
        params = [("filter", entry) for entry in filters]
        payload = self._request_json("/v1/sessions", params=params)
        try:
            response = SessionListResponse.model_validate(payload)
            return _normalized_sessions(response)
        except (ValidationError, ValueError) as error:
            raise ControlError("Control API returned an invalid sessions response") from error

    def performance(self, filters: Sequence[str] = ()) -> PerformanceListResponse:
        """Negotiate capabilities and return validated performance snapshots."""
        health = self.health(_required_capability="performance")
        _require_capability(health, "performance")
        params = [("filter", entry) for entry in filters]
        message = "Control API returned an invalid performance response"
        payload = self._request_json(
            "/v1/performance",
            params=params,
            missing_endpoint="performance",
            invalid_json_message=message,
            suppress_invalid_json_cause=True,
        )
        try:
            return PerformanceListResponse.model_validate(payload)
        except (ValidationError, ValueError) as error:
            raise ControlError(message) from None

    def performance_events(
        self,
        filters: Sequence[str] = (),
        *,
        after: int | None = None,
        process: ProcessIdentityResponse | None = None,
    ) -> PerformanceEventStream:
        """Return a closeable iterator over validated performance events."""
        if after is not None:
            _require_control_integer(after)
        events = self._performance_event_stream(filters, after, process)
        return PerformanceEventStream(events)

    def _performance_event_stream(
        self,
        filters: Sequence[str],
        after: int | None,
        process: ProcessIdentityResponse | None,
    ) -> Generator[PerformanceStreamEvent, None, None]:
        health = self.health(_required_capability="performance")
        _require_capability(health, "performance")
        _require_capability(health, "performance_events")
        params = _performance_event_params(filters, after, process)
        self._ensure_open()
        try:
            with self._client.stream(
                "GET",
                "/v1/performance/events",
                params=params,
                timeout=_STREAM_TIMEOUT,
            ) as response:
                self._active_streams.add(response)
                try:
                    _validate_stream_response(response)
                    yield from _validated_performance_events(
                        response, after=after, process=process
                    )
                finally:
                    self._active_streams.discard(response)
        except httpx.RequestError as error:
            raise ControlUnavailable(self.socket_path, str(error)) from error

    def close(self) -> None:
        """Close active responses and the owned HTTP client once."""
        if self._closed:
            return
        self._closed = True
        try:
            for response in tuple(self._active_streams):
                response.close()
        finally:
            self._active_streams.clear()
            self._client.close()

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _request_json(
        self,
        path: str,
        *,
        params: list[tuple[str, str]] | None = None,
        missing_endpoint: str | None = None,
        invalid_json_message: str = "Control API returned invalid JSON",
        suppress_invalid_json_cause: bool = False,
    ) -> dict[str, Any]:
        self._ensure_open()
        try:
            response = self._client.get(path, params=params)
        except httpx.RequestError as error:
            raise ControlUnavailable(self.socket_path, str(error)) from error
        endpoint = "health" if path == "/v1/health" else missing_endpoint
        if endpoint is not None and response.status_code == 404:
            raise IncompatibleProtocol(
                f"Control API {endpoint} endpoint is missing (HTTP 404)"
            )
        if not response.is_success:
            raise ControlError(_http_error_message(response))
        try:
            payload = response.json()
        except (ValueError, RecursionError) as error:
            if suppress_invalid_json_cause:
                raise ControlError(invalid_json_message) from None
            raise ControlError(invalid_json_message) from error
        if not isinstance(payload, dict):
            raise ControlError(invalid_json_message)
        return payload

    def _ensure_open(self) -> None:
        if self._closed:
            raise ControlError("Control client is closed")


def _validated_performance_events(
    response: httpx.Response,
    *,
    after: int | None,
    process: ProcessIdentityResponse | None,
) -> Iterator[PerformanceStreamEvent]:
    expected_process = process if after is not None else None
    last_sequence = after
    for payload in _iter_ndjson_objects(response):
        try:
            event = _PERFORMANCE_EVENT_ADAPTER.validate_python(payload)
        except (ValidationError, ValueError) as error:
            raise _invalid_performance_stream() from None
        if isinstance(event, PerformanceResetResponse):
            if _reset_rewinds_stream(event, expected_process, last_sequence):
                raise _invalid_performance_stream()
            expected_process = event.process
            last_sequence = event.sequence
            yield event
            continue
        if expected_process is None or last_sequence is None:
            raise _invalid_performance_stream()
        if event.process != expected_process:
            raise _invalid_performance_stream()
        if event.sequence != last_sequence + 1:
            raise _invalid_performance_stream()
        last_sequence = event.sequence
        yield event


def _reset_rewinds_stream(
    event: PerformanceResetResponse,
    expected_process: ProcessIdentityResponse | None,
    last_sequence: int | None,
) -> bool:
    return (
        expected_process is not None
        and last_sequence is not None
        and event.process == expected_process
        and event.sequence <= last_sequence
    )


def _iter_ndjson_objects(response: httpx.Response) -> Iterator[dict[str, Any]]:
    line = bytearray()
    for chunk in response.iter_raw():
        offset = 0
        while offset < len(chunk):
            newline = chunk.find(b"\n", offset)
            end = len(chunk) if newline < 0 else newline
            _extend_ndjson_line(line, chunk, offset, end, newline >= 0)
            if newline < 0:
                break
            if line.endswith(b"\r"):
                line.pop()
            if line:
                yield _parse_ndjson_object(bytes(line))
            line.clear()
            offset = newline + 1
    if line:
        if line.endswith(b"\r"):
            line.pop()
        if line:
            yield _parse_ndjson_object(bytes(line))


def _extend_ndjson_line(
    line: bytearray, chunk: bytes, start: int, end: int, complete: bool
) -> None:
    added = end - start
    total = len(line) + added
    last = chunk[end - 1] if added else line[-1] if line else None
    content_length = total - int(complete and last == 13)
    pending_cr = not complete and total == _MAX_NDJSON_LINE_BYTES + 1 and last == 13
    if content_length > _MAX_NDJSON_LINE_BYTES and not pending_cr:
        raise _invalid_performance_stream()
    line.extend(memoryview(chunk)[start:end])


def _parse_ndjson_object(line: bytes) -> dict[str, Any]:
    try:
        text = line.decode("utf-8", errors="strict")
        payload = json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise _invalid_performance_stream() from None
    if not isinstance(payload, dict):
        raise _invalid_performance_stream()
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _performance_event_params(
    filters: Sequence[str],
    after: int | None,
    process: ProcessIdentityResponse | None,
) -> list[tuple[str, str]]:
    params = [("filter", entry) for entry in filters]
    if after is None:
        return params
    params.append(("after", str(after)))
    if process is not None:
        params.extend(
            (
                ("pid", str(process.pid)),
                ("started_at", process.started_at.isoformat()),
            )
        )
    return params


def _validate_stream_response(response: httpx.Response) -> None:
    if response.status_code == 404:
        raise IncompatibleProtocol(
            "Control API performance events endpoint is missing (HTTP 404)"
        )
    if not response.is_success:
        raise ControlError(f"Control API returned HTTP {response.status_code}")
    media_type = response.headers.get("content-type", "").split(";", 1)[0]
    if media_type.strip().lower() != "application/x-ndjson":
        raise _invalid_performance_stream()
    content_encoding = response.headers.get("content-encoding")
    if content_encoding and content_encoding.strip().lower() != "identity":
        raise _invalid_performance_stream()


def _invalid_performance_stream() -> ControlError:
    return ControlError("Control API returned an invalid performance event stream")


def _require_capability(health: HealthResponse, name: str) -> None:
    if name not in health.capabilities:
        _raise_missing_capability(name)


def _raise_missing_capability(name: str) -> None:
    raise IncompatibleProtocol(
        f"Control API does not advertise the required {name} capability"
    )


def _validate_health_numbers(response: HealthResponse) -> None:
    _require_control_integer(response.pid, minimum=1)
    _require_control_integer(response.sessions.active)
    _require_control_integer(response.sessions.retained)
    _require_control_integer(response.inactive_limit)
    if response.sessions.active > response.sessions.retained:
        raise ValueError("active sessions exceed retained sessions")
    if not math.isfinite(response.uptime_seconds) or response.uptime_seconds < 0:
        raise ValueError("uptime_seconds must be finite and non-negative")


def _validate_session_numbers(session: SessionResponse) -> None:
    _require_control_integer(session.active_requests)
    _require_control_integer(session.requests)
    if session.active_requests > session.requests:
        raise ValueError("active requests exceed total requests")
    if session.context_window is not None:
        _require_control_integer(session.context_window, minimum=1)
    if not math.isfinite(session.elapsed_seconds) or session.elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative")


def _require_control_integer(value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= MAX_CONTROL_INTEGER:
        raise ValueError("integer is outside the control protocol range")


def _normalized_sessions(response: SessionListResponse) -> SessionListResponse:
    sessions = []
    for session in response.sessions:
        _validate_session_numbers(session)
        sessions.append(
            session.model_copy(
                update={
                    "first_seen": _utc_datetime(session.first_seen),
                    "last_seen": _utc_datetime(session.last_seen),
                }
            )
        )
    return response.model_copy(
        update={
            "captured_at": _utc_datetime(response.captured_at),
            "sessions": tuple(sessions),
        }
    )


def _utc_datetime(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    try:
        return value.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("datetime is outside the supported UTC range") from error


def _http_error_message(
    response: httpx.Response, *, content: bytes | None = None
) -> str:
    message = f"Control API returned HTTP {response.status_code}"
    content_type = response.headers.get("content-type", "").lower()
    if "json" not in content_type:
        return message
    try:
        payload = response.json() if content is None else json.loads(content)
    except (ValueError, RecursionError):
        return message
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if not isinstance(detail, str) or not detail:
        return message
    return f"{message}: {_safe_text(detail, _MAX_ERROR_DETAIL)}"


def _safe_text(value: str, limit: int = _MAX_ERROR_DETAIL) -> str:
    atoms = [escaped_text_atom(character) for character in value]
    escaped = "".join(atoms)
    if len(escaped) <= limit:
        return escaped

    marker = "..."
    budget = limit - len(marker)
    selected = []
    selected_length = 0
    for atom in atoms:
        if selected_length + len(atom) > budget:
            break
        selected.append(atom)
        selected_length += len(atom)
    return "".join(selected) + marker
