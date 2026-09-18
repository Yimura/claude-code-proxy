"""Synchronous client for the local control API."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
import math
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import ValidationError

from .schemas import HealthResponse, SessionListResponse, SessionResponse

MAX_CONTROL_INTEGER = 2**63 - 1
_PROTOCOL_VERSION = 1
_DEFAULT_TIMEOUT_SECONDS = 2.0
_MAX_ERROR_DETAIL = 200


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

    def health(self) -> HealthResponse:
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
            raise IncompatibleProtocol(
                "Control API does not advertise the required sessions capability"
            )
        try:
            response = HealthResponse.model_validate(payload)
            _validate_health_numbers(response)
            return response
        except (ValidationError, ValueError) as error:
            raise ControlError("Control API returned an invalid health response") from error

    def sessions(self, filters: Sequence[str] = ()) -> SessionListResponse:
        """Negotiate capabilities and return validated session snapshots."""
        health = self.health()
        if "sessions" not in health.capabilities:
            raise IncompatibleProtocol(
                "Control API does not advertise the required sessions capability"
            )
        params = [("filter", entry) for entry in filters]
        payload = self._request_json("/v1/sessions", params=params)
        try:
            response = SessionListResponse.model_validate(payload)
            return _normalized_sessions(response)
        except (ValidationError, ValueError) as error:
            raise ControlError("Control API returned an invalid sessions response") from error

    def close(self) -> None:
        """Close the owned HTTP client once."""
        if self._closed:
            return
        self._closed = True
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
    ) -> dict[str, Any]:
        self._ensure_open()
        try:
            response = self._client.get(path, params=params)
        except httpx.RequestError as error:
            raise ControlUnavailable(self.socket_path, str(error)) from error
        if not response.is_success:
            raise ControlError(_http_error_message(response))
        try:
            payload = response.json()
        except (ValueError, RecursionError) as error:
            raise ControlError("Control API returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ControlError("Control API returned invalid JSON")
        return payload

    def _ensure_open(self) -> None:
        if self._closed:
            raise ControlError("Control client is closed")


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


def _http_error_message(response: httpx.Response) -> str:
    message = f"Control API returned HTTP {response.status_code}"
    content_type = response.headers.get("content-type", "").lower()
    if "json" not in content_type:
        return message
    try:
        payload = response.json()
    except (ValueError, RecursionError):
        return message
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if not isinstance(detail, str) or not detail:
        return message
    return f"{message}: {_safe_text(detail, _MAX_ERROR_DETAIL)}"


def _safe_text(value: str, limit: int = _MAX_ERROR_DETAIL) -> str:
    atoms = [_safe_atom(character) for character in value]
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


def _safe_atom(character: str) -> str:
    code = ord(character)
    if code < 0x20 or 0x7F <= code <= 0x9F:
        return f"\\x{code:02x}"
    if character.isprintable():
        return character
    if code <= 0xFFFF:
        return f"\\u{code:04x}"
    return f"\\U{code:08x}"
