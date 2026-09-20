"""Application logging configuration and request correlation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

from .console_logging import SeverityFormatter as _SeverityFormatter
from .domain.models import ClientIdentity, StreamError, StreamEvent
from .failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
    retryable_status,
    unexpected_failure_diagnostic,
)
from .observability import SessionRegistry
from .providers.base import ProviderError
from .reasoning import ReasoningPolicy
from .text_safety import bounded_log_token, log_text

if TYPE_CHECKING:
    from .providers.codex.auth import CodexAccountIdentity

SESSION_HEADER = "x-claude-code-session-id"
AGENT_HEADER = "x-claude-code-agent-id"
PARENT_AGENT_HEADER = "x-claude-code-parent-agent-id"
FAILURE_LOGGED = "failure_logged"
REQUEST_LOG_CONTEXT = "request_log_context"
DIAGNOSTIC_FIELD_MAX_LENGTH = 128
SESSION_COLORS = (
    "\033[96m",
    "\033[94m",
    "\033[92m",
    "\033[95m",
    "\033[93m",
)
_RESET = "\033[0m"

logger = logging.getLogger(__name__)
readiness_logger = logging.getLogger(f"{__name__}.readiness")
session_logger = logging.getLogger(f"{__name__}.session")


class MessageFilter(logging.Filter):
    blocked_phrases = (
        "LiteLLM completion()",
        "HTTP Request:",
        "selected model name for cost calculation",
        "utils.py",
        "cost_calculator",
    )

    def filter(self, record):
        return not (
            isinstance(record.msg, str)
            and any(phrase in record.msg for phrase in self.blocked_phrases)
        )


@dataclass(frozen=True)
class SessionIdentity:
    label: str
    rendered: str
    is_new: bool


@dataclass(frozen=True)
class AgentIdentity:
    label: str
    rendered: str
    parent_label: str | None
    is_new: bool


@dataclass(frozen=True)
class RequestLogContext:
    session: SessionIdentity
    method: str
    endpoint: str
    original_model: str
    upstream_model: str
    provider: str
    effort: str
    agent: AgentIdentity | None = None


def _nonblank_header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None or not value.strip():
        return None
    return value


def client_identity_from_headers(headers: Mapping[str, str]) -> ClientIdentity:
    session_id = _nonblank_header(headers, SESSION_HEADER)
    agent_id = _nonblank_header(headers, AGENT_HEADER)
    parent_agent_id = (
        _nonblank_header(headers, PARENT_AGENT_HEADER)
        if agent_id is not None
        else None
    )
    return ClientIdentity(session_id, agent_id, parent_agent_id)


def palette_index(identifier: str) -> int:
    digest = hashlib.sha256(identifier.encode()).digest()
    return int.from_bytes(digest[:8], "big") % len(SESSION_COLORS)


def session_identity(
    public_id: str,
    *,
    request_scoped: bool,
    is_new: bool,
    environ: Mapping[str, str] | None = None,
) -> SessionIdentity:
    """Render an opaque registry ID without exposing its source identifier."""
    label = public_id[:12]
    rendered_label = label
    environment = os.environ if environ is None else environ
    if "NO_COLOR" not in environment:
        color = SESSION_COLORS[palette_index(public_id)]
        rendered_label = f"{color}{label}{_RESET}"
    scope = "request" if request_scoped else "session"
    return SessionIdentity(label, f"[{scope} {rendered_label}]", is_new)


def agent_identity(
    public_id: str | None,
    parent_public_id: str | None,
    *,
    is_new: bool,
    environ: Mapping[str, str] | None = None,
) -> AgentIdentity | None:
    if public_id is None:
        return None
    label = public_id[:12]
    rendered_label = label
    environment = os.environ if environ is None else environ
    if "NO_COLOR" not in environment:
        color = SESSION_COLORS[palette_index(public_id)]
        rendered_label = f"{color}{label}{_RESET}"
    return AgentIdentity(
        label=label,
        rendered=f"[agent {rendered_label}]",
        parent_label=(
            parent_public_id[:12] if parent_public_id is not None else None
        ),
        is_new=is_new,
    )


def _render_correlation(
    session: SessionIdentity,
    agent: AgentIdentity | None,
) -> str:
    parts = [session.rendered]
    if agent is not None:
        parts.append(agent.rendered)
        if agent.parent_label is not None:
            parts.append(f"parent={agent.parent_label}")
    return " ".join(parts)


def _correlation(context: RequestLogContext) -> str:
    return _render_correlation(context.session, context.agent)


def effective_effort(policy: ReasoningPolicy) -> str:
    if policy.enabled is False:
        return "none"
    return policy.effort or "default"


def configure_logging() -> None:
    root = logging.getLogger()
    root.filters[:] = [
        item for item in root.filters if not isinstance(item, MessageFilter)
    ]
    handler = logging.StreamHandler()
    handler.addFilter(MessageFilter())
    handler.setFormatter(_SeverityFormatter())
    logging.basicConfig(
        level=logging.WARN, handlers=[handler], force=True
    )
    root.filters[:] = [
        item for item in root.filters if not isinstance(item, MessageFilter)
    ]
    session_logger.setLevel(logging.INFO)
    readiness_logger.setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log_startup_summary(
    transport: str, identity: CodexAccountIdentity | None = None
) -> None:
    """Report safe OpenAI startup metadata before public requests can run."""
    readiness_logger.info("OpenAI transport: %s", log_text(transport))
    if identity is None:
        return
    account_id = log_text(identity.account_id)
    source = log_text(identity.source)
    if identity.masked_email is None:
        readiness_logger.info("OpenCode account: %s (%s)", account_id, source)
    else:
        readiness_logger.info(
            "OpenCode account: %s [%s] (%s)",
            log_text(identity.masked_email),
            account_id,
            source,
        )
    readiness_logger.info(
        "To use another account, stop the proxy, switch the active OpenAI "
        "account in OpenCode, and restart."
    )


def log_proxy_ready(host: str, port: int, socket_path: Path) -> None:
    """Report the production endpoint only after both servers are live."""
    readiness_logger.info(
        "Proxy ready host=%s port=%s control_socket=%s",
        log_text(host),
        port,
        log_text(str(socket_path)),
    )


def _request_fields(
    context: RequestLogContext,
    *,
    include_agent: bool = True,
) -> tuple[object, ...]:
    correlation = (
        _correlation(context)
        if include_agent
        else context.session.rendered
    )
    return (
        correlation,
        log_text(context.method),
        log_text(context.endpoint),
        log_text(context.original_model),
        log_text(context.upstream_model),
        bounded_log_token(
            context.provider, max_length=DIAGNOSTIC_FIELD_MAX_LENGTH
        ),
        bounded_log_token(
            context.effort, max_length=DIAGNOSTIC_FIELD_MAX_LENGTH
        ),
    )


def log_session_started(context: RequestLogContext) -> None:
    session_logger.info(
        "[NEW] %s %s %s %s → %s provider=%s effort=%s",
        *_request_fields(context, include_agent=False),
    )


def log_agent_started(context: RequestLogContext) -> None:
    session_logger.info(
        "[NEW AGENT] %s %s %s %s → %s provider=%s effort=%s",
        *_request_fields(context),
    )


def _structured_log_token(value: object) -> str:
    return bounded_log_token(str(value), max_length=DIAGNOSTIC_FIELD_MAX_LENGTH)


def _render_diagnostic(diagnostic: FailureDiagnostic) -> str:
    fields = [
        f"category={_structured_log_token(diagnostic.category)}",
        f"stage={_structured_log_token(diagnostic.stage)}",
        f"code={_structured_log_token(diagnostic.code)}",
    ]
    optional_fields = (
        ("provider_code", diagnostic.provider_code),
        ("exception", diagnostic.exception_type),
        ("location", diagnostic.location),
    )
    fields.extend(
        f"{name}={_structured_log_token(value)}"
        for name, value in optional_fields
        if value is not None
    )
    return " ".join(fields)


def _provider_diagnostic(error: ProviderError) -> FailureDiagnostic:
    return error.diagnostic or FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP, FailureStage.REQUEST, "provider_error"
    )


def _stream_diagnostic(error: StreamError) -> FailureDiagnostic:
    if error.diagnostic is not None:
        return error.diagnostic
    category = (
        FailureCategory.UPSTREAM_HTTP
        if error.status_code is not None
        else FailureCategory.PROVIDER_PROTOCOL
    )
    return FailureDiagnostic(category, FailureStage.STREAM, "stream_error")


def log_provider_failure(context: RequestLogContext, error: ProviderError) -> None:
    logger.warning(
        "%s %s %s provider request failed %s status=%s retryable=%s "
        "model=%s upstream=%s "
        "provider=%s effort=%s",
        _correlation(context),
        log_text(context.method),
        log_text(context.endpoint),
        _render_diagnostic(_provider_diagnostic(error)),
        error.status_code,
        retryable_status(error.status_code),
        _structured_log_token(context.original_model),
        _structured_log_token(context.upstream_model),
        _structured_log_token(error.provider),
        _structured_log_token(context.effort),
    )


def log_stream_failure(context: RequestLogContext, error: StreamError) -> None:
    provider = _structured_log_token(error.provider or context.provider)
    logger.warning(
        "%s %s %s provider stream failed %s error=%s status=%s retryable=%s "
        "model=%s upstream=%s provider=%s effort=%s",
        _correlation(context),
        log_text(context.method),
        log_text(context.endpoint),
        _render_diagnostic(_stream_diagnostic(error)),
        _structured_log_token(error.error_type),
        error.status_code,
        error.retryable,
        _structured_log_token(context.original_model),
        _structured_log_token(context.upstream_model),
        provider,
        _structured_log_token(context.effort),
    )


def log_unexpected_failure(
    context: RequestLogContext,
    error: Exception,
    *, stage: FailureStage = FailureStage.ROUTE,
) -> None:
    diagnostic = unexpected_failure_diagnostic(error, stage=stage)
    logger.error(
        "%s %s %s unexpected request failure %s "
        "model=%s upstream=%s provider=%s effort=%s",
        _correlation(context),
        log_text(context.method),
        log_text(context.endpoint),
        _render_diagnostic(diagnostic),
        _structured_log_token(context.original_model),
        _structured_log_token(context.upstream_model),
        _structured_log_token(context.provider),
        _structured_log_token(context.effort),
    )


def _fallback_context_identity(
    request, sessions: SessionRegistry
) -> tuple[SessionIdentity, AgentIdentity | None]:
    identity = client_identity_from_headers(request.headers)
    raw_session = (identity.session_id or "").strip()
    source_id = raw_session or uuid.uuid4().hex
    session = session_identity(
        sessions.public_id(source_id),
        request_scoped=not raw_session,
        is_new=False,
    )
    raw_agent = (identity.agent_id or "").strip()
    if not raw_agent:
        return session, None
    public_agent = sessions.public_agent_id(source_id, raw_agent)
    raw_parent = (identity.parent_agent_id or "").strip()
    public_parent = (
        sessions.public_agent_id(source_id, raw_parent)
        if raw_parent
        else None
    )
    return session, agent_identity(
        public_agent,
        public_parent,
        is_new=False,
    )


def log_http_failure(
    request, sessions: SessionRegistry, status_code: int
) -> None:
    context = getattr(request.state, REQUEST_LOG_CONTEXT, None)
    if context is not None:
        logger.warning(
            "%s %s %s HTTP request failed status=%s model=%s upstream=%s "
            "provider=%s effort=%s",
            _correlation(context),
            log_text(context.method),
            log_text(context.endpoint),
            status_code,
            _structured_log_token(context.original_model),
            _structured_log_token(context.upstream_model),
            _structured_log_token(context.provider),
            _structured_log_token(context.effort),
        )
        return

    session, agent = _fallback_context_identity(request, sessions)
    logger.warning(
        "%s %s %s HTTP request failed status=%s",
        _render_correlation(session, agent),
        log_text(request.method),
        log_text(request.url.path),
        status_code,
    )


def log_middleware_exception(
    request, sessions: SessionRegistry, error: Exception
) -> None:
    context = getattr(request.state, REQUEST_LOG_CONTEXT, None)
    if context is not None:
        log_unexpected_failure(context, error)
        return

    diagnostic = unexpected_failure_diagnostic(
        error, stage=FailureStage.ROUTE
    )
    session, agent = _fallback_context_identity(request, sessions)
    logger.error(
        "%s %s %s unexpected HTTP failure %s",
        _render_correlation(session, agent),
        log_text(request.method),
        log_text(request.url.path),
        _render_diagnostic(diagnostic),
    )


def request_logging_middleware(sessions: SessionRegistry):
    async def middleware(request, call_next):
        try:
            response = await call_next(request)
        except Exception as error:
            if not getattr(request.state, FAILURE_LOGGED, False):
                log_middleware_exception(request, sessions, error)
                setattr(request.state, FAILURE_LOGGED, True)
            raise

        if (
            response.status_code >= 300
            and not getattr(request.state, FAILURE_LOGGED, False)
        ):
            log_http_failure(request, sessions, response.status_code)
            setattr(request.state, FAILURE_LOGGED, True)
        return response

    return middleware


async def observe_stream(
    events: AsyncIterator[StreamEvent], context: RequestLogContext
) -> AsyncIterator[StreamEvent]:
    iterator = aiter(events)
    try:
        async for event in iterator:
            if isinstance(event, StreamError):
                log_stream_failure(context, event)
            yield event
    except ProviderError as error:
        log_provider_failure(context, error)
        raise
    except Exception as error:
        log_unexpected_failure(context, error, stage=FailureStage.STREAM)
        raise
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
