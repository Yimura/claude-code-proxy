"""Application logging configuration and request correlation."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import hashlib
import logging
import os
import sys
import threading
import uuid

from .domain.models import StreamError, StreamEvent
from .providers.base import ProviderError
from .reasoning import ReasoningPolicy

SESSION_HEADER = "x-claude-code-session-id"
FAILURE_LOGGED = "failure_logged"
REQUEST_LOG_CONTEXT = "request_log_context"
SESSION_COLORS = (
    "\033[96m",
    "\033[94m",
    "\033[92m",
    "\033[95m",
    "\033[93m",
)
_RESET = "\033[0m"

logger = logging.getLogger(__name__)
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


class ColorizedFormatter(logging.Formatter):
    green = "\033[92m"
    reset = "\033[0m"
    bold = "\033[1m"

    def format(self, record):
        if record.levelno == logging.DEBUG and "MODEL MAPPING" in str(record.msg):
            return f"{self.bold}{self.green}{record.msg}{self.reset}"
        return super().format(record)


@dataclass(frozen=True)
class SessionIdentity:
    label: str
    rendered: str
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


def palette_index(identifier: str) -> int:
    digest = hashlib.sha256(identifier.encode()).digest()
    return int.from_bytes(digest[:8], "big") % len(SESSION_COLORS)


class SessionTracker:
    def __init__(self, stream=None, environ: Mapping[str, str] | None = None) -> None:
        self._stream = stream or sys.stderr
        self._environ = os.environ if environ is None else environ
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def observe(self, identifier: str | None) -> SessionIdentity:
        identifier = identifier.strip() if identifier else ""
        if not identifier:
            label = f"req-{uuid.uuid4().hex[:8]}"
            return SessionIdentity(label, f"[request {label}]", False)

        with self._lock:
            is_new = identifier not in self._seen
            self._seen.add(identifier)

        label = identifier[:8]
        rendered_label = label
        if self._color_enabled():
            color = SESSION_COLORS[palette_index(identifier)]
            rendered_label = f"{color}{label}{_RESET}"
        return SessionIdentity(label, f"[session {rendered_label}]", is_new)

    def _color_enabled(self) -> bool:
        return self._stream.isatty() and "NO_COLOR" not in self._environ


def effective_effort(policy: ReasoningPolicy) -> str:
    if policy.enabled is False:
        return "none"
    return policy.effort or "default"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addFilter(MessageFilter())
    session_logger.setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _request_fields(context: RequestLogContext) -> tuple[object, ...]:
    return (
        context.session.rendered,
        context.method,
        context.endpoint,
        context.original_model,
        context.upstream_model,
        context.provider,
        context.effort,
    )


def log_session_started(context: RequestLogContext) -> None:
    session_logger.info(
        "[NEW] %s %s %s %s → %s provider=%s effort=%s",
        *_request_fields(context),
    )


def log_provider_failure(context: RequestLogContext, status_code: int) -> None:
    logger.warning(
        "%s %s %s provider request failed status=%s model=%s upstream=%s "
        "provider=%s effort=%s",
        context.session.rendered,
        context.method,
        context.endpoint,
        status_code,
        context.original_model,
        context.upstream_model,
        context.provider,
        context.effort,
    )


def log_stream_failure(context: RequestLogContext) -> None:
    logger.warning(
        "%s %s %s provider stream failed model=%s upstream=%s provider=%s "
        "effort=%s",
        *_request_fields(context),
    )


def log_unexpected_failure(context: RequestLogContext, error_type: str) -> None:
    logger.error(
        "%s %s %s unexpected request failure error=%s model=%s upstream=%s "
        "provider=%s effort=%s",
        context.session.rendered,
        context.method,
        context.endpoint,
        error_type,
        context.original_model,
        context.upstream_model,
        context.provider,
        context.effort,
    )


def log_http_failure(request, session_tracker: SessionTracker, status_code: int) -> None:
    context = getattr(request.state, REQUEST_LOG_CONTEXT, None)
    if context is not None:
        logger.warning(
            "%s %s %s HTTP request failed status=%s model=%s upstream=%s "
            "provider=%s effort=%s",
            context.session.rendered,
            context.method,
            context.endpoint,
            status_code,
            context.original_model,
            context.upstream_model,
            context.provider,
            context.effort,
        )
        return

    session = session_tracker.observe(request.headers.get(SESSION_HEADER))
    logger.warning(
        "%s %s %s HTTP request failed status=%s",
        session.rendered,
        request.method,
        request.url.path,
        status_code,
    )


def log_middleware_exception(
    request, session_tracker: SessionTracker, error_type: str
) -> None:
    context = getattr(request.state, REQUEST_LOG_CONTEXT, None)
    if context is not None:
        log_unexpected_failure(context, error_type)
        return

    session = session_tracker.observe(request.headers.get(SESSION_HEADER))
    logger.error(
        "%s %s %s unexpected HTTP failure error=%s",
        session.rendered,
        request.method,
        request.url.path,
        error_type,
    )


def request_logging_middleware(session_tracker: SessionTracker):
    async def middleware(request, call_next):
        try:
            response = await call_next(request)
        except Exception as error:
            if not getattr(request.state, FAILURE_LOGGED, False):
                log_middleware_exception(
                    request, session_tracker, type(error).__name__
                )
                setattr(request.state, FAILURE_LOGGED, True)
            raise

        if (
            response.status_code >= 300
            and not getattr(request.state, FAILURE_LOGGED, False)
        ):
            log_http_failure(request, session_tracker, response.status_code)
            setattr(request.state, FAILURE_LOGGED, True)
        return response

    return middleware


async def observe_stream(
    events: AsyncIterator[StreamEvent], context: RequestLogContext
) -> AsyncIterator[StreamEvent]:
    try:
        async for event in events:
            if isinstance(event, StreamError):
                log_stream_failure(context)
            yield event
    except ProviderError as error:
        log_provider_failure(context, error.status_code)
        raise
    except Exception as error:
        log_unexpected_failure(context, type(error).__name__)
        raise
