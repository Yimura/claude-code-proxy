"""Application logging configuration and request summaries."""

import logging
import sys
import time


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


class Colors:
    cyan = "\033[96m"
    blue = "\033[94m"
    green = "\033[92m"
    red = "\033[91m"
    magenta = "\033[95m"
    reset = "\033[0m"
    bold = "\033[1m"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addFilter(MessageFilter())
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def log_requests(request, call_next):
    started = time.time()
    response = await call_next(request)
    logging.getLogger(__name__).debug(
        "%s %s completed with %s in %.2fs",
        request.method,
        request.url.path,
        response.status_code,
        time.time() - started,
    )
    return response


def log_request_summary(
    method,
    path,
    claude_model,
    upstream_model,
    num_messages,
    num_tools,
    status_code,
):
    endpoint = path.split("?", 1)[0]
    upstream_name = upstream_model.rsplit("/", 1)[-1]
    status = (
        f"{Colors.green}✓ {status_code} OK{Colors.reset}"
        if status_code == 200
        else f"{Colors.red}✗ {status_code}{Colors.reset}"
    )
    print(f"{Colors.bold}{method} {endpoint}{Colors.reset} {status}")
    print(
        f"{Colors.cyan}{claude_model}{Colors.reset} → "
        f"{Colors.green}{upstream_name}{Colors.reset} "
        f"{Colors.magenta}{num_tools} tools{Colors.reset} "
        f"{Colors.blue}{num_messages} messages{Colors.reset}"
    )
    sys.stdout.flush()
