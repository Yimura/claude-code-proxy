"""Console formatter for human-readable severity styling."""

from collections.abc import Mapping
from copy import copy
import logging
import os

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_RESET = "\033[0m"
_LEVEL_STYLES = {
    logging.DEBUG: "\033[2m",
    logging.WARNING: "\033[1;33m",
    logging.ERROR: "\033[1;31m",
    logging.CRITICAL: "\033[1;31m",
}


class SeverityFormatter(logging.Formatter):
    """Style only the textual severity token in console log records."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        super().__init__(LOG_FORMAT)
        environment = os.environ if environ is None else environ
        self._color_enabled = "NO_COLOR" not in environment

    def format(self, record: logging.LogRecord) -> str:
        formatted_record = copy(record)
        style = _LEVEL_STYLES.get(formatted_record.levelno)
        if self._color_enabled and style:
            formatted_record.levelname = (
                f"{style}{formatted_record.levelname}{_RESET}"
            )
        return super().format(formatted_record)
