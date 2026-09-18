"""Shared test isolation for dependency startup behavior."""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading

import pytest


# Tests importing LiteLLM types directly must never consult the public cost map.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


class CountingHTTPServer(ThreadingHTTPServer):
    """Loopback HTTP server that records every request it receives."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _CountingHandler)
        self.request_count = 0

    @property
    def url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/model-cost-map.json"


class _CountingHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, CountingHTTPServer)
        server.request_count += 1
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def counting_http_server() -> Iterator[CountingHTTPServer]:
    server = CountingHTTPServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
