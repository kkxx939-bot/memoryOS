from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from memoryos.api.sdk.http_client import HTTPMemoryOSClient


class _QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *args: Any) -> None:
        return


def _start(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_cross_origin_redirect_never_receives_authentication_headers() -> None:
    received: list[dict[str, str]] = []

    class Target(_QuietHandler):
        def do_GET(self) -> None:  # noqa: N802
            received.append({key: value for key, value in self.headers.items()})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"unexpected"}')

    target, target_thread = _start(Target)
    target_url = f"http://127.0.0.1:{target.server_port}/capture"

    class Redirect(_QuietHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", target_url)
            self.end_headers()

    redirect, redirect_thread = _start(Redirect)
    try:
        client = HTTPMemoryOSClient(
            f"http://127.0.0.1:{redirect.server_port}",
            api_token="secret-token",
            account_id="account-a",
            user_id="user-a",
            tenant_id="tenant-a",
            retries=0,
        )
        result = client.request("POST", "/redirect", {"value": 1})
        assert result["error"]["code"] == "HTTP_ERROR"
        assert result["error"]["status_code"] == 302
        assert received == []
    finally:
        _stop(redirect, redirect_thread)
        _stop(target, target_thread)


def test_same_origin_redirect_preserves_headers_and_compares_effective_port() -> None:
    received: list[dict[str, str]] = []

    class SameOrigin(_QuietHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            received.append({key: value for key, value in self.headers.items()})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

    server, thread = _start(SameOrigin)
    try:
        client = HTTPMemoryOSClient(
            f"http://127.0.0.1:{server.server_port}",
            api_token="same-origin-token",
            user_id="user-a",
            tenant_id="tenant-a",
            retries=0,
        )
        assert client.request("POST", "/redirect", {"value": 1}) == {"status": "ok"}
        assert received[0]["Authorization"] == "Bearer same-origin-token"
        assert received[0]["X-Memoryos-User"] == "user-a"
        assert received[0]["X-Memoryos-Tenant"] == "tenant-a"
    finally:
        _stop(server, thread)
