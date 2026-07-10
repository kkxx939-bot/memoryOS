from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class HTTPMemoryOSClient:
    """Small dependency-free remote client; failures expose structured diagnostics."""

    def __init__(
        self,
        base_url: str,
        *,
        api_token: str | None = None,
        account_id: str | None = None,
        user_id: str | None = None,
        connect_timeout: float = 2.0,
        read_timeout: float = 10.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.account_id = account_id
        self.user_id = user_id
        self.connect_timeout = max(0.05, connect_timeout)
        self.read_timeout = max(0.05, read_timeout)
        self.timeout = max(connect_timeout, read_timeout)
        self.retries = max(0, min(retries, 3))

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload or {}).encode() if payload is not None else None
        request_id = _request_id()
        headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if self.account_id:
            headers["X-MemoryOS-Account"] = self.account_id
        if self.user_id:
            headers["X-MemoryOS-User"] = self.user_id
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    decoded = json.loads(response.read().decode())
                    return decoded if isinstance(decoded, dict) else {"data": decoded}
            except urllib.error.HTTPError as exc:
                retryable = exc.code >= 500
                if not retryable or attempt >= self.retries:
                    return {"error": {"code": "HTTP_ERROR", "message": f"HTTP {exc.code}", "retryable": retryable, "request_id": request_id, "operation": path}}
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt >= self.retries:
                    return {"error": {"code": "REMOTE_UNAVAILABLE", "message": str(exc)[:200], "retryable": True, "request_id": request_id, "operation": path}}
            time.sleep(0.05 * (attempt + 1))
        return {"error": {"code": "REMOTE_UNAVAILABLE", "message": "request failed", "retryable": True, "request_id": request_id, "operation": path}}

    def search_context(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.request("POST", "/v1/context/search", {"query": query, **kwargs}).get("results", []))

    def assemble_context(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("POST", "/v1/context/assemble", {"query": query, **kwargs})

    def commit_agent_session(self, **kwargs: Any) -> dict[str, Any]:
        archived = self.append_session_event(kwargs)
        if archived.get("error"):
            return archived
        session_key = str(archived.get("session_key") or kwargs.get("session_key") or "")
        if not session_key:
            return {
                "error": {
                    "code": "INVALID_SESSION_RESPONSE",
                    "message": "session event response omitted session_key",
                    "retryable": False,
                    "request_id": _request_id(),
                    "operation": "commit_agent_session",
                }
            }
        return self.finalize_session(session_key, async_commit=bool(kwargs.get("async_commit", True)))

    def append_session_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/sessions/events", payload)

    def checkpoint_session(self, session_key: str) -> dict[str, Any]:
        return self.request("POST", f"/v1/sessions/{session_key}/checkpoint", {})

    def finalize_session(self, session_key: str, *, async_commit: bool = True) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/v1/sessions/{session_key}/finalize",
            {"async_commit": async_commit},
        )

    def remember(self, **kwargs: Any) -> dict[str, Any]:
        return self.request("POST", "/v1/memories/remember", kwargs)

    def forget(self, **kwargs: Any) -> dict[str, Any]:
        return self.request("POST", "/v1/memories/forget", kwargs)

    def read(self, uri: str, *, layer: str = "L2") -> dict[str, Any]:
        query = urllib.parse.urlencode({"uri": uri, "layer": layer})
        return self.request("GET", f"/v1/context/read?{query}")

    def recall_trace(self, trace_id: str) -> dict[str, Any]:
        return self.request("GET", f"/v1/recall-traces/{urllib.parse.quote(trace_id, safe='')}")

    def archive_search(self, query: str, *, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        response = self.request(
            "POST",
            "/v1/archives/search",
            {"query": query, "user_id": user_id, "limit": limit},
        )
        return list(response.get("results", []))

    def archive_read(self, archive_uri: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"archive_uri": archive_uri})
        return self.request("GET", f"/v1/archives/read?{query}")

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")


def _request_id() -> str:
    import uuid
    return str(uuid.uuid4())
