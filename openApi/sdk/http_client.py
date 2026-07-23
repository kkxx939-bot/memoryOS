"""访问远程 MemoryOS HTTP 服务的公开 Python SDK。

客户端负责序列化公开契约、限制重试、阻止跨源重定向并把远端错误
转换为 ``RemoteMemoryOSError``；服务端业务逻辑不在客户端重复实现。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from infrastructure.context.retrieval.query_plan import RetrievalOptions
from openApi.retrieval_contract import serialize_retrieval_options


@dataclass(frozen=True)
class RemoteMemoryOSError(RuntimeError):
    """远端 HTTP 错误的稳定 SDK 表示。"""

    code: str
    message: str
    retryable: bool = False
    request_id: str = ""
    operation: str = ""
    status_code: int | None = None

    def __str__(self) -> str:
        return self.message or self.code


class HTTPMemoryOSClient:
    """与 ``MemoryOSClient`` 对齐的远程 HTTP 调用门面。"""

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = 2.0,
        read_timeout: float = 10.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = max(0.05, connect_timeout)
        self.read_timeout = max(0.05, read_timeout)
        self.timeout = max(connect_timeout, read_timeout)
        self.retries = max(0, min(retries, 3))
        self._opener = urllib.request.build_opener(_SameOriginRedirectHandler())
        self._last_recall_trace_id: ContextVar[str] = ContextVar(
            f"memoryos_http_last_recall_trace_id_{id(self)}",
            default="",
        )

    @property
    def last_recall_trace_id(self) -> str:
        """返回当前请求上下文最近一次远程检索产生的轨迹 ID。"""

        return self._last_recall_trace_id.get()

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """发送一次有请求 ID 的 JSON 请求，并仅重试可恢复失败。"""

        body = json.dumps(payload or {}).encode() if payload is not None else None
        request_id = _request_id()
        headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        for attempt in range(self.retries + 1):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    decoded = json.loads(response.read().decode())
                    return decoded if isinstance(decoded, dict) else {"data": decoded}
            except urllib.error.HTTPError as exc:
                error = _remote_http_error(exc, request_id=request_id, operation=path)
                retryable = bool(error["retryable"])
                if not retryable or attempt >= self.retries:
                    return {"error": error}
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt >= self.retries:
                    return {
                        "error": {
                            "code": "REMOTE_UNAVAILABLE",
                            "message": str(exc)[:200],
                            "retryable": True,
                            "request_id": request_id,
                            "operation": path,
                        }
                    }
            time.sleep(0.05 * (attempt + 1))
        return {
            "error": {
                "code": "REMOTE_UNAVAILABLE",
                "message": "request failed",
                "retryable": True,
                "request_id": request_id,
                "operation": path,
            }
        }

    def search_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        serialized_options = serialize_retrieval_options(options)
        payload = {"query": query, **kwargs}
        if serialized_options is not None:
            payload["options"] = serialized_options
        response = self.request("POST", "/v1/context/search", payload)
        _raise_remote_error(response)
        self._last_recall_trace_id.set(str(response.get("trace_id") or ""))
        return list(response.get("results", []))

    def assemble_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        serialized_options = serialize_retrieval_options(options)
        payload = {"query": query, **kwargs}
        if serialized_options is not None:
            payload["options"] = serialized_options
        response = self.request("POST", "/v1/context/assemble", payload)
        _raise_remote_error(response)
        return response

    def commit_agent_session(self, **kwargs: Any) -> dict[str, Any]:
        """按 HTTP 会话协议先追加事件，再完成会话提交。"""

        archived = self.append_session_event(kwargs)
        session_key = str(archived.get("session_key") or kwargs.get("session_key") or "")
        if not session_key:
            raise RemoteMemoryOSError(
                code="INVALID_SESSION_RESPONSE",
                message="session event response omitted session_key",
                retryable=False,
                request_id=_request_id(),
                operation="commit_agent_session",
            )
        return self.finalize_session(session_key, async_commit=bool(kwargs.get("async_commit", True)))

    def append_session_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.request("POST", "/v1/sessions/events", payload)
        _raise_remote_error(response)
        return response

    def checkpoint_session(self, session_key: str) -> dict[str, Any]:
        response = self.request("POST", f"/v1/sessions/{session_key}/checkpoint", {})
        _raise_remote_error(response)
        return response

    def finalize_session(self, session_key: str, *, async_commit: bool = True) -> dict[str, Any]:
        response = self.request(
            "POST",
            f"/v1/sessions/{session_key}/finalize",
            {"async_commit": async_commit},
        )
        _raise_remote_error(response)
        return response

    def read(self, uri: str, *, layer: str = "L2") -> dict[str, Any]:
        query = urllib.parse.urlencode({"uri": uri, "layer": layer})
        response = self.request("GET", f"/v1/context/read?{query}")
        _raise_remote_error(response)
        return response

    def recall_trace(self, trace_id: str) -> dict[str, Any]:
        response = self.request("GET", f"/v1/recall-traces/{urllib.parse.quote(trace_id, safe='')}")
        _raise_remote_error(response)
        return response

    def archive_search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 20,
        project_id: str = "",
    ) -> list[dict[str, Any]]:
        response = self.request(
            "POST",
            "/v1/archives/search",
            {
                "query": query,
                "user_id": user_id,
                "limit": limit,
                **({"project_id": project_id} if project_id else {}),
            },
        )
        _raise_remote_error(response)
        return list(response.get("results", []))

    def archive_read(self, archive_uri: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"archive_uri": archive_uri})
        response = self.request("GET", f"/v1/archives/read?{query}")
        _raise_remote_error(response)
        return response

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")


def _request_id() -> str:
    import uuid

    return str(uuid.uuid4())


def _remote_http_error(
    exc: urllib.error.HTTPError,
    *,
    request_id: str,
    operation: str,
) -> dict[str, Any]:
    """在不信任响应正文的前提下，保留有界的结构化服务端错误。"""

    remote: dict[str, Any] = {}
    try:
        raw = exc.read(65_537)
        if len(raw) <= 65_536:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                remote = dict(payload["error"])
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        remote = {}
    code = str(remote.get("code") or "HTTP_ERROR")[:64]
    message = str(remote.get("message") or f"HTTP {exc.code}")[:500]
    remote_request_id = str(remote.get("request_id") or request_id)[:128]
    remote_operation = str(remote.get("operation") or operation)[:200]
    raw_retryable = remote.get("retryable")
    retryable = raw_retryable if isinstance(raw_retryable, bool) else exc.code >= 500
    return {
        "code": code,
        "message": message,
        "retryable": retryable,
        "request_id": remote_request_id,
        "operation": remote_operation,
        "status_code": int(exc.code),
    }


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """拒绝跨源跳转，防止本地用户标识和请求数据被转发到其他服务。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        if _origin(req.full_url) != _origin(newurl):
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                "cross-origin redirect blocked",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.casefold()
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parsed.hostname or "").casefold(), port


def _raise_remote_error(response: dict[str, Any]) -> None:
    raw = response.get("error")
    if not isinstance(raw, dict):
        return
    status_code = raw.get("status_code")
    raise RemoteMemoryOSError(
        code=str(raw.get("code") or "REMOTE_ERROR"),
        message=str(raw.get("message") or "remote MemoryOS request failed"),
        retryable=bool(raw.get("retryable", False)),
        request_id=str(raw.get("request_id") or ""),
        operation=str(raw.get("operation") or ""),
        status_code=int(status_code) if isinstance(status_code, int) else None,
    )


def _without_request_id(response: dict[str, Any]) -> dict[str, Any]:
    """ASGI 请求 ID 是传输元数据，不属于命令结果数据。"""

    payload = dict(response)
    payload.pop("request_id", None)
    return payload
