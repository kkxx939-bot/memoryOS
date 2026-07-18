"""接口层里的HTTP客户端。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from memoryos.api.memory_contract import validate_memory_request, validate_memory_response
from memoryos.api.retrieval_contract import serialize_retrieval_options
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions


@dataclass(frozen=True)
class RemoteMemoryOSError(RuntimeError):
    code: str
    message: str
    retryable: bool = False
    request_id: str = ""
    operation: str = ""
    status_code: int | None = None

    def __str__(self) -> str:
        return self.message or self.code


class HTTPMemoryOSClient:
    """负责 HTTPMemoryOSClient 这部分逻辑。"""

    def __init__(
        self,
        base_url: str,
        *,
        api_token: str | None = None,
        account_id: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        connect_timeout: float = 2.0,
        read_timeout: float = 10.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.account_id = account_id
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.connect_timeout = max(0.05, connect_timeout)
        self.read_timeout = max(0.05, read_timeout)
        self.timeout = max(connect_timeout, read_timeout)
        self.retries = max(0, min(retries, 3))
        self._opener = urllib.request.build_opener(_SameOriginRedirectHandler())

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
        if self.tenant_id:
            headers["X-MemoryOS-Tenant"] = self.tenant_id
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

    def remember(
        self,
        content: str,
        occurred_at: str | None = None,
        target_hint: str | None = None,
        expected_document_digest: str | None = None,
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "remember",
            {
                "content": content,
                "occurred_at": occurred_at,
                "target_hint": target_hint,
                "expected_document_digest": expected_document_digest,
            },
        )
        response = self.request("POST", "/v1/memories/remember", payload)
        _raise_remote_error(response)
        return validate_memory_response("remember", _without_request_id(response))

    def adopt_memory_document(
        self,
        relative_path: str,
        expected_raw_sha256: str,
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "adopt",
            {
                "relative_path": relative_path,
                "expected_raw_sha256": expected_raw_sha256,
            },
        )
        response = self.request("POST", "/v1/memories/adopt", payload)
        _raise_remote_error(response)
        return validate_memory_response("adopt", _without_request_id(response))

    def edit_memory_document(
        self,
        document_uri: str,
        edit: str,
        expected_digest: str,
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "edit",
            {"document_uri": document_uri, "edit": edit, "expected_digest": expected_digest},
        )
        response = self.request("POST", "/v1/memories/edit", payload)
        _raise_remote_error(response)
        return validate_memory_response("edit", _without_request_id(response))

    def rename_memory_document(
        self,
        document_uri: str,
        new_relative_path: str,
        expected_digest: str,
        edit: str | None = None,
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "rename",
            {
                "document_uri": document_uri,
                "new_relative_path": new_relative_path,
                "expected_digest": expected_digest,
                "edit": edit,
            },
        )
        response = self.request("POST", "/v1/memories/rename", payload)
        _raise_remote_error(response)
        return validate_memory_response("rename", _without_request_id(response))

    def merge_memory_documents(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "merge",
            {
                "target_document_uri": target_document_uri,
                "merged_edit": merged_edit,
                "expected_target_digest": expected_target_digest,
                "source_documents": source_documents,
            },
        )
        response = self.request("POST", "/v1/memories/merge", payload)
        _raise_remote_error(response)
        return validate_memory_response("merge", _without_request_id(response))

    def propose_memory_consolidation(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "merge_propose",
            {
                "target_document_uri": target_document_uri,
                "merged_edit": merged_edit,
                "expected_target_digest": expected_target_digest,
                "source_documents": source_documents,
            },
        )
        response = self.request("POST", "/v1/memories/merge/propose", payload)
        _raise_remote_error(response)
        return validate_memory_response("merge_propose", _without_request_id(response))

    def resume_memory_consolidation(self, saga_id: str) -> dict[str, Any]:
        payload = validate_memory_request("merge_resume", {"saga_id": saga_id})
        response = self.request("POST", "/v1/memories/merge/resume", payload)
        _raise_remote_error(response)
        return validate_memory_response("merge_resume", _without_request_id(response))

    def forget(
        self,
        document_uri: str,
        section_anchor: str | None = None,
        mode: str = "SOFT_FORGET",
        expected_digest: str | None = None,
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "forget",
            {
                "document_uri": document_uri,
                "section_anchor": section_anchor,
                "mode": mode,
                "expected_digest": expected_digest,
            },
        )
        response = self.request("POST", "/v1/memories/forget", payload)
        _raise_remote_error(response)
        return validate_memory_response("forget", _without_request_id(response))

    def list_memory_history(self, document_uri: str) -> dict[str, Any]:
        payload = validate_memory_request("history", {"document_uri": document_uri})
        query = urllib.parse.urlencode(payload)
        response = self.request("GET", f"/v1/memories/history?{query}")
        _raise_remote_error(response)
        return validate_memory_response("history", _without_request_id(response))

    def restore_memory_revision(
        self,
        document_uri: str,
        revision: int,
        expected_digest: str,
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "restore",
            {
                "document_uri": document_uri,
                "revision": revision,
                "expected_digest": expected_digest,
            },
        )
        response = self.request("POST", "/v1/memories/restore", payload)
        _raise_remote_error(response)
        return validate_memory_response("restore", _without_request_id(response))

    def review_memory_edit(
        self,
        proposal_id: str,
        decision: str,
        corrected_edit: str | None = None,
    ) -> dict[str, Any]:
        payload = validate_memory_request(
            "review",
            {"proposal_id": proposal_id, "decision": decision, "corrected_edit": corrected_edit},
        )
        response = self.request("POST", "/v1/memories/review", payload)
        _raise_remote_error(response)
        return validate_memory_response("review", _without_request_id(response))

    def preview_memory_edit(self, proposal_id: str) -> dict[str, Any]:
        payload = validate_memory_request("review_preview", {"proposal_id": proposal_id})
        response = self.request("POST", "/v1/memories/review/preview", payload)
        _raise_remote_error(response)
        return validate_memory_response("review_preview", _without_request_id(response))

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
        tenant_id: str | None = None,
        project_id: str = "",
    ) -> list[dict[str, Any]]:
        response = self.request(
            "POST",
            "/v1/archives/search",
            {
                "query": query,
                "user_id": user_id,
                "limit": limit,
                **({"tenant_id": tenant_id} if tenant_id is not None else {}),
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
    """Preserve a bounded structured server error without trusting its body."""

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
    """Allow redirects only when scheme, host, and effective port are unchanged."""

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
    """The ASGI request id is transport metadata, not command result data."""

    payload = dict(response)
    payload.pop("request_id", None)
    return payload
