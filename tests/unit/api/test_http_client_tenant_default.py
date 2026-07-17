from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from memoryos.api.sdk.http_client import HTTPMemoryOSClient


class _CaptureClient(HTTPMemoryOSClient):
    def __init__(self, *, tenant_id: str | None) -> None:
        super().__init__("http://memoryos.invalid", tenant_id=tenant_id)
        self.paths: list[str] = []

    def request(self, method: str, path: str, payload=None):  # noqa: ANN001, ANN201
        del method, payload
        self.paths.append(path)
        return {"results": []}


def test_http_pending_tenant_uses_explicit_then_client_and_never_forces_default() -> None:
    configured = _CaptureClient(tenant_id="tenant-a")
    configured.list_pending(user_id="u1")
    configured.list_pending(user_id="u1", tenant_id="tenant-b")
    implicit = _CaptureClient(tenant_id=None)
    implicit.list_pending(user_id="u1")

    configured_query = parse_qs(urlsplit(configured.paths[0]).query)
    explicit_query = parse_qs(urlsplit(configured.paths[1]).query)
    implicit_query = parse_qs(urlsplit(implicit.paths[0]).query)
    assert configured_query["tenant_id"] == ["tenant-a"]
    assert explicit_query["tenant_id"] == ["tenant-b"]
    assert "tenant_id" not in implicit_query
