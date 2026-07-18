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
        if path.startswith("/v1/memories/history?"):
            return {
                "document_uri": "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "document_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "document_kind": "topic",
                "relative_path": "knowledge/topics/example.md",
                "revisions": [],
            }
        return {"results": []}


def test_http_memory_history_uses_exact_document_uri_without_identity_query_fields() -> None:
    client = _CaptureClient(tenant_id="tenant-a")
    document_uri = "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV"

    client.list_memory_history(document_uri)

    query = parse_qs(urlsplit(client.paths[0]).query)
    assert query == {"document_uri": [document_uri]}
    assert "tenant_id" not in query
    assert "user_id" not in query
