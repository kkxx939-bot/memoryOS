"""精确上下文读取的公开出口清洗测试。"""

from __future__ import annotations

from infrastructure.context.exact_reader import ContextExactReader
from infrastructure.store.model.context.context_layer import ContextLayers
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType


class _SourceStore:
    def __init__(self, obj: ContextObject, content: str) -> None:
        self.obj = obj
        self.content = content

    def read_content(self, uri: str) -> str:
        assert uri == self.obj.layers.l2_uri
        return self.content


class _ContextDB:
    def __init__(self, obj: ContextObject) -> None:
        self.obj = obj

    def read_object(self, uri: str) -> ContextObject:
        assert uri == self.obj.uri
        return self.obj


def test_exact_read_sanitizes_content_and_object_metadata() -> None:
    uri = "memoryos://user/u1/resources/report"
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.RESOURCE,
        title="private report",
        owner_user_id="u1",
        layers=ContextLayers(l2_uri=f"{uri}/layers/l2"),
        metadata={
            "source_kind": "resource",
            "path": "/Users/u1/private/report.txt",
            "note": "password=top-secret",
        },
    )
    reader = ContextExactReader(
        source_store=_SourceStore(obj, "token=abc /Users/u1/private/report.txt"),  # type: ignore[arg-type]
        index_store=None,
        context_reader=_ContextDB(obj),
        document_overlay=None,
        require_exact_read_scope=lambda _uri, _obj, _caller: None,
    )

    result = reader.read(uri, layer="L2", tenant_id="default", caller=None)

    serialized = str(result)
    assert "top-secret" not in serialized
    assert "token=abc" not in serialized
    assert "/Users/u1" not in serialized
    assert result["object"]["metadata"]["projection_sanitized"] is True
