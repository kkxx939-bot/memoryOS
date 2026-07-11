"""上下文数据库里的分层刷新器。"""

from __future__ import annotations

from dataclasses import dataclass

from memoryos.contextdb.layers.layer_generator import (
    generate_l0_for_object,
    generate_l1_for_object,
    l0_abstract,
    l1_overview,
)
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import SourceStore


@dataclass(frozen=True)
class LayerRefreshResult:
    object_uri: str
    l0_uri: str
    l1_uri: str
    l2_uri: str


class LayerRefresher:
    def __init__(self, source_store: SourceStore) -> None:
        self.source_store = source_store

    def refresh(self, obj: ContextObject, content: str, bullets: list[str] | None = None) -> LayerRefreshResult:
        base = obj.uri
        l0_uri = f"{base}/.abstract.md"
        l1_uri = f"{base}/.overview.md"
        l2_uri = f"{base}/content.md"
        try:
            l0_content = generate_l0_for_object(obj, content)
            l1_content = generate_l1_for_object(obj, content)
        except Exception:
            l0_content = l0_abstract(content)
            l1_content = l1_overview(obj.title, bullets or [content[:240]])
        if bullets:
            l1_content = l1_content or l1_overview(obj.title, bullets)
        self.source_store.write_content(l0_uri, l0_content)
        self.source_store.write_content(l1_uri, l1_content)
        self.source_store.write_content(l2_uri, content)
        obj.layers = ContextLayers(l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri)
        self.source_store.write_object(obj)
        return LayerRefreshResult(object_uri=obj.uri, l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri)
