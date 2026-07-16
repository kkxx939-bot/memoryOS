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
from memoryos.contextdb.store.source_store import SourceStore, is_canonical_memory_object
from memoryos.security.context_projection import ContextProjectionSanitizer


@dataclass(frozen=True)
class LayerRefreshResult:
    object_uri: str
    l0_uri: str
    l1_uri: str
    l2_uri: str


class LayerRefresher:
    def __init__(self, source_store: SourceStore, *, migration_gate=None) -> None:  # noqa: ANN001
        self.source_store = source_store
        self.migration_gate = migration_gate or getattr(source_store, "migration_gate", None)
        self.sanitizer = ContextProjectionSanitizer()

    def refresh(self, obj: ContextObject, content: str, bullets: list[str] | None = None) -> LayerRefreshResult:
        acquire = getattr(self.migration_gate, "acquire_projection_fence", None)
        release = getattr(self.migration_gate, "release_projection_fence", None)
        fence = acquire() if callable(acquire) else None
        try:
            return self._refresh_unfenced(obj, content, bullets=bullets)
        finally:
            if callable(release):
                release(fence)

    def _refresh_unfenced(
        self,
        obj: ContextObject,
        content: str,
        bullets: list[str] | None = None,
    ) -> LayerRefreshResult:
        if is_canonical_memory_object(obj):
            raise ValueError("canonical memory layers require the receipt-bound projector")
        base = obj.uri
        l0_uri = f"{base}/.abstract.md"
        l1_uri = f"{base}/.overview.md"
        l2_uri = f"{base}/content.md"
        try:
            l0_content = generate_l0_for_object(obj, content)
            l1_content = generate_l1_for_object(obj, content)
        except (AttributeError, KeyError, TypeError, ValueError):
            l0_content = l0_abstract(content)
            l1_content = l1_overview(obj.title, bullets or [content[:240]])
        if bullets:
            l1_content = l1_content or l1_overview(obj.title, bullets)
        metadata = dict(obj.metadata or {})
        safe = self.sanitizer.sanitize(
            title=obj.title,
            l0_text=l0_content,
            l1_text=l1_content,
            metadata=metadata,
            source_kind=str(metadata.get("source_kind") or obj.context_type.value),
        )
        # L0/L1 are rebuildable serving projections and must never retain raw
        # credentials, private paths, binary data, or unbounded output.  L2 is
        # immutable Source evidence and intentionally remains byte-for-byte
        # equivalent to the caller's content behind the SourceStore boundary.
        self.source_store.write_content(l0_uri, safe.l0_text)
        self.source_store.write_content(l1_uri, safe.l1_text)
        self.source_store.write_content(l2_uri, content)
        obj.layers = ContextLayers(l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri)
        self.source_store.write_object(obj)
        return LayerRefreshResult(object_uri=obj.uri, l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri)
