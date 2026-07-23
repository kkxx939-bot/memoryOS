"""上下文数据库里的分层刷新器。"""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.context.layers.generator import (
    generate_l0_for_object,
    generate_l1_for_object,
    l0_abstract,
    l1_overview,
)
from infrastructure.store.contracts.domain import ContextDomainClassifier, NoContextDomainClassifier
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_layer import ContextLayers
from infrastructure.store.model.context.context_object import ContextObject
from sanitization.context_projection import ContextProjectionSanitizer


@dataclass(frozen=True)
class LayerRefreshResult:
    object_uri: str
    l0_uri: str
    l1_uri: str
    l2_uri: str


class LayerRefresher:
    def __init__(
        self,
        source_store: SourceStore,
        *,
        domain_classifier: ContextDomainClassifier | None = None,
    ) -> None:
        self.source_store = source_store
        self.domain_classifier = (
            domain_classifier or getattr(source_store, "domain_classifier", None) or NoContextDomainClassifier()
        )
        self.sanitizer = ContextProjectionSanitizer()

    def refresh(self, obj: ContextObject, content: str, bullets: list[str] | None = None) -> LayerRefreshResult:
        return self._refresh_unfenced(obj, content, bullets=bullets)

    def _refresh_unfenced(
        self,
        obj: ContextObject,
        content: str,
        bullets: list[str] | None = None,
    ) -> LayerRefreshResult:
        if self.domain_classifier.owns_object(obj):
            raise ValueError("domain-owned layers require their authoritative projector")
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
        # L0/L1 是可重建的 Serving 投影，不能保留原始凭证、私有路径、二进制或
        # 无界输出。L2 是 SourceStore 边界后的不可变证据，必须与调用者内容逐字节一致。
        self.source_store.write_content(l0_uri, safe.l0_text)
        self.source_store.write_content(l1_uri, safe.l1_text)
        self.source_store.write_content(l2_uri, content)
        obj.layers = ContextLayers(l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri)
        self.source_store.write_object(obj)
        return LayerRefreshResult(object_uri=obj.uri, l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri)
