"""负责重建索引的后台任务。"""

from __future__ import annotations

from memoryos.contextdb.extensions import (
    ContextDomainClassifier,
    ContextIndexPolicy,
    NoContextIndexPolicy,
    NoDomainOverlay,
)
from memoryos.contextdb.store.index_consistency import prepare_generic_index_rebuild
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.readiness import require_source_store_ready


class ReindexWorker:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        *,
        domain_classifier: ContextDomainClassifier | None = None,
        index_policy: ContextIndexPolicy | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.domain_classifier = (
            domain_classifier
            or getattr(source_store, "domain_classifier", None)
            or NoDomainOverlay()
        )
        self.index_policy = index_policy or NoContextIndexPolicy()

    def rebuild(self) -> dict:
        require_source_store_ready(self.source_store)
        tenant_id = str(getattr(self.source_store, "tenant_id", "default") or "default")
        preparation = prepare_generic_index_rebuild(
            self.source_store,
            self.index_store,
            tenant_id=tenant_id,
            index_policy=self.index_policy,
        )
        count = 0
        for obj in self.source_store.list_objects():
            if self.domain_classifier.owns_object(obj):
                continue
            try:
                content = self.source_store.read_content(obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = obj.title
            self.index_store.upsert_index(
                obj,
                content=content,
                tenant_id=tenant_id,
            )
            count += 1
        return {
            "status": "rebuilt",
            "indexed": count,
            "derived_preserved": preparation["derived_preserved"],
        }
