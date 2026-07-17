"""负责重建索引的后台任务。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from memoryos.contextdb.extensions import (
    ContextDomainClassifier,
    ContextIndexPolicy,
    NoContextIndexPolicy,
    NoDomainOverlay,
)
from memoryos.contextdb.store.index_consistency import (
    _checkpoint_projection_fence,
    _prepare_generic_index_rebuild_unfenced,
)
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.readiness import require_source_store_ready


class ReindexWorker:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        *,
        migration_gate: Any | None = None,
        domain_classifier: ContextDomainClassifier | None = None,
        index_policy: ContextIndexPolicy | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.migration_gate = migration_gate or getattr(source_store, "migration_gate", None)
        self.domain_classifier = (
            domain_classifier
            or getattr(source_store, "domain_classifier", None)
            or NoDomainOverlay()
        )
        self.index_policy = index_policy or NoContextIndexPolicy()

    @contextmanager
    def _projection_fence(self, *, projection_fence_held: bool) -> Iterator[Any | None]:
        if projection_fence_held:
            yield None
            return
        acquire = getattr(self.migration_gate, "acquire_projection_fence", None)
        release = getattr(self.migration_gate, "release_projection_fence", None)
        fence = acquire() if callable(acquire) else None
        try:
            yield fence
        finally:
            if callable(release):
                release(fence)

    def rebuild(self, *, projection_fence_held: bool = False) -> dict:
        with self._projection_fence(projection_fence_held=projection_fence_held) as fence:
            return self._rebuild_unfenced(fence=fence)

    def _rebuild_unfenced(self, *, fence: Any | None) -> dict:
        require_source_store_ready(self.source_store)
        preparation = _prepare_generic_index_rebuild_unfenced(
            self.source_store,
            self.index_store,
            fence=fence,
            index_policy=self.index_policy,
        )
        count = 0
        for offset, obj in enumerate(self.source_store.list_objects()):
            if offset % 256 == 0:
                _checkpoint_projection_fence(fence)
            if self.domain_classifier.owns_object(obj):
                continue
            try:
                content = self.source_store.read_content(obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = obj.title
            self.index_store.upsert_index(obj, content=content)
            count += 1
        return {
            "status": "rebuilt",
            "indexed": count,
            "canonical_preserved": preparation["canonical_preserved"],
        }
