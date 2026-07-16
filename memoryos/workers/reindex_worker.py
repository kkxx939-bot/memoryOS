"""负责重建索引的后台任务。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from memoryos.contextdb.store.index_consistency import (
    _checkpoint_projection_fence,
    _prepare_generic_index_rebuild_unfenced,
)
from memoryos.contextdb.store.source_store import IndexStore, SourceStore, is_canonical_memory_object
from memoryos.workers.readiness import require_source_store_ready


class ReindexWorker:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        *,
        migration_gate: Any | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.migration_gate = migration_gate or getattr(source_store, "migration_gate", None)

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
        )
        count = 0
        for offset, obj in enumerate(self.source_store.list_objects()):
            if offset % 256 == 0:
                _checkpoint_projection_fence(fence)
            if is_canonical_memory_object(obj):
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
