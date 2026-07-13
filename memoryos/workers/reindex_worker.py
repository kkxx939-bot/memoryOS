"""负责重建索引的后台任务。"""

from __future__ import annotations

from memoryos.contextdb.store.index_consistency import prepare_generic_index_rebuild
from memoryos.contextdb.store.source_store import IndexStore, SourceStore, is_canonical_memory_object
from memoryos.workers.readiness import require_source_store_ready


class ReindexWorker:
    def __init__(self, source_store: SourceStore, index_store: IndexStore) -> None:
        self.source_store = source_store
        self.index_store = index_store

    def rebuild(self) -> dict:
        require_source_store_ready(self.source_store)
        preparation = prepare_generic_index_rebuild(self.source_store, self.index_store)
        count = 0
        for obj in self.source_store.list_objects():
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
