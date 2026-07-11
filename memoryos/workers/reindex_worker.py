"""负责重建索引的后台任务。"""

from __future__ import annotations

from memoryos.contextdb.store.source_store import IndexStore, SourceStore


class ReindexWorker:
    def __init__(self, source_store: SourceStore, index_store: IndexStore) -> None:
        self.source_store = source_store
        self.index_store = index_store

    def rebuild(self) -> dict:
        if hasattr(self.index_store, "clear"):
            self.index_store.clear()
        count = 0
        for obj in self.source_store.list_objects():
            try:
                content = self.source_store.read_content(obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = obj.title
            self.index_store.upsert_index(obj, content=content)
            count += 1
        return {"status": "rebuilt", "indexed": count}
