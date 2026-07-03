from __future__ import annotations

from dataclasses import dataclass

from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore


@dataclass(frozen=True)
class IndexConsistencyResult:
    source_count: int
    index_count: int
    missing_in_index: list[str]

    @property
    def consistent(self) -> bool:
        return not self.missing_in_index


class IndexConsistencyService:
    def __init__(self, source_store: FileSystemSourceStore, index_store: InMemoryIndexStore) -> None:
        self.source_store = source_store
        self.index_store = index_store

    def verify(self) -> IndexConsistencyResult:
        objects = self.source_store.list_objects()
        missing = [obj.uri for obj in objects if obj.uri not in self.index_store.rows]
        return IndexConsistencyResult(
            source_count=len(objects),
            index_count=len(self.index_store.rows),
            missing_in_index=missing,
        )

    def rebuild(self) -> IndexConsistencyResult:
        self.index_store.rows.clear()
        for obj in self.source_store.list_objects():
            try:
                content = self.source_store.read_content(obj.uri)
            except FileNotFoundError:
                content = ""
            self.index_store.upsert_index(obj, content=content)
        return self.verify()
