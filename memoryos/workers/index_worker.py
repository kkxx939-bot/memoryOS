from __future__ import annotations

from memoryos.adapters.events.jsonl_index_jobs import JsonlIndexJobRepository
from memoryos.ports.indexes.vector_index import VectorIndex, VectorRecord
from memoryos.ports.repositories.memory_repository import MemoryRepository


class IndexWorker:
    def __init__(self, store: MemoryRepository, vector_index: VectorIndex | None = None) -> None:
        self.store = store
        self.jobs = JsonlIndexJobRepository(store.root)
        self.vector_index = vector_index

    def process_pending(self, user_id: str | None = None, limit: int = 50) -> dict:
        rows = self.jobs.pending(user_id=user_id, limit=limit)
        processed = []
        skipped = []
        for row in rows:
            if self.vector_index is None:
                skipped.append({"job_id": row["job_id"], "reason": "no vector index configured"})
                continue
            try:
                if row.get("operation") == "delete":
                    self.vector_index.delete(namespace=str(row.get("namespace", "memory")), id=str(row.get("source_id", "")))
                    processed.append(self.jobs.mark(str(row["job_id"]), "deleted"))
                    continue
                chunk = row.get("metadata", {}).get("chunk", {})
                embedding = row.get("metadata", {}).get("embedding", {})
                vector = embedding.get("vector", [])
                self.vector_index.upsert(
                    VectorRecord(
                        namespace=str(row.get("namespace", "memory")),
                        id=str(chunk.get("chunk_id") or row.get("source_id")),
                        vector=[float(value) for value in vector],
                        text=str(chunk.get("text", "")),
                        metadata=chunk,
                        content_hash=str(row.get("content_hash", "")),
                        provider=str(row.get("embedding_provider", "")),
                        model=str(row.get("embedding_model", "")),
                        dimension=int(row.get("embedding_dimension", 0)),
                    )
                )
                processed.append(self.jobs.mark(str(row["job_id"]), "indexed"))
            except Exception as exc:
                processed.append(self.jobs.mark(str(row["job_id"]), "failed", {"last_error": str(exc)}))
        return {"processed": len(processed), "skipped": len(skipped), "results": processed, "skipped_jobs": skipped}
