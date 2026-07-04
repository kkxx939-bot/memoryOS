from __future__ import annotations

import sqlite3

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.providers.embedding import HashingEmbeddingProvider
from memoryos.workers.embedding_worker import EmbeddingWorker
from memoryos.workers.semantic_worker import SemanticWorker


def _status(path, job_id: str) -> str:
    with sqlite3.connect(path) as conn:
        return str(conn.execute("SELECT status FROM queue_jobs WHERE job_id = ?", (job_id,)).fetchone()[0])


def test_semantic_worker_uses_sqlite_queue_and_acks_success(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    queue_path = tmp_path / "queue.sqlite3"
    queue = SQLiteQueueStore(queue_path)
    uri = "memoryos://user/u1/memories/m1"
    source.write_object(ContextObject(uri=uri, context_type=ContextType.MEMORY, title="M", owner_user_id="u1"), content="content")
    queue.enqueue(QueueJob(job_id="j1", queue_name="semantic", action="refresh", target_uri=uri))

    assert SemanticWorker(source, queue).process_pending()["processed"] == ["j1"]
    assert _status(queue_path, "j1") == "done"


def test_embedding_worker_uses_provider_metadata_and_sqlite_ack(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    queue_path = tmp_path / "queue.sqlite3"
    queue = SQLiteQueueStore(queue_path)
    vector = InMemoryVectorStore()
    uri = "memoryos://user/u1/memories/m1"
    source.write_object(ContextObject(uri=uri, context_type=ContextType.MEMORY, title="M", owner_user_id="u1"), content="hot room")
    queue.enqueue(QueueJob(job_id="j2", queue_name="embedding", action="embed", target_uri=uri))

    assert EmbeddingWorker(source, queue, vector, HashingEmbeddingProvider()).process_pending()["processed"] == ["j2"]
    assert _status(queue_path, "j2") == "done"
    metadata = vector.rows[uri][1]
    assert metadata["embedding_model"] == "hashing-v1"
    assert metadata["embedding_dimension"] == 16
    assert metadata["source_uri"] == uri
    assert metadata["schema_version"] == "vector_embedding_v1"


def test_worker_failure_does_not_ack_job(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    queue_path = tmp_path / "queue.sqlite3"
    queue = SQLiteQueueStore(queue_path)
    queue.enqueue(QueueJob(job_id="bad", queue_name="semantic", action="refresh", target_uri="memoryos://user/u1/missing"))

    assert SemanticWorker(source, queue).process_pending()["processed"] == []
    assert _status(queue_path, "bad") == "failed"
