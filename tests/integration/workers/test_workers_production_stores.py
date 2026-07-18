from __future__ import annotations

import json
import sqlite3

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore, vector_row_id
from memoryos.providers.embedding import HashingEmbeddingProvider
from memoryos.security.context_projection import (
    ContextProjectionSanitizationError,
    ContextProjectionSanitizer,
)
from memoryos.workers.embedding_worker import EmbeddingWorker
from memoryos.workers.semantic_worker import SemanticWorker


class _CapturingEmbeddingProvider:
    model_name = "capture-v1"
    dimension = 2

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [1.0, 0.0]


class _FailingProjectionSanitizer:
    def sanitize(self, **_kwargs):  # noqa: ANN003, ANN201
        raise ContextProjectionSanitizationError("forced projection failure")


def _status(path, job_id: str) -> str:
    with sqlite3.connect(path) as conn:
        return str(conn.execute("SELECT status FROM queue_jobs WHERE job_id = ?", (job_id,)).fetchone()[0])


def test_semantic_worker_uses_sqlite_queue_and_acks_success(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    queue_path = tmp_path / "queue.sqlite3"
    queue = SQLiteQueueStore(queue_path)
    uri = "memoryos://user/u1/resources/m1"
    source.write_object(ContextObject(uri=uri, context_type=ContextType.RESOURCE, title="M", owner_user_id="u1"), content="content")
    queue.enqueue(QueueJob(job_id="j1", queue_name="semantic", action="refresh", target_uri=uri))

    assert SemanticWorker(source, queue).process_pending()["processed"] == ["j1"]
    assert _status(queue_path, "j1") == "done"


def test_embedding_worker_uses_provider_metadata_and_sqlite_ack(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    queue_path = tmp_path / "queue.sqlite3"
    queue = SQLiteQueueStore(queue_path)
    vector = InMemoryVectorStore()
    uri = "memoryos://user/u1/resources/m1"
    source.write_object(ContextObject(uri=uri, context_type=ContextType.RESOURCE, title="M", owner_user_id="u1"), content="hot room")
    queue.enqueue(QueueJob(job_id="j2", queue_name="embedding", action="embed", target_uri=uri))

    assert EmbeddingWorker(source, queue, vector, HashingEmbeddingProvider()).process_pending()["processed"] == ["j2"]
    assert _status(queue_path, "j2") == "done"
    metadata = vector.rows[vector_row_id("default", uri)][1]
    assert metadata["embedding_model"] == "hashing-v1"
    assert metadata["embedding_dimension"] == 16
    assert metadata["source_uri"] == uri
    assert metadata["schema_version"] == "vector_embedding_v1"


def test_semantic_and_embedding_workers_sanitize_derived_layers_and_vector_input(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    queue = SQLiteQueueStore(tmp_path / "queue.sqlite3")
    vector = InMemoryVectorStore()
    provider = _CapturingEmbeddingProvider()
    uri = "memoryos://user/u1/resources/sensitive-derived-projection"
    raw = (
        "Authorization: Bearer abcdefghijklmnop\n"
        "password=hunter2\n"
        "/Users/u1/Desktop/secret-report.txt"
    )
    source.write_object(
        ContextObject(
            uri=uri,
            context_type=ContextType.RESOURCE,
            title="Sensitive projection",
            owner_user_id="u1",
            metadata={"source_kind": "tool_result", "authorization": "Bearer metadata-secret"},
        ),
        content=raw,
    )

    queue.enqueue(QueueJob(job_id="safe-vector", queue_name="embedding", action="embed", target_uri=uri))
    assert EmbeddingWorker(source, queue, vector, provider).process_pending()["processed"] == ["safe-vector"]
    assert len(provider.inputs) == 1
    ContextProjectionSanitizer().assert_safe(provider.inputs[0])
    assert "secret-report.txt" in provider.inputs[0]

    vector_metadata = vector.rows[vector_row_id("default", uri)][1]
    ContextProjectionSanitizer().assert_safe(vector_metadata)
    assert vector_metadata["projection_sanitized"] is True
    assert vector_metadata["projection_redacted"] is True

    queue.enqueue(QueueJob(job_id="safe-layers", queue_name="semantic", action="refresh", target_uri=uri))
    assert SemanticWorker(source, queue).process_pending()["processed"] == ["safe-layers"]
    projected = source.read_object(uri)
    assert projected.layers.l0_uri and projected.layers.l1_uri and projected.layers.l2_uri
    l0 = source.read_content(projected.layers.l0_uri)
    l1 = source.read_content(projected.layers.l1_uri)
    l2 = source.read_content(projected.layers.l2_uri)
    ContextProjectionSanitizer().assert_safe({"l0": l0, "l1": l1})
    assert l2 == raw

    derived = json.dumps(
        {
            "provider_input": provider.inputs,
            "vector_metadata": vector_metadata,
            "l0": l0,
            "l1": l1,
        },
        ensure_ascii=False,
    )
    for forbidden in ("abcdefghijklmnop", "hunter2", "metadata-secret", "/Users/u1"):
        assert forbidden not in derived


def test_embedding_worker_fails_closed_before_provider_and_vector_when_sanitization_fails(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    queue = SQLiteQueueStore(tmp_path / "queue.sqlite3")
    vector = InMemoryVectorStore()
    provider = _CapturingEmbeddingProvider()
    uri = "memoryos://user/u1/resources/fail-closed-vector"
    source.write_object(
        ContextObject(uri=uri, context_type=ContextType.RESOURCE, title="fail closed", owner_user_id="u1"),
        content="must never reach the provider",
    )
    queue.enqueue(QueueJob(job_id="unsafe", queue_name="embedding", action="embed", target_uri=uri))
    worker = EmbeddingWorker(source, queue, vector, provider)
    worker.sanitizer = _FailingProjectionSanitizer()  # type: ignore[assignment]

    result = worker.process_pending()

    assert result["processed"] == []
    assert result["dead_letter"] == ["unsafe"]
    assert provider.inputs == []
    assert not vector.rows


def test_worker_failure_does_not_ack_job(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    queue_path = tmp_path / "queue.sqlite3"
    queue = SQLiteQueueStore(queue_path)
    queue.enqueue(QueueJob(job_id="bad", queue_name="semantic", action="refresh", target_uri="memoryos://user/u1/missing"))

    assert SemanticWorker(source, queue).process_pending()["processed"] == []
    assert _status(queue_path, "bad") == "pending"
