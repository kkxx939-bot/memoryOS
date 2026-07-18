from __future__ import annotations

from pathlib import Path

import pytest

from memoryos.adapters.persistence.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from memoryos.adapters.persistence.in_memory.queue_store import InMemoryQueueStore
from memoryos.contextdb.store.queue_store import QueueJob
from memoryos.memory.documents import (
    ABSENT,
    DocumentEditKind,
    DocumentEditPlan,
    DocumentIntentStatus,
    MemoryDocumentCommitter,
    MemoryDocumentControlStore,
    MemoryDocumentPathPolicy,
    MemoryDocumentRevisionStore,
    new_document_id,
    render_new_document,
)
from memoryos.workers.memory_document_edit_worker import MemoryDocumentEditWorker


def _plan(document_id: str, *, key: str = "worker-intent") -> DocumentEditPlan:
    return DocumentEditPlan(
        idempotency_key=key,
        tenant_id="default",
        owner_user_id="user-a",
        edit_kind=DocumentEditKind.CREATE,
        expected_state=ABSENT,
        evidence_digest="a" * 64,
        edit_summary="bounded worker recovery",
        document_id=document_id,
        relative_path="knowledge/topics/worker.md",
        after_bytes=render_new_document(document_id, "worker recovered content"),
        expected_registration_document_id=document_id,
    )


def _committer(
    root: Path,
    queue: InMemoryQueueStore,
    *,
    hook=None,  # noqa: ANN001
) -> MemoryDocumentCommitter:
    return MemoryDocumentCommitter(
        FileSystemMemoryDocumentStore(root),
        MemoryDocumentControlStore(root),
        MemoryDocumentRevisionStore(root),
        queue,
        test_hook=hook,
    )


def test_document_edit_worker_rolls_forward_exact_intent_and_acks(tmp_path: Path) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    def crash_after_prepare(stage, _intent) -> None:  # noqa: ANN001
        if stage == "intent_prepared":
            raise SimulatedProcessCrash

    queue = InMemoryQueueStore()
    document_id = new_document_id()
    committer = _committer(tmp_path, queue, hook=crash_after_prepare)
    with pytest.raises(SimulatedProcessCrash):
        committer.commit(
            _plan(document_id),
            actor_binding="trusted:user-a",
            evidence_reference="sealed-review:test",
        )
    intent = committer.control_store.incomplete_intents("default", "user-a")[0]
    committer.test_hook = None
    produced = queue.get(f"memory_document_edit_{intent.intent_id}")
    assert produced is not None
    assert produced.queue_name == "memory_document_edit"
    assert produced.action == "recover_document_intent"
    assert produced.target_uri == MemoryDocumentPathPolicy.document_uri("user-a", document_id)
    assert produced.payload == {
        "tenant_id": "default",
        "owner_user_id": "user-a",
        "document_id": document_id,
        "intent_id": intent.intent_id,
    }

    result = MemoryDocumentEditWorker(
        committer,
        queue,
        tenant_id="default",
        worker_id="document-worker",
    ).process_pending()

    assert result == {"claimed": 1, "committed": 1, "failed": 0, "dead_letter": 0}
    settled = queue.get(f"memory_document_edit_{intent.intent_id}")
    assert settled is not None and settled.status == "done"
    durable = committer.control_store.load_intent("default", "user-a", intent.intent_id)
    assert durable is not None and durable.status is DocumentIntentStatus.COMPLETED
    assert queue.get(durable.projection_job_id) is not None


def test_document_edit_worker_rejects_unproduced_sealed_review_variant(tmp_path: Path) -> None:
    queue = InMemoryQueueStore()
    committer = _committer(tmp_path, queue)
    digest = "b" * 64
    review_id = f"mdreview_{digest}"
    job_id = f"memory_document_edit_{review_id}"
    payload = {
        "tenant_id": "default",
        "owner_user_id": "user-a",
        "sealed_review_id": review_id,
        "sealed_review_digest": digest,
    }
    queue.enqueue(
        QueueJob(
            job_id=job_id,
            queue_name="memory_document_edit",
            action="commit_sealed_review",
            target_uri=f"memoryos://user/user-a/memory/reviews/{review_id}",
            payload=payload,
        )
    )
    result = MemoryDocumentEditWorker(committer, queue, tenant_id="default").process_pending()

    assert result == {
        "claimed": 1,
        "committed": 0,
        "failed": 1,
        "dead_letter": 1,
    }
    settled = queue.get(job_id)
    assert settled is not None and settled.status == "dead_letter"
    assert settled.last_error == "ValueError"


def test_document_edit_worker_dead_letters_payload_with_markdown_body(tmp_path: Path) -> None:
    queue = InMemoryQueueStore()
    committer = _committer(tmp_path, queue)
    document_id = new_document_id()
    job_id = "memory_document_edit_body_leak"
    queue.enqueue(
        QueueJob(
            job_id=job_id,
            queue_name="memory_document_edit",
            action="recover_document_intent",
            target_uri=MemoryDocumentPathPolicy.document_uri("user-a", document_id),
            payload={
                "tenant_id": "default",
                "owner_user_id": "user-a",
                "document_id": document_id,
                "intent_id": f"mdintent_{'c' * 64}",
                "markdown": "must never be interpreted",
            },
        )
    )

    result = MemoryDocumentEditWorker(committer, queue, tenant_id="default").process_pending()

    assert result == {"claimed": 1, "committed": 0, "failed": 1, "dead_letter": 1}
    settled = queue.get(job_id)
    assert settled is not None and settled.status == "dead_letter"
    assert settled.last_error == "ValueError"
