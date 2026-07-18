from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from memoryos.core.readiness import RuntimeReadinessState
from memoryos.memory.documents import (
    ABSENT,
    ConsolidationSource,
    ConsolidationStatus,
    DocumentEditKind,
    DocumentEditPlan,
    MemoryDocumentConsolidator,
    PresentPath,
    new_document_id,
    render_new_document,
)
from memoryos.runtime import RuntimeConfig, build_runtime_container


def _create_plan(document_id: str, relative_path: str, body: str, *, key: str) -> DocumentEditPlan:
    return DocumentEditPlan(
        idempotency_key=key,
        tenant_id="default",
        owner_user_id="user-a",
        edit_kind=DocumentEditKind.CREATE,
        expected_state=ABSENT,
        evidence_digest=hashlib.sha256(body.encode()).hexdigest(),
        edit_summary="runtime consolidation recovery fixture",
        document_id=document_id,
        relative_path=relative_path,
        after_bytes=render_new_document(document_id, body),
    )


def test_startup_resumes_consolidation_around_projection_drain(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    first = build_runtime_container(RuntimeConfig(root=str(root)))
    source_id = new_document_id()
    source_plan = _create_plan(
        source_id,
        "knowledge/topics/redundant.md",
        "redundant source survives until target projection",
        key="runtime-source-create",
    )
    source_commit = first.memory_document_committer.commit(
        source_plan,
        actor_binding="trusted:user:user-a:user-a",
        evidence_reference="runtime-test:source",
    )
    assert source_commit.control is not None
    source_state = PresentPath(
        source_plan.relative_path,
        source_commit.control.raw_sha256,
        source_commit.control.size,
    )
    target_id = new_document_id()
    target_plan = _create_plan(
        target_id,
        "knowledge/topics/consolidated.md",
        "target retains redundant source survives until target projection",
        key="runtime-target-create",
    )

    def crash_before_target_checkpoint(stage: str, _record) -> None:  # noqa: ANN001
        if stage == "after_target_commit":
            raise RuntimeError("restart after durable target commit")

    crashing = MemoryDocumentConsolidator(
        first.memory_document_committer,
        first.index_store,  # type: ignore[arg-type]
        saga_store=first.memory_document_consolidation_store,
        test_hook=crash_before_target_checkpoint,
    )
    with pytest.raises(RuntimeError, match="restart after durable target commit"):
        crashing.consolidate(
            target_plan,
            (
                ConsolidationSource(
                    source_id,
                    source_state.relative_path,
                    source_state.raw_sha256,
                    source_state.size,
                ),
            ),
            idempotency_key="runtime-recover-consolidation",
            actor_binding="trusted:user:user-a:user-a",
        )

    pending = first.memory_document_consolidation_store.list_pending("default", "user-a")
    assert len(pending) == 1 and pending[0].status == ConsolidationStatus.PREPARED

    restarted = build_runtime_container(RuntimeConfig(root=str(root)))

    assert restarted.readiness.state == RuntimeReadinessState.READY
    assert restarted.memory_document_store.read_state(
        "default",
        "user-a",
        source_plan.relative_path,
    ) == ABSENT
    assert restarted.memory_document_store.read_raw(
        "default",
        "user-a",
        document_id=target_id,
    ) == target_plan.after_bytes
    records = restarted.memory_document_consolidation_store.list_records("default", "user-a")
    assert len(records) == 1 and records[0].status == ConsolidationStatus.COMPLETED
    details = restarted.readiness.details
    assert details["memory_consolidations_pre_projection"]["awaiting_projection"] == 1
    assert details["memory_consolidations_post_projection"]["completed"] == 1
    assert details["memory_projection_queue_after_consolidation"].get("pending", 0) == 0
