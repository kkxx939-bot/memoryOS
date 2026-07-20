from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from infrastructure.store.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from infrastructure.store.memory import (
    MemoryDocumentControlStore,
    MemoryDocumentRevisionStore,
    MemoryEditReviewIntegrityError,
    MemoryEditReviewStore,
    MemoryEditReviewWorkflow,
    ReviewConsolidationSource,
)
from infrastructure.store.memory.consolidation_store import MemoryDocumentConsolidationStore
from infrastructure.store.memory.erasure_store import MemoryDocumentEraseStore
from infrastructure.store.model.catalog import CatalogRecord
from memory.commit import (
    ConsolidationSource,
    ConsolidationStatus,
    MemoryDocumentCommitter,
    MemoryDocumentConsolidator,
    MemoryDocumentEraser,
)
from memory.core import (
    ABSENT,
    DocumentEditKind,
    DocumentEditPlan,
    PresentPath,
    new_document_id,
    render_new_document,
)
from memory.execute import MemoryDocumentPlanner
from memory.execute.command_service import MemoryCommandService
from memory.execute.pending_review_service import MemoryEditReviewService
from memory.ports import DocumentConflictError
from foundation.identity import LocalUserContext
from tests.support.persistence.in_memory import InMemoryQueueStore


class _ProjectionStore:
    def __init__(self) -> None:
        self.states: dict[tuple[str, str, str], dict[str, object]] = {}

    def confirm(self, *, document_id: str, source_digest: str, generation: int) -> None:
        self.states[("default", "user-a", document_id)] = {
            "tenant_id": "default",
            "owner_user_id": "user-a",
            "document_id": document_id,
            "source_digest": source_digest,
            "projection_generation": generation,
            "projection_status": "PROJECTED",
            "deletion_status": "",
        }

    def get_memory_document_projection_state(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> Mapping[str, object] | None:
        return self.states.get((tenant_id, owner_user_id, document_id))

    def replace_memory_document_projection(
        self,
        document_record: CatalogRecord | Mapping[str, object],
        block_records: Sequence[CatalogRecord | Mapping[str, object]],
        expected_previous_generation: int | None,
        *,
        tenant_id: str,
        owner_user_id: str,
        restore_soft_deleted: bool = False,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def tombstone_memory_document_projection(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        deletion_generation: int,
        deletion_event_digest: str,
        deletion_status: str,
        relative_path: str = "",
    ) -> tuple[str, ...]:
        raise NotImplementedError


def _components(root: Path):  # noqa: ANN202 - compact test fixture factory.
    documents = FileSystemMemoryDocumentStore(root)
    controls = MemoryDocumentControlStore(root)
    revisions = MemoryDocumentRevisionStore(root)
    queue = InMemoryQueueStore()
    committer = MemoryDocumentCommitter(
        documents,
        controls,
        revisions,
        queue,
        erasure_store=MemoryDocumentEraseStore(controls.root),
    )
    projections = _ProjectionStore()
    return documents, controls, revisions, committer, projections


def _create(
    committer: MemoryDocumentCommitter,
    *,
    document_id: str,
    relative_path: str,
    body: str,
) -> tuple[bytes, PresentPath]:
    raw = render_new_document(document_id, body)
    result = committer.commit(
        DocumentEditPlan(
            idempotency_key=f"create:{document_id}",
            tenant_id="default",
            owner_user_id="user-a",
            edit_kind=DocumentEditKind.CREATE,
            expected_state=ABSENT,
            evidence_digest=hashlib.sha256(body.encode()).hexdigest(),
            edit_summary="test source create",
            document_id=document_id,
            relative_path=relative_path,
            after_bytes=raw,
        ),
        actor_binding="trusted:user:user-a:user-a",
        evidence_reference=f"test-create:{document_id}",
    )
    assert result.control is not None
    return raw, PresentPath(relative_path, result.control.raw_sha256, result.control.size)


def _target_plan(document_id: str, body: str) -> DocumentEditPlan:
    raw = render_new_document(document_id, body)
    return DocumentEditPlan(
        idempotency_key=f"target-create:{document_id}",
        tenant_id="default",
        owner_user_id="user-a",
        edit_kind=DocumentEditKind.CREATE,
        expected_state=ABSENT,
        evidence_digest=hashlib.sha256(body.encode()).hexdigest(),
        edit_summary="consolidate exact source content into target",
        document_id=document_id,
        relative_path="knowledge/topics/combined.md",
        after_bytes=raw,
    )


def test_consolidation_waits_for_exact_target_projection_before_deleting_sources(tmp_path: Path) -> None:
    documents, _, _, committer, projections = _components(tmp_path)
    source_a = new_document_id()
    source_b = new_document_id()
    _, state_a = _create(
        committer,
        document_id=source_a,
        relative_path="knowledge/topics/a.md",
        body="alpha source secret",
    )
    _, state_b = _create(
        committer,
        document_id=source_b,
        relative_path="knowledge/topics/b.md",
        body="beta source secret",
    )
    target_id = new_document_id()
    target = _target_plan(target_id, "combined alpha source secret and beta source secret")
    sources = (
        ConsolidationSource(source_a, state_a.relative_path, state_a.raw_sha256, state_a.size),
        ConsolidationSource(source_b, state_b.relative_path, state_b.raw_sha256, state_b.size),
    )
    consolidator = MemoryDocumentConsolidator(
        committer,
        projections,
        saga_store=MemoryDocumentConsolidationStore(committer.control_store.root),
    )

    waiting = consolidator.consolidate(
        target,
        sources,
        idempotency_key="combine-a-and-b",
        actor_binding="trusted:user:user-a:user-a",
    )

    assert waiting.status == ConsolidationStatus.AWAITING_TARGET_PROJECTION
    assert waiting.target_projection_confirmed is False
    assert documents.read_state("default", "user-a", state_a.relative_path) == state_a
    assert documents.read_state("default", "user-a", state_b.relative_path) == state_b

    # A wrong digest at the correct generation is not confirmation.
    projections.confirm(
        document_id=target_id,
        source_digest="f" * 64,
        generation=waiting.target_projection_generation,
    )
    still_waiting = consolidator.consolidate(
        target,
        sources,
        idempotency_key="combine-a-and-b",
        actor_binding="trusted:user:user-a:user-a",
    )
    assert still_waiting.status == ConsolidationStatus.AWAITING_TARGET_PROJECTION
    assert documents.read_state("default", "user-a", state_a.relative_path) == state_a


def test_partial_source_commit_crash_replays_forward_without_content_loss(tmp_path: Path) -> None:
    documents, _, revisions, committer, projections = _components(tmp_path)
    source_a = new_document_id()
    source_b = new_document_id()
    _, state_a = _create(
        committer,
        document_id=source_a,
        relative_path="knowledge/topics/a.md",
        body="alpha source secret",
    )
    _, state_b = _create(
        committer,
        document_id=source_b,
        relative_path="knowledge/topics/b.md",
        body="beta source secret",
    )
    target_id = new_document_id()
    target_raw = render_new_document(target_id, "combined alpha source secret and beta source secret")
    target = _target_plan(target_id, "combined alpha source secret and beta source secret")
    sources = (
        ConsolidationSource(source_a, state_a.relative_path, state_a.raw_sha256, state_a.size),
        ConsolidationSource(source_b, state_b.relative_path, state_b.raw_sha256, state_b.size),
    )
    store = MemoryDocumentConsolidationStore(tmp_path)
    initial = MemoryDocumentConsolidator(committer, projections, saga_store=store).consolidate(
        target,
        sources,
        idempotency_key="crashable-combine",
        actor_binding="trusted:user:user-a:user-a",
    )
    projections.confirm(
        document_id=target_id,
        source_digest=hashlib.sha256(target_raw).hexdigest(),
        generation=initial.target_projection_generation,
    )

    def crash_after_first_source(stage: str, record) -> None:  # noqa: ANN001
        if stage == "after_source_commit" and record.next_source_index == 0:
            raise RuntimeError("simulated process crash")

    crashing = MemoryDocumentConsolidator(
        committer,
        projections,
        saga_store=store,
        test_hook=crash_after_first_source,
    )
    with pytest.raises(RuntimeError, match="simulated process crash"):
        crashing.consolidate(
            target,
            sources,
            idempotency_key="crashable-combine",
            actor_binding="trusted:user:user-a:user-a",
        )

    # The target is exact, source A is already gone, and source B remains.  No
    # rollback occurs and the journal cursor intentionally still precedes A.
    assert documents.read_raw("default", "user-a", document_id=target_id) == target_raw
    assert documents.read_state("default", "user-a", state_a.relative_path) == ABSENT
    assert documents.read_state("default", "user-a", state_b.relative_path) == state_b
    journal = store.load("default", "user-a", initial.saga_id)
    assert journal is not None and journal.next_source_index == 0
    journal_bytes = next(tmp_path.rglob(f"{initial.saga_id}.json")).read_bytes()
    assert b"alpha source secret" not in journal_bytes
    assert b"beta source secret" not in journal_bytes

    completed = MemoryDocumentConsolidator(
        committer,
        projections,
        saga_store=store,
    ).resume(
        tenant_id="default",
        owner_user_id="user-a",
        saga_id=initial.saga_id,
        actor_binding="trusted:user:user-a:user-a",
    )

    assert completed.status == ConsolidationStatus.COMPLETED
    assert completed.soft_forgotten_document_ids == (source_a, source_b)
    assert completed.pending_document_ids == ()
    assert documents.read_raw("default", "user-a", document_id=target_id) == target_raw
    assert documents.read_state("default", "user-a", state_a.relative_path) == ABSENT
    assert documents.read_state("default", "user-a", state_b.relative_path) == ABSENT
    assert revisions.read_revision_blob("default", "user-a", source_a, 2) is not None
    assert revisions.read_revision_blob("default", "user-a", source_b, 2) is not None


def test_resume_recovers_target_intent_after_crash_before_saga_checkpoint(tmp_path: Path) -> None:
    documents, _, _, committer, projections = _components(tmp_path)
    target_id = new_document_id()
    target = _target_plan(target_id, "durable target body")

    def crash_after_target(stage: str, _record) -> None:  # noqa: ANN001
        if stage == "after_target_commit":
            raise RuntimeError("target checkpoint crash")

    store = MemoryDocumentConsolidationStore(tmp_path)
    with pytest.raises(RuntimeError, match="target checkpoint crash"):
        MemoryDocumentConsolidator(
            committer,
            projections,
            saga_store=store,
            test_hook=crash_after_target,
        ).consolidate(
            target,
            (),
            idempotency_key="target-checkpoint-crash",
            actor_binding="trusted:user:user-a:user-a",
        )

    saga_files = tuple(tmp_path.rglob("memsaga_*.json"))
    assert len(saga_files) == 1
    saga_id = saga_files[0].stem
    journal = store.load("default", "user-a", saga_id)
    assert journal is not None and journal.status == ConsolidationStatus.PREPARED
    assert documents.read_raw("default", "user-a", document_id=target_id) == target.after_bytes

    resumed = MemoryDocumentConsolidator(committer, projections, saga_store=store).resume(
        tenant_id="default",
        owner_user_id="user-a",
        saga_id=saga_id,
        actor_binding="trusted:user:user-a:user-a",
    )
    assert resumed.status == ConsolidationStatus.AWAITING_TARGET_PROJECTION
    assert resumed.pending_document_ids == ()


def test_resume_all_reports_prepared_saga_without_target_intent_and_preserves_source(tmp_path: Path) -> None:
    documents, _, _, committer, projections = _components(tmp_path)
    source_id = new_document_id()
    _, source_state = _create(
        committer,
        document_id=source_id,
        relative_path="knowledge/topics/source.md",
        body="source must survive missing target input",
    )
    target_id = new_document_id()
    target = _target_plan(target_id, "target body supplied only by caller")
    source = ConsolidationSource(
        source_id,
        source_state.relative_path,
        source_state.raw_sha256,
        source_state.size,
    )
    store = MemoryDocumentConsolidationStore(tmp_path)

    def crash_after_journal(stage: str, _record) -> None:  # noqa: ANN001
        if stage == "after_saga_checkpoint":
            raise RuntimeError("caller disappeared before target prepare")

    with pytest.raises(RuntimeError, match="caller disappeared"):
        MemoryDocumentConsolidator(
            committer,
            projections,
            saga_store=store,
            test_hook=crash_after_journal,
        ).consolidate(
            target,
            (source,),
            idempotency_key="missing-target-input",
            actor_binding="trusted:user:user-a:user-a",
        )

    pending = store.list_pending("default", "user-a", limit=1)
    assert len(pending) == 1 and pending[0].status == ConsolidationStatus.PREPARED
    report = MemoryDocumentConsolidator(
        committer,
        projections,
        saga_store=store,
    ).resume_all(tenant_id="default", owner_user_id="user-a", limit=1)

    assert report.examined == 1
    assert report.awaiting_input_saga_ids == (pending[0].saga_id,)
    assert report.completed_saga_ids == ()
    assert documents.read_state("default", "user-a", source_state.relative_path) == source_state
    assert documents.read_state("default", "user-a", target.relative_path) == ABSENT
    with pytest.raises(ValueError, match="list limit"):
        store.list_pending("default", "user-a", limit=0)


def test_trusted_command_service_starts_configured_consolidation(tmp_path: Path) -> None:
    documents, controls, revisions, committer, projections = _components(tmp_path)
    source_id = new_document_id()
    _, source_state = _create(
        committer,
        document_id=source_id,
        relative_path="knowledge/topics/command-source.md",
        body="command source remains until projection",
    )
    target_id = new_document_id()
    target = _target_plan(target_id, "command target contains source")
    consolidator = MemoryDocumentConsolidator(
        committer,
        projections,
        saga_store=MemoryDocumentConsolidationStore(committer.control_store.root),
    )
    commands = MemoryCommandService(
        MemoryDocumentPlanner(documents),
        committer,
        MemoryDocumentEraser(
            documents,
            controls,
            revisions,
            erase_store=MemoryDocumentEraseStore(controls.root),
        ),
        consolidator=consolidator,
    )
    caller = LocalUserContext(
        user_id="user-a",
    )

    result = commands.consolidate_memory_documents(
        target,
        (
            ConsolidationSource(
                source_id,
                source_state.relative_path,
                source_state.raw_sha256,
                source_state.size,
            ),
        ),
        idempotency_key="trusted-command-consolidation",
        caller=caller,
    )

    assert result.status == ConsolidationStatus.AWAITING_TARGET_PROJECTION
    assert documents.read_state("default", "user-a", source_state.relative_path) == source_state


def test_public_merge_inputs_are_resolved_to_exact_live_documents_and_resume_by_saga(tmp_path: Path) -> None:
    documents, controls, revisions, committer, projections = _components(tmp_path)
    target_id = new_document_id()
    _, target_state = _create(
        committer,
        document_id=target_id,
        relative_path="knowledge/topics/merge-target.md",
        body="target before merge",
    )
    source_id = new_document_id()
    _, source_state = _create(
        committer,
        document_id=source_id,
        relative_path="knowledge/topics/merge-source.md",
        body="source content retained until target projection",
    )
    commands = MemoryCommandService(
        MemoryDocumentPlanner(documents),
        committer,
        MemoryDocumentEraser(
            documents,
            controls,
            revisions,
            erase_store=MemoryDocumentEraseStore(controls.root),
        ),
        consolidator=MemoryDocumentConsolidator(
            committer,
            projections,
            saga_store=MemoryDocumentConsolidationStore(committer.control_store.root),
        ),
    )
    caller = LocalUserContext(
        user_id="user-a",
    )
    target_uri = f"memoryos://user/user-a/memory/documents/{target_id}"
    source_uri = f"memoryos://user/user-a/memory/documents/{source_id}"

    waiting = commands.merge_memory_documents(
        target_uri,
        "target plus source content retained until target projection",
        target_state.raw_sha256,
        ({"document_uri": source_uri, "expected_digest": source_state.raw_sha256},),
        caller=caller,
    )

    assert waiting.status is ConsolidationStatus.AWAITING_TARGET_PROJECTION
    assert documents.read_state("default", "user-a", source_state.relative_path) == source_state
    target_control = controls.load_control("default", "user-a", target_id)
    assert target_control is not None
    projections.confirm(
        document_id=target_id,
        source_digest=target_control.raw_sha256,
        generation=target_control.projection_generation,
    )

    completed = commands.resume_memory_consolidation(waiting.saga_id, caller=caller)

    assert completed.status is ConsolidationStatus.COMPLETED
    assert completed.soft_forgotten_document_ids == (source_id,)
    assert documents.read_state("default", "user-a", source_state.relative_path) == ABSENT
    assert b"target plus source content" in documents.read_raw(
        "default",
        "user-a",
        document_id=target_id,
    )


def test_copy_on_write_consolidation_is_sealed_previewed_approved_and_restorable(
    tmp_path: Path,
) -> None:
    documents, controls, revisions, committer, projections = _components(tmp_path)
    target_id = new_document_id()
    target_raw, target_state = _create(
        committer,
        document_id=target_id,
        relative_path="knowledge/topics/dream-target.md",
        body="target before Dreams consolidation",
    )
    source_id = new_document_id()
    source_raw, source_state = _create(
        committer,
        document_id=source_id,
        relative_path="knowledge/topics/dream-source.md",
        body="source fact retained until reviewed projection",
    )
    review_store = MemoryEditReviewStore(tmp_path)
    consolidator = MemoryDocumentConsolidator(
        committer,
        projections,
        saga_store=MemoryDocumentConsolidationStore(committer.control_store.root),
    )
    commands = MemoryCommandService(
        MemoryDocumentPlanner(documents),
        committer,
        MemoryDocumentEraser(
            documents,
            controls,
            revisions,
            review_store=review_store,
            erase_store=MemoryDocumentEraseStore(controls.root),
        ),
        consolidator=consolidator,
        review_store=review_store,
    )
    reviews = MemoryEditReviewService(
        review_store,
        committer,
        consolidator=consolidator,
        erasure_store=MemoryDocumentEraseStore(committer.control_store.root),
    )
    caller = LocalUserContext(
        user_id="user-a",
    )
    target_uri = f"memoryos://user/user-a/memory/documents/{target_id}"
    source_uri = f"memoryos://user/user-a/memory/documents/{source_id}"

    proposal = commands.propose_memory_consolidation(
        target_uri,
        "target plus source fact retained until reviewed projection",
        target_state.raw_sha256,
        ({"document_uri": source_uri, "expected_digest": source_state.raw_sha256},),
        caller=caller,
    )

    assert proposal.status == "PENDING"
    assert proposal.workflow_kind == "CONSOLIDATION"
    assert "source fact retained" in proposal.proposed_diff
    assert documents.read_raw("default", "user-a", document_id=target_id) == target_raw
    assert documents.read_raw("default", "user-a", document_id=source_id) == source_raw
    record = review_store.load("default", "user-a", proposal.proposal_id)
    assert record is not None
    assert tuple(source.document_id for source in record.consolidation_sources) == (source_id,)
    assert b"target plus source fact" in (review_store.load_after_blob(record) or b"")
    preview = reviews.preview_edit(proposal.proposal_id, caller=caller)
    assert preview.proposed_diff == proposal.proposed_diff
    assert preview.consolidation_sources == proposal.consolidation_sources

    waiting = reviews.review_edit(proposal.proposal_id, "APPROVE", caller=caller)

    assert waiting.status == "APPROVED"
    assert waiting.consolidation_status == ConsolidationStatus.AWAITING_TARGET_PROJECTION.value
    assert waiting.consolidation_saga_id
    assert documents.read_state("default", "user-a", source_state.relative_path) == source_state
    target_control = controls.load_control("default", "user-a", target_id)
    assert target_control is not None
    projections.confirm(
        document_id=target_id,
        source_digest=target_control.raw_sha256,
        generation=target_control.projection_generation,
    )

    completed = reviews.review_edit(proposal.proposal_id, "APPROVE", caller=caller)

    assert completed.consolidation_status == ConsolidationStatus.COMPLETED.value
    assert completed.soft_forgotten_document_ids == (source_id,)
    assert documents.read_state("default", "user-a", source_state.relative_path) == ABSENT
    restored = commands.restore_memory_revision(source_uri, 1, "", caller=caller)
    assert restored.changed is True
    assert b"source fact retained" in documents.read_raw("default", "user-a", document_id=source_id)


def test_consolidation_proposal_rejects_or_conflicts_without_mutating_target(tmp_path: Path) -> None:
    documents, controls, revisions, committer, projections = _components(tmp_path)
    target_id = new_document_id()
    target_raw, target_state = _create(
        committer,
        document_id=target_id,
        relative_path="knowledge/topics/review-target.md",
        body="target remains unchanged before approval",
    )
    source_id = new_document_id()
    _, source_state = _create(
        committer,
        document_id=source_id,
        relative_path="knowledge/topics/review-source.md",
        body="source v1",
    )
    review_store = MemoryEditReviewStore(tmp_path)
    consolidator = MemoryDocumentConsolidator(
        committer,
        projections,
        saga_store=MemoryDocumentConsolidationStore(committer.control_store.root),
    )
    commands = MemoryCommandService(
        MemoryDocumentPlanner(documents),
        committer,
        MemoryDocumentEraser(
            documents,
            controls,
            revisions,
            review_store=review_store,
            erase_store=MemoryDocumentEraseStore(controls.root),
        ),
        consolidator=consolidator,
        review_store=review_store,
    )
    reviews = MemoryEditReviewService(
        review_store,
        committer,
        consolidator=consolidator,
        erasure_store=MemoryDocumentEraseStore(committer.control_store.root),
    )
    caller = LocalUserContext(
        user_id="user-a",
    )
    target_uri = f"memoryos://user/user-a/memory/documents/{target_id}"
    source_uri = f"memoryos://user/user-a/memory/documents/{source_id}"
    request = (
        target_uri,
        "target plus source v1",
        target_state.raw_sha256,
        ({"document_uri": source_uri, "expected_digest": source_state.raw_sha256},),
    )

    rejected_proposal = commands.propose_memory_consolidation(*request, caller=caller)
    rejected = reviews.review_edit(rejected_proposal.proposal_id, "REJECT", caller=caller)
    assert rejected.status == "REJECTED"
    assert documents.read_raw("default", "user-a", document_id=target_id) == target_raw

    conflict_proposal = commands.propose_memory_consolidation(
        target_uri,
        "a different consolidated target",
        target_state.raw_sha256,
        request[3],
        caller=caller,
    )
    edited_source = commands.edit_memory_document(
        source_uri,
        "source v2 changed after proposal",
        source_state.raw_sha256,
        caller=caller,
    )
    with pytest.raises(DocumentConflictError, match="source changed after its sealed proposal"):
        reviews.review_edit(conflict_proposal.proposal_id, "APPROVE", caller=caller)
    assert edited_source.changed is True
    assert documents.read_raw("default", "user-a", document_id=target_id) == target_raw


def test_pending_consolidation_review_recovers_existing_saga_after_target_commit_crash(
    tmp_path: Path,
) -> None:
    documents, controls, revisions, committer, projections = _components(tmp_path)
    target_id = new_document_id()
    _, target_state = _create(
        committer,
        document_id=target_id,
        relative_path="knowledge/topics/review-crash-target.md",
        body="target before review crash",
    )
    source_id = new_document_id()
    _, source_state = _create(
        committer,
        document_id=source_id,
        relative_path="knowledge/topics/review-crash-source.md",
        body="source retained across review crash",
    )
    review_store = MemoryEditReviewStore(tmp_path)
    crashed = False

    def stop_after_target_commit(stage: str, _record) -> None:  # noqa: ANN001
        nonlocal crashed
        if stage == "after_target_commit" and not crashed:
            crashed = True
            raise RuntimeError("process stopped before review approval transition")

    consolidator = MemoryDocumentConsolidator(
        committer,
        projections,
        test_hook=stop_after_target_commit,
        saga_store=MemoryDocumentConsolidationStore(committer.control_store.root),
    )
    commands = MemoryCommandService(
        MemoryDocumentPlanner(documents),
        committer,
        MemoryDocumentEraser(
            documents,
            controls,
            revisions,
            review_store=review_store,
            erase_store=MemoryDocumentEraseStore(controls.root),
        ),
        consolidator=consolidator,
        review_store=review_store,
    )
    reviews = MemoryEditReviewService(
        review_store,
        committer,
        consolidator=consolidator,
        erasure_store=MemoryDocumentEraseStore(committer.control_store.root),
    )
    caller = LocalUserContext(
        user_id="user-a",
    )
    target_uri = f"memoryos://user/user-a/memory/documents/{target_id}"
    source_uri = f"memoryos://user/user-a/memory/documents/{source_id}"
    proposal = commands.propose_memory_consolidation(
        target_uri,
        "target plus source retained across review crash",
        target_state.raw_sha256,
        ({"document_uri": source_uri, "expected_digest": source_state.raw_sha256},),
        caller=caller,
    )

    with pytest.raises(RuntimeError, match="approval transition"):
        reviews.review_edit(proposal.proposal_id, "APPROVE", caller=caller)

    pending = review_store.load("default", "user-a", proposal.proposal_id)
    assert pending is not None and pending.status.value == "PENDING"
    assert documents.read_state("default", "user-a", source_state.relative_path) == source_state
    assert documents.read_state("default", "user-a", target_state.relative_path) != target_state
    consolidator.test_hook = None

    resumed = reviews.review_edit(proposal.proposal_id, "APPROVE", caller=caller)

    assert resumed.status == "APPROVED"
    assert resumed.consolidation_saga_id
    assert resumed.consolidation_status == ConsolidationStatus.AWAITING_TARGET_PROJECTION.value
    assert documents.read_state("default", "user-a", source_state.relative_path) == source_state


def test_consolidation_proposal_canonicalizes_source_order_for_idempotent_reseal(
    tmp_path: Path,
) -> None:
    documents, controls, revisions, committer, projections = _components(tmp_path)
    target_id = new_document_id()
    _, target_state = _create(
        committer,
        document_id=target_id,
        relative_path="knowledge/topics/order-target.md",
        body="target",
    )
    source_states: list[tuple[str, PresentPath]] = []
    for name in ("a", "b"):
        document_id = new_document_id()
        _, state = _create(
            committer,
            document_id=document_id,
            relative_path=f"knowledge/topics/order-{name}.md",
            body=f"source {name}",
        )
        source_states.append((document_id, state))
    review_store = MemoryEditReviewStore(tmp_path)
    commands = MemoryCommandService(
        MemoryDocumentPlanner(documents),
        committer,
        MemoryDocumentEraser(
            documents,
            controls,
            revisions,
            review_store=review_store,
            erase_store=MemoryDocumentEraseStore(controls.root),
        ),
        consolidator=MemoryDocumentConsolidator(
            committer,
            projections,
            saga_store=MemoryDocumentConsolidationStore(committer.control_store.root),
        ),
        review_store=review_store,
    )
    caller = LocalUserContext(
        user_id="user-a",
    )
    target_uri = f"memoryos://user/user-a/memory/documents/{target_id}"
    sources = tuple(
        {
            "document_uri": f"memoryos://user/user-a/memory/documents/{document_id}",
            "expected_digest": state.raw_sha256,
        }
        for document_id, state in source_states
    )

    first = commands.propose_memory_consolidation(
        target_uri,
        "target plus source a and source b",
        target_state.raw_sha256,
        sources,
        caller=caller,
    )
    reversed_retry = commands.propose_memory_consolidation(
        target_uri,
        "target plus source a and source b",
        target_state.raw_sha256,
        tuple(reversed(sources)),
        caller=caller,
    )

    assert reversed_retry.proposal_id == first.proposal_id
    assert reversed_retry.consolidation_sources == first.consolidation_sources


def test_hard_erase_closes_shared_review_blob_references_without_dangling_records(
    tmp_path: Path,
) -> None:
    store = MemoryEditReviewStore(tmp_path)
    target_id = new_document_id()
    source_id = new_document_id()
    shared_after = render_new_document(target_id, "shared erased source body")

    def plan(label: str, document_id: str = target_id) -> DocumentEditPlan:
        return DocumentEditPlan(
            idempotency_key=f"review-shared:{label}",
            tenant_id="default",
            owner_user_id="user-a",
            edit_kind=DocumentEditKind.CREATE,
            expected_state=ABSENT,
            evidence_digest=hashlib.sha256(label.encode()).hexdigest(),
            edit_summary=f"shared review {label}",
            document_id=document_id,
            relative_path=f"knowledge/topics/{document_id}.md",
            after_bytes=(
                shared_after
                if document_id == target_id
                else render_new_document(document_id, "unrelated retained body")
            ),
        )

    consolidation = store.seal(
        plan("consolidation"),
        proposed_diff="shared erased source diff",
        workflow_kind=MemoryEditReviewWorkflow.CONSOLIDATION,
        consolidation_sources=(
            ReviewConsolidationSource(
                document_id=source_id,
                relative_path="knowledge/topics/source.md",
                raw_sha256="a" * 64,
                size=123,
            ),
        ),
    )
    shares_both = store.seal(
        plan("shares-both"),
        proposed_diff="shared erased source diff",
    )
    shares_after = store.seal(
        plan("shares-after"),
        proposed_diff="unique collateral diff that must also be removed",
    )
    retained = store.seal(
        plan("unrelated", new_document_id()),
        proposed_diff="unrelated retained diff",
    )

    store.purge_document("default", "user-a", source_id)

    for record in (consolidation, shares_both, shares_after):
        assert store.load("default", "user-a", record.proposal_id) is None
    assert store.load("default", "user-a", retained.proposal_id) == retained
    target_blobs = tuple(
        (tmp_path / "system" / "memory-documents" / "user-a" / "review-blobs" / target_id).glob("*.blob")
    )
    assert target_blobs == ()
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert b"shared erased source body" not in artifact.read_bytes()
            assert b"unique collateral diff" not in artifact.read_bytes()


def test_review_hard_erase_fails_closed_before_unbounded_owner_enumeration(
    tmp_path: Path,
) -> None:
    store = MemoryEditReviewStore(tmp_path, max_owner_records=1)
    records = tuple(
        store.seal(
            _target_plan(new_document_id(), f"bounded body {index}"),
            proposed_diff=f"bounded diff {index}",
        )
        for index in range(2)
    )

    with pytest.raises(MemoryEditReviewIntegrityError, match="hard limit"):
        store.purge_document("default", "user-a", records[0].document_id)

    for record in records:
        assert store.load("default", "user-a", record.proposal_id) == record
        assert store.load_after_blob(record) is not None


def test_consolidation_review_crash_cannot_orphan_an_unbound_body_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryEditReviewStore(tmp_path)
    target_id = new_document_id()
    source_id = new_document_id()
    plan = _target_plan(target_id, "source secret copied into proposed target")
    durable_stage = store._stage_blob
    staged = 0

    def stop_after_first_blob(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - crash boundary.
        nonlocal staged
        durable_stage(*args, **kwargs)
        staged += 1
        if staged == 1:
            raise RuntimeError("process stopped after first review body blob")

    monkeypatch.setattr(store, "_stage_blob", stop_after_first_blob)
    with pytest.raises(RuntimeError, match="first review body blob"):
        store.seal(
            plan,
            proposed_diff="source secret proposed diff",
            workflow_kind=MemoryEditReviewWorkflow.CONSOLIDATION,
            consolidation_sources=(
                ReviewConsolidationSource(
                    document_id=source_id,
                    relative_path="knowledge/topics/source.md",
                    raw_sha256="a" * 64,
                    size=10,
                ),
            ),
        )

    records = tuple((tmp_path / "system" / "memory-documents" / "user-a" / "reviews").glob("*.json"))
    assert len(records) == 1
    assert tuple(tmp_path.rglob("*.blob"))
    monkeypatch.setattr(store, "_stage_blob", durable_stage)

    store.purge_document("default", "user-a", source_id)

    assert not records[0].exists()
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert b"source secret" not in artifact.read_bytes()
