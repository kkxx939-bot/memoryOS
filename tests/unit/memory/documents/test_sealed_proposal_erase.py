from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from memoryos.adapters.persistence.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from memoryos.adapters.persistence.in_memory.queue_store import InMemoryQueueStore
from memoryos.application.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.integrity import canonical_digest
from memoryos.memory.documents import (
    DerivedEraseRequest,
    DocumentErasedError,
    DocumentEraseStatus,
    MemoryCandidateKind,
    MemoryDocumentCommitter,
    MemoryDocumentControlStore,
    MemoryDocumentEraser,
    MemoryDocumentPlanner,
    MemoryDocumentRevisionStore,
    MemoryEditProposal,
    PresentPath,
)
from memoryos.memory.evidence import SealedProposalEraseBackend, SealedProposalStore
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend

_SECRET = "sealed-session-proposal-secret"
_DOCUMENT_A = "memdoc_AAAAAAAAAAAAAAAA"
_DOCUMENT_B = "memdoc_BBBBBBBBBBBBBBBB"
_DOCUMENT_C = "memdoc_CCCCCCCCCCCCCCCC"


def _proposal(body: str = _SECRET) -> MemoryEditProposal:
    return MemoryEditProposal(
        candidate_kind=MemoryCandidateKind.PREFERENCE,
        title="Private preference",
        body=body,
        evidence_refs=("event-1",),
    )


def _seal(
    store: SealedProposalStore,
    *,
    owner: str,
    task: str,
    proposal: MemoryEditProposal | None = None,
):  # noqa: ANN202 - compact artifact fixture.
    return store.seal(
        task_id=task,
        owner_user_id=owner,
        archive_uri=f"memoryos://user/{owner}/sessions/session-1",
        archive_digest="a" * 64,
        manifest_digest="b" * 64,
        proposals=(proposal or _proposal(),),
    )


def _request(document_id: str, *, owner: str = "user-a", tenant: str = "default") -> DerivedEraseRequest:
    return DerivedEraseRequest(
        tenant_id=tenant,
        owner_user_id=owner,
        document_id=document_id,
        document_uri=f"memoryos://user/{owner}/memory/documents/{document_id}",
        relative_path="preferences.md",
        document_kind="preferences",
        erasure_epoch=f"erase_{'e' * 64}",
        source_digest="f" * 64,
        document_revision_floor=1,
        projection_generation_floor=1,
    )


def test_exact_document_binding_erases_whole_multi_document_task_without_dangling_links(
    tmp_path: Path,
) -> None:
    store = SealedProposalStore(tmp_path, tenant_id="default")
    sealed = _seal(store, owner="user-a", task="task-multi")
    binding = store.bind_documents(
        task_id=sealed.task_id,
        owner_user_id=sealed.owner_user_id,
        proposal_set_digest=sealed.proposal_set_digest,
        document_bindings=((_DOCUMENT_A, "1" * 64), (_DOCUMENT_B, "2" * 64)),
    )
    other = _seal(store, owner="user-a", task="task-other", proposal=_proposal("other body"))
    store.bind_documents(
        task_id=other.task_id,
        owner_user_id=other.owner_user_id,
        proposal_set_digest=other.proposal_set_digest,
        document_bindings=((_DOCUMENT_C, "3" * 64),),
    )
    other_owner = _seal(store, owner="user-b", task="task-multi", proposal=_proposal("owner-b body"))
    store.bind_documents(
        task_id=other_owner.task_id,
        owner_user_id=other_owner.owner_user_id,
        proposal_set_digest=other_owner.proposal_set_digest,
        document_bindings=((_DOCUMENT_A, "4" * 64),),
    )

    catalog_bytes = store.binding_catalog_path("user-a").read_bytes()
    assert _SECRET.encode() not in catalog_bytes
    assert binding.binding_digest.encode() in catalog_bytes

    assert SealedProposalEraseBackend(store).erase_document(_request(_DOCUMENT_A)) is True

    assert not store.path("user-a", "task-multi").exists()
    assert store.bindings_for_document("user-a", _DOCUMENT_A) == ()
    assert store.bindings_for_document("user-a", _DOCUMENT_B) == ()
    assert [item.task_id for item in store.bindings_for_document("user-a", _DOCUMENT_C)] == [
        "task-other"
    ]
    assert store.path("user-a", "task-other").exists()
    assert store.path("user-b", "task-multi").exists()
    assert [item.task_id for item in store.bindings_for_document("user-b", _DOCUMENT_A)] == [
        "task-multi"
    ]
    barrier = store.erasure_barrier_path("user-a", "task-multi")
    assert barrier.exists()
    assert _SECRET.encode() not in barrier.read_bytes()
    assert _DOCUMENT_A.encode() not in barrier.read_bytes()
    assert _DOCUMENT_B.encode() not in barrier.read_bytes()

    # The backend is idempotent after its exact binding has already been consumed.
    assert SealedProposalEraseBackend(store).erase_document(_request(_DOCUMENT_A)) is True
    with pytest.raises(DocumentErasedError):
        _seal(store, owner="user-a", task="task-multi")


def test_session_replay_hits_proposal_erasure_barrier_before_model_or_reseal(tmp_path: Path) -> None:
    response = json.dumps(
        {
            "candidates": [
                {
                    "candidate_kind": "preference",
                    "title": "Private preference",
                    "body": _SECRET,
                    "evidence_refs": ["event-1"],
                    "field_evidence_refs": {"body": ["event-1"]},
                    "confidence": 0.95,
                }
            ]
        }
    )
    provider = FakeMemoryModelProvider(response)
    store = SealedProposalStore(tmp_path, tenant_id="default")
    planner = MemoryCommitPlanner(
        MemoryDocumentPlanner(FileSystemMemoryDocumentStore(tmp_path)),
        extractor=LLMMemoryExtractorBackend(provider),
        proposal_store=store,
        tenant_id="default",
    )
    archive = SessionArchive(
        user_id="user-a",
        session_id="session-1",
        archive_uri="memoryos://user/user-a/sessions/session-1",
        task_id="task-replay",
        created_at="2026-07-18T00:00:00Z",
        archive_digest="a" * 64,
        manifest_digest="b" * 64,
        messages=[
            {
                "id": "event-1",
                "role": "user",
                "content": "Remember a private preference.",
                "occurred_at": "2026-07-18T00:00:00Z",
            }
        ],
        metadata={"tenant_id": "default"},
    )
    first = planner.plan_session(
        archive,
        tenant_id="default",
        owner_user_id="user-a",
        commit_group_id="commit-group-replay",
    )
    document_id = first.edits[0].plan.document_id
    SealedProposalEraseBackend(store).erase_document(_request(document_id))

    with pytest.raises(DocumentErasedError):
        planner.plan_session(
            archive,
            tenant_id="default",
            owner_user_id="user-a",
            commit_group_id="commit-group-replay",
        )
    assert provider.calls == 1
    assert not store.path("user-a", archive.task_id).exists()


@pytest.mark.parametrize("unsafe_kind", ["hardlink", "symlink"])
def test_unsafe_sealed_set_keeps_durable_eraser_pending_until_safe_retry(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    source = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    revisions = MemoryDocumentRevisionStore(tmp_path)
    committer = MemoryDocumentCommitter(source, controls, revisions, InMemoryQueueStore())
    proposal = _proposal()
    plan = MemoryDocumentPlanner(source).plan(
        proposal,
        tenant_id="default",
        owner_user_id="user-a",
        idempotency_key="session-task-unsafe:memory:0",
        evidence_digest="a" * 64,
    )
    committer.commit(
        plan,
        actor_binding="trusted-runtime:user-a",
        evidence_reference="memoryos://user/user-a/sessions/session-unsafe#event-1",
    )
    live = source.read_state("default", "user-a", plan.relative_path)
    assert isinstance(live, PresentPath)

    proposal_store = SealedProposalStore(tmp_path, tenant_id="default")
    sealed = _seal(proposal_store, owner="user-a", task="task-unsafe", proposal=proposal)
    proposal_store.bind_documents(
        task_id=sealed.task_id,
        owner_user_id=sealed.owner_user_id,
        proposal_set_digest=sealed.proposal_set_digest,
        document_bindings=((plan.document_id, canonical_digest({"effect": "unsafe-test"})),),
    )
    sealed_path = proposal_store.path("user-a", "task-unsafe")
    detached = sealed_path.with_name(f"{sealed_path.stem}.{unsafe_kind}.json")
    if unsafe_kind == "hardlink":
        os.link(sealed_path, detached)
    else:
        sealed_path.rename(detached)
        sealed_path.symlink_to(detached.name)

    eraser = MemoryDocumentEraser(
        source,
        controls,
        revisions,
        cleanup_backends=(SealedProposalEraseBackend(proposal_store),),
    )
    first = eraser.hard_erase(
        tenant_id="default",
        owner_user_id="user-a",
        document_id=plan.document_id,
        expected_source_digest=live.raw_sha256,
        relative_path=plan.relative_path,
    )
    assert first.record.status is DocumentEraseStatus.ERASE_PENDING
    assert first.record.pending_backends == ("derived.sealed_proposals",)
    assert proposal_store.bindings_for_document("user-a", plan.document_id)

    if unsafe_kind == "hardlink":
        detached.unlink()
    else:
        sealed_path.unlink()
        detached.rename(sealed_path)
    completed = eraser.hard_erase(
        tenant_id="default",
        owner_user_id="user-a",
        document_id=plan.document_id,
        expected_source_digest=live.raw_sha256,
        relative_path=plan.relative_path,
    )
    assert completed.record.status is DocumentEraseStatus.ERASED
    assert proposal_store.bindings_for_document("user-a", plan.document_id) == ()
    assert not sealed_path.exists()


def test_sealed_proposal_backend_rejects_cross_tenant_request(tmp_path: Path) -> None:
    backend = SealedProposalEraseBackend(SealedProposalStore(tmp_path, tenant_id="default"))
    with pytest.raises(ValueError, match="configured tenant"):
        backend.erase_document(_request(_DOCUMENT_A, tenant="tenant-b"))
