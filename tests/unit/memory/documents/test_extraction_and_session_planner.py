from __future__ import annotations

import json

import pytest

from memoryos.adapters.persistence.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from memoryos.application.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.documents.model import ABSENT
from memoryos.memory.documents.planner import MemoryDocumentPlanner
from memoryos.memory.documents.review import MemoryEditReviewStatus, MemoryEditReviewStore
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend
from memoryos.memory.extraction.errors import MemoryExtractionSecurityError
from memoryos.memory.schema import MemoryCandidateRegistry


def _archive() -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        task_id="task-1",
        created_at="2026-07-17T10:00:00+08:00",
        archive_digest="a" * 64,
        manifest_digest="b" * 64,
        messages=[
            {
                "id": "e1",
                "role": "user",
                "content": "请记住我更喜欢简洁直接的回答。",
                "occurred_at": "2026-07-17T10:00:00+08:00",
            }
        ],
        metadata={"tenant_id": "default"},
    )


def _response(**extra: object) -> str:
    candidate = {
        "candidate_kind": "preference",
        "title": "Communication style",
        "subject": "responses",
        "body": "The user prefers concise, direct answers.",
        "evidence_refs": ["e1"],
        "field_evidence_refs": {"body": ["e1"]},
        "confidence": 0.95,
        **extra,
    }
    return json.dumps({"candidates": [candidate]})


def test_model_cannot_author_document_or_trusted_scope() -> None:
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(_response(document_id="memdoc_bad")))
    with pytest.raises(MemoryExtractionSecurityError):
        backend.extract(_archive(), MemoryCandidateRegistry().list())


def test_sealed_proposal_replay_does_not_call_model_twice_and_keeps_document_id(tmp_path) -> None:
    provider = FakeMemoryModelProvider(_response())
    document_store = FileSystemMemoryDocumentStore(tmp_path)
    planner = MemoryCommitPlanner(
        MemoryDocumentPlanner(document_store),
        extractor=LLMMemoryExtractorBackend(provider),
        root=tmp_path,
        tenant_id="default",
    )
    first = planner.plan_session(
        _archive(),
        tenant_id="default",
        owner_user_id="u1",
        commit_group_id="group-1",
    )
    second = planner.plan_session(
        _archive(),
        tenant_id="default",
        owner_user_id="u1",
        commit_group_id="group-1",
    )
    assert provider.calls == 1
    assert first.proposal_set_digest == second.proposal_set_digest
    assert first.edit_proposal_count == second.edit_proposal_count == 0
    assert first.candidate_count == second.candidate_count == 1
    assert first.edits[0].plan.document_id == second.edits[0].plan.document_id
    assert first.edits[0].plan.relative_path == "preferences.md"


def test_uncertain_automatic_candidate_is_sealed_for_review_without_live_mutation(tmp_path) -> None:
    provider = FakeMemoryModelProvider(_response(confidence=0.72))
    document_store = FileSystemMemoryDocumentStore(tmp_path)
    planner = MemoryCommitPlanner(
        MemoryDocumentPlanner(document_store),
        extractor=LLMMemoryExtractorBackend(provider),
        root=tmp_path,
        tenant_id="default",
    )

    first = planner.plan_session(
        _archive(),
        tenant_id="default",
        owner_user_id="u1",
        commit_group_id="group-review",
    )
    second = planner.plan_session(
        _archive(),
        tenant_id="default",
        owner_user_id="u1",
        commit_group_id="group-review",
    )

    assert provider.calls == 1
    assert first.edits == second.edits == ()
    assert first.edit_proposal_count == second.edit_proposal_count == 1
    assert first.edit_proposal_ids == second.edit_proposal_ids
    assert first.candidate_count == second.candidate_count == 1
    proposal_id = first.edit_proposal_ids[0]
    record = MemoryEditReviewStore(tmp_path).load("default", "u1", proposal_id)
    assert record is not None
    assert record.status is MemoryEditReviewStatus.PENDING
    assert document_store.read_state("default", "u1", "preferences.md") == ABSENT
