from __future__ import annotations

from collections.abc import Sequence

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
from memoryos.memory.merge import MemoryMergePlanner
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.operation_action import OperationAction


def _db(tmp_path):
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relation = InMemoryRelationStore()
    committer = OperationCommitter(source, index, str(tmp_path), relation_store=relation)
    return ContextDB(source, index, relation, queue_store=InMemoryQueueStore(), committer=committer), source


class FakeExtractor:
    def __init__(self, candidate: MemoryCandidateDraft) -> None:
        self.candidate = candidate

    def extract(self, archive: SessionArchive, schemas: Sequence[MemoryTypeSchema]) -> list[MemoryCandidateDraft]:  # noqa: ARG002
        return [self.candidate]


def test_merge_key_prevents_duplicate_add(tmp_path) -> None:
    db, source = _db(tmp_path)
    planner = MemoryCommitPlanner(source_store=source)
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "I prefer findings first during code reviews."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )
    first = planner.plan(archive)
    db.commit_operation(first[0])

    second = planner.plan(archive)
    db.commit_operation(second[0])

    objects = [obj for obj in source.list_objects() if obj.metadata.get("memory_type") == "preference"]
    assert len(objects) == 1
    assert second[0].action == OperationAction.UPDATE
    assert second[0].payload["merge_decision"] == "MERGE"
    assert second[0].payload["existing_uri"] == objects[0].uri


def test_field_level_merge_preference() -> None:
    planner = MemoryMergePlanner()

    merged = planner.merge_fields(
        MemoryType.PREFERENCE,
        {"topic": "review", "content": "Findings first.", "evidence": ["m1"]},
        {"topic": "changed", "content": "Be concise.", "evidence": ["m1", "m2"]},
    )

    assert merged["topic"] == "review"
    assert merged["content"] == "Findings first.\nBe concise."
    assert merged["evidence"] == ["m1", "m2"]


def test_conflicting_project_rule_pending_or_supersede(tmp_path) -> None:
    db, source = _db(tmp_path)
    first_candidate = MemoryCandidateDraft(
        memory_type=MemoryType.PROJECT_RULE,
        title="Source audits",
        content="MemoryOS must keep source-only audits.",
        fields={"rule": "source-only audits", "project_id": "memoryOS", "rule_key": "audit_source"},
        confidence=0.9,
        source_role="user",
        source_adapter_id="codex",
        source_session_id="s1",
        merge_key="project_rule:source_audit",
    )
    first_archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )
    first = MemoryCommitPlanner(source_store=source, extractor=FakeExtractor(first_candidate)).plan(first_archive)[0]
    db.commit_operation(first)
    second_candidate = MemoryCandidateDraft(
        memory_type=MemoryType.PROJECT_RULE,
        title="Docs audits",
        content="MemoryOS must allow docs during audits.",
        fields={"rule": "allow docs during audits", "project_id": "memoryOS", "rule_key": "audit_source"},
        confidence=0.9,
        source_role="user",
        source_adapter_id="codex",
        source_session_id="s2",
        merge_key="project_rule:source_audit",
    )
    second_archive = SessionArchive(
        user_id="u1",
        session_id="s2",
        archive_uri="memoryos://user/u1/sessions/history/s2",
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )
    second = MemoryCommitPlanner(source_store=source, extractor=FakeExtractor(second_candidate)).plan(second_archive)[0]

    assert second.action == OperationAction.SUPERSEDE
    assert second.payload["merge_decision"] == "SUPERSEDE"


def test_no_existing_memory_normal_add(tmp_path) -> None:
    _db_obj, source = _db(tmp_path)
    planner = MemoryCommitPlanner(source_store=source)
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "I prefer short final reports."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )

    operation = planner.plan(archive)[0]

    assert operation.action == OperationAction.ADD
    assert operation.payload["merge_decision"] == "ADD"
