from __future__ import annotations

from collections.abc import Sequence

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
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
    SessionArchiveStore(tmp_path).write_sync_archive(archive)
    first = planner.plan(archive)
    db.commit_operations(list(first.operations))

    second = planner.plan(archive)

    objects = [
        obj
        for obj in source.list_objects()
        if obj.metadata.get("memory_type") == "preference" and obj.metadata.get("canonical_kind") == "claim"
    ]
    assert len(objects) == 1
    assert second.operations == ()


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
        source_message_ids=["m1"],
        merge_key="project_rule:source_audit",
    )
    first_archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": "Rule topic audit_source: MemoryOS must keep source-only audits.",
            }
        ],
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )
    SessionArchiveStore(tmp_path).write_sync_archive(first_archive)
    first = MemoryCommitPlanner(source_store=source, extractor=FakeExtractor(first_candidate)).plan(first_archive)
    db.commit_operations(list(first.operations))
    second_candidate = MemoryCandidateDraft(
        memory_type=MemoryType.PROJECT_RULE,
        title="Docs audits",
        content="MemoryOS must allow docs during audits.",
        fields={"rule": "allow docs during audits", "project_id": "memoryOS", "rule_key": "audit_source"},
        confidence=0.9,
        source_role="user",
        source_adapter_id="codex",
        source_session_id="s2",
        source_message_ids=["m1"],
        merge_key="project_rule:source_audit",
    )
    second_archive = SessionArchive(
        user_id="u1",
        session_id="s2",
        archive_uri="memoryos://user/u1/sessions/history/s2",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": "Rule topic audit_source: MemoryOS must allow docs during audits.",
            }
        ],
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )
    SessionArchiveStore(tmp_path).write_sync_archive(second_archive)
    second = MemoryCommitPlanner(source_store=source, extractor=FakeExtractor(second_candidate)).plan(second_archive)
    assert len(second.operations) == 3
    assert all(operation.payload.get("canonical_memory") is True for operation in second.operations)
    db.commit_operations(list(second.operations))
    claims = [obj for obj in source.list_objects() if obj.metadata.get("canonical_kind") == "claim"]
    assert {obj.metadata["canonical_value"]: obj.metadata["state"] for obj in claims} == {
        "source-only-audits": "SUPERSEDED",
        "allow-docs-during-audits": "ACTIVE",
    }


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
    SessionArchiveStore(tmp_path).write_sync_archive(archive)

    operation = planner.plan(archive).operations[0]

    assert operation.action == OperationAction.ADD
    assert operation.payload["merge_decision"] == "ADD"
