from __future__ import annotations

import json

from memoryos.contextdb.session.planners.action_policy_commit_planner import ActionPolicyCommitPlanner
from memoryos.contextdb.session.planners.behavior_commit_planner import BehaviorCommitPlanner
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryQueueStore
from memoryos.memory.extraction import RuleFallbackExtractor
from memoryos.operations.commit.operation_committer import OperationCommitter


def test_codex_tool_result_not_written_as_shared_memory() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        tool_results=[
            {
                "tool_name": "shell",
                "tool_output": "pytest failed\nTraceback (most recent call last):\nAssertionError",
                "changed_files": ["memoryos/contextdb/session/planners/memory_commit_planner.py"],
            }
        ],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )

    planner = MemoryCommitPlanner()
    operations = planner.plan(archive)

    assert operations.operations == ()
    assert "ordinary_tool_result" in operations.context.salience_reasons
    assert "unconfirmed_tool_output" in operations.context.salience_reasons


def test_project_rule_fallback_pending_keeps_shared_view_and_adapter_metadata() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {
                "role": "user",
                "content": "Project rule: MemoryOS must keep raw tool output out of shared long-term memory.",
            }
        ],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )

    operation = next(
        item
        for item in MemoryCommitPlanner(extractor=RuleFallbackExtractor()).plan(archive).operations
        if item.payload["memory_type"] == "project_rule"
    )
    metadata = operation.payload["context_object"]["metadata"]
    proposal_metadata = metadata["proposal"]["metadata"]

    assert operation.payload["canonical_pending_proposal"] is True
    assert operation.payload["admission"] == {
        "decision": "pending",
        "reason": "PENDING_FALLBACK_REQUIRES_SEMANTIC_REVIEW",
    }
    assert operation.payload["schema_version"] == "canonical_pending_proposal_v1"
    assert operation.target_uri and "/memories/pending/" in operation.target_uri
    assert metadata["canonical_kind"] == "pending_proposal"
    assert metadata["lifecycle_state"] == "pending"
    assert proposal_metadata["fallback_pending_only"] is True
    assert proposal_metadata["source_adapter_id"] == "codex"
    assert proposal_metadata["source_session_id"] == "s1"
    assert "project:memoryos:rules" in metadata["retrieval_views"]
    assert "agent:codex:private" not in metadata["retrieval_views"]


def test_committed_memory_context_object_keeps_schema_metadata(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "I prefer findings first during code reviews."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )
    operation = MemoryCommitPlanner(extractor=RuleFallbackExtractor()).plan(archive).operations[0]
    SessionArchiveStore(tmp_path).write_sync_archive(archive)

    diff = committer.commit("u1", [operation])
    obj = source.read_object(str(operation.target_uri))

    assert len(diff.operations) == 1
    assert diff.operations[0].target_uri == operation.target_uri
    assert operation.payload["canonical_pending_proposal"] is True
    assert obj.schema_version == "canonical_pending_proposal_v1"
    assert obj.lifecycle_state.value == "pending"
    assert obj.metadata["canonical_kind"] == "pending_proposal"
    assert obj.metadata["memory_type"] == "preference"
    assert obj.metadata["admission"] == {
        "decision": "pending",
        "reason": "PENDING_FALLBACK_REQUIRES_SEMANTIC_REVIEW",
    }
    assert "user:u1:preferences" in obj.metadata["retrieval_views"]
    proposal_metadata = obj.metadata["proposal"]["metadata"]
    assert proposal_metadata["fallback_pending_only"] is True
    assert proposal_metadata["source_adapter_id"] == "codex"
    assert proposal_metadata["source_session_id"] == "s1"


def test_session_diff_reports_planned_fallback_pending_contract(tmp_path) -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "Project rule: MemoryOS must keep schema metadata in memory diffs."}],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )
    service = SessionCommitService(
        SessionArchiveStore(tmp_path),
        InMemoryQueueStore(),
        memory_planner=MemoryCommitPlanner(extractor=RuleFallbackExtractor()),
        allow_plan_only=True,
    )

    service.async_commit(archive)
    payload = json.loads(
        (tmp_path / "tenants/default/users/u1/sessions/history/s1/memory_diff.json").read_text(encoding="utf-8")
    )
    operation = next(
        item for item in payload["operations"] if item["payload"]["memory_type"] == "project_rule"
    )
    metadata = operation["payload"]["context_object"]["metadata"]
    proposal_metadata = metadata["proposal"]["metadata"]

    assert payload["status"] == "planned"
    assert payload["archive_committed"] is True
    assert payload["canonical_active_operation_count"] == 0
    assert payload["pending_count"] == payload["operation_count"] == len(payload["operations"])
    assert payload["pending_persisted"] is False
    assert all(item["payload"]["canonical_pending_proposal"] is True for item in payload["operations"])
    assert operation["payload"]["memory_type"] == "project_rule"
    assert operation["payload"]["admission"] == {
        "decision": "pending",
        "reason": "PENDING_FALLBACK_REQUIRES_SEMANTIC_REVIEW",
    }
    assert operation["payload"]["schema_version"] == "canonical_pending_proposal_v1"
    assert "project:memoryos:rules" in operation["payload"]["retrieval_views"]
    assert metadata["memory_type"] == "project_rule"
    assert metadata["canonical_kind"] == "pending_proposal"
    assert metadata["lifecycle_state"] == "pending"
    assert metadata["admission"] == operation["payload"]["admission"]
    assert "project:memoryos:rules" in metadata["retrieval_views"]
    assert proposal_metadata["fallback_pending_only"] is True
    assert proposal_metadata["source_adapter_id"] == "codex"
    assert proposal_metadata["source_session_id"] == "s1"


def test_behavior_action_policy_not_modified_by_memory_pipeline() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "I prefer findings first."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )

    memory_ops = MemoryCommitPlanner(extractor=RuleFallbackExtractor()).plan(archive)
    behavior_ops = BehaviorCommitPlanner().plan(archive)
    action_ops = ActionPolicyCommitPlanner().plan(archive)

    assert memory_ops.operations
    assert behavior_ops == []
    assert action_ops == []
