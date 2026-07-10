from __future__ import annotations

import json

from memoryos.contextdb.session.planners.action_policy_commit_planner import ActionPolicyCommitPlanner
from memoryos.contextdb.session.planners.behavior_commit_planner import BehaviorCommitPlanner
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryQueueStore
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

    assert operations == []
    assert planner.last_group.archive_only
    assert not planner.last_group.accepted


def test_project_rule_shared_across_agent_views_metadata() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "Project rule: MemoryOS must keep raw tool output out of shared long-term memory."}],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner().plan(archive)[0]
    metadata = operation.payload["context_object"]["metadata"]

    assert operation.payload["source_adapter_id"] == "codex"
    assert metadata["source"]["adapter_id"] == "codex"
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
    operation = MemoryCommitPlanner().plan(archive)[0]

    committer.commit("u1", [operation])
    obj = source.read_object(str(operation.target_uri))

    assert obj.metadata["memory_type"] == "preference"
    assert obj.metadata["admission"]["decision"] == "accept"
    assert "user:u1:preferences" in obj.metadata["retrieval_views"]
    assert obj.metadata["source"]["session_id"] == "s1"


def test_session_commit_uses_schema_memory_commit_planner(tmp_path) -> None:
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
        allow_plan_only=True,
    )

    service.async_commit(archive)
    payload = json.loads((tmp_path / "tenants/default/users/u1/sessions/history/s1/memory_diff.json").read_text(encoding="utf-8"))
    operation = payload["operations"][0]
    metadata = operation["payload"]["context_object"]["metadata"]

    assert operation["payload"]["memory_type"] == "project_rule"
    assert operation["payload"]["admission"]["decision"] == "accept"
    assert "project:memoryos:rules" in operation["payload"]["retrieval_views"]
    assert metadata["memory_type"] == "project_rule"
    assert metadata["admission"]["decision"] == "accept"
    assert "project:memoryos:rules" in metadata["retrieval_views"]


def test_behavior_action_policy_not_modified_by_memory_pipeline() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "I prefer findings first."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )

    memory_ops = MemoryCommitPlanner().plan(archive)
    behavior_ops = BehaviorCommitPlanner().plan(archive)
    action_ops = ActionPolicyCommitPlanner().plan(archive)

    assert memory_ops
    assert behavior_ops == []
    assert action_ops == []
