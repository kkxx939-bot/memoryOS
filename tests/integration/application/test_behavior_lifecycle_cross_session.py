from __future__ import annotations

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.planners.behavior_commit_planner import BehaviorCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.operation_action import OperationAction


def _archive(user_id: str, session_id: str, scene_key: str, older_than_days: int = 0) -> SessionArchive:
    return SessionArchive(
        user_id=user_id,
        session_id=session_id,
        archive_uri=f"memoryos://user/{user_id}/sessions/history/{session_id}",
        observations=[{"scene_key": scene_key, "temperature": 30, "older_than_days": older_than_days}],
        predictions=[{"observation": {"scene_key": scene_key}, "candidates": [{"action": "turn_on_ac"}], "decision": {"action": "turn_on_ac"}}],
    )


def test_behavior_lifecycle_aggregates_across_history(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(source, index, tmp_path, relation_store=relations)
    planner = BehaviorCommitPlanner(index, source)

    first_ops = planner.plan(_archive("u1", "s1", "hot_room"))
    committer.commit("u1", first_ops)
    assert not [op for op in first_ops if op.context_type == ContextType.BEHAVIOR_CLUSTER]

    second_ops = planner.plan(_archive("u1", "s2", "hot_room"))
    committer.commit("u1", second_ops)
    assert any(op.context_type == ContextType.BEHAVIOR_CLUSTER for op in second_ops)
    assert any(op.context_type == ContextType.MEMORY for op in second_ops)

    third_ops = planner.plan(_archive("u1", "s3", "hot_room"))
    committer.commit("u1", third_ops)
    pattern_ops = [op for op in third_ops if op.context_type == ContextType.BEHAVIOR_PATTERN]
    assert pattern_ops
    assert pattern_ops[0].target_uri is not None
    pattern_obj = source.read_object(pattern_ops[0].target_uri)
    assert pattern_obj.metadata["memory_anchor_uri"]

    stale_ops = planner.plan(_archive("u1", "old", "one_off", older_than_days=4))
    assert any(op.action == OperationAction.ARCHIVE for op in stale_ops)

    diff_files = list((tmp_path / "system" / "diffs").glob("*.json"))
    assert diff_files
