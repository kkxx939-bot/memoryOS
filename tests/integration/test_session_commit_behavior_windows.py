from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.update.behavior_case_writer import BehaviorCaseWriter
from memoryos.contextdb.session.planners.behavior_commit_planner import BehaviorCommitPlanner
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.operation_action import OperationAction

NOW = datetime(2026, 7, 4, tzinfo=timezone.utc)


def _archive(location: str = "home", older_than_days: int = 0) -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="s",
        archive_uri="memoryos://user/u1/sessions/history/s",
        observations=[
            {
                "scene_key": "hot_room",
                "raw_text": "hot room",
                "location": location,
                "environment": {"temperature": 30},
                "older_than_days": older_than_days,
            }
        ],
        predictions=[{"observation": {"scene_key": "hot_room"}, "decision": {"action": "turn_on_ac"}, "candidates": [{"action": "turn_on_ac"}]}],
    )


def _seed_case(source, index, root, case_id: str, days_ago: int, location: str = "home") -> None:
    case = BehaviorCase(
        user_id="u1",
        scene_key="hot_room",
        observation={"scene_key": "hot_room", "raw_text": "hot room", "location": location, "environment": {"temperature": 30}},
        case_id=case_id,
        selected_action="turn_on_ac",
        created_at=(NOW - timedelta(days=days_ago)).isoformat(),
    )
    OperationCommitter(source, index, str(root)).commit("u1", [BehaviorCaseWriter().add_case(case)])


def _actions(operations):
    return [operation.action for operation in operations]


def test_one_case_only_writes_case_without_cluster_or_pattern(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    operations = BehaviorCommitPlanner(index, source).plan(_archive())
    assert _actions(operations).count(OperationAction.ADD) == 1
    assert all(operation.context_type.value != "behavior_pattern" for operation in operations)


def test_two_similar_cases_within_three_days_generate_cluster_and_anchor(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    _seed_case(source, index, tmp_path, "h1", 2)

    operations = BehaviorCommitPlanner(index, source).plan(_archive())

    assert any(operation.context_type.value == "behavior_cluster" for operation in operations)
    assert any(operation.context_type.value == "memory" for operation in operations)
    assert all(operation.context_type.value != "behavior_pattern" for operation in operations)


def test_three_similar_cases_within_seven_days_generate_pattern(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    _seed_case(source, index, tmp_path, "h1", 2)
    _seed_case(source, index, tmp_path, "h2", 6)

    operations = BehaviorCommitPlanner(index, source).plan(_archive())

    pattern_ops = [operation for operation in operations if operation.context_type.value == "behavior_pattern"]
    assert pattern_ops
    payload = pattern_ops[0].payload["context_object"]["metadata"]
    assert payload["memory_anchor_uri"]


def test_missing_created_at_history_can_archive_but_not_upgrade_to_cluster_or_pattern(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    case = BehaviorCase(
        user_id="u1",
        scene_key="hot_room",
        observation={"scene_key": "hot_room", "raw_text": "hot room", "location": "home", "environment": {"temperature": 30}},
        case_id="missing_time",
        selected_action="turn_on_ac",
        created_at="",
    )
    op = BehaviorCaseWriter().add_case(case)
    metadata = op.payload["context_object"]["metadata"]
    metadata["created_at"] = ""
    metadata["observed_at"] = ""
    op.payload["content"] = metadata
    OperationCommitter(source, index, str(tmp_path)).commit("u1", [op])

    operations = BehaviorCommitPlanner(index, source).plan(_archive())

    assert all(operation.context_type.value != "behavior_cluster" for operation in operations)
    assert all(operation.context_type.value != "behavior_pattern" for operation in operations)


def test_behavior_planner_matches_feedback_by_request_scene_and_single_fallback() -> None:
    request_archive = SessionArchive(
        user_id="u1",
        session_id="s-request",
        archive_uri="memoryos://user/u1/sessions/history/s-request",
        observations=[{"scene_key": "hot_room", "request_id": "req-1", "location": "home"}],
        predictions=[{"observation": {"scene_key": "hot_room"}, "decision": {"action": "turn_on_ac"}, "candidates": [{"action": "turn_on_ac"}]}],
        feedback=[{"request_id": "req-1", "feedback_type": "execution_success", "reward": 1.0, "executed_action": "turn_on_ac"}],
    )
    scene_archive = SessionArchive(
        user_id="u1",
        session_id="s-scene",
        archive_uri="memoryos://user/u1/sessions/history/s-scene",
        observations=[{"scene_key": "hot_room", "location": "home"}],
        predictions=[{"observation": {"scene_key": "hot_room"}, "decision": {"action": "turn_on_ac"}, "candidates": [{"action": "turn_on_ac"}]}],
        feedback=[{"scene_key": "hot_room", "feedback_type": "execution_failure", "reward": -1.0, "executed_action": "turn_on_ac"}],
    )
    single_archive = SessionArchive(
        user_id="u1",
        session_id="s-single",
        archive_uri="memoryos://user/u1/sessions/history/s-single",
        observations=[{"scene_key": "hot_room", "location": "home"}],
        predictions=[{"observation": {"scene_key": "hot_room"}, "decision": {"action": "turn_on_ac"}, "candidates": [{"action": "turn_on_ac"}]}],
        feedback=[{"feedback_type": "implicit_positive", "reward": 0.5, "executed_action": "turn_on_ac"}],
    )

    planner = BehaviorCommitPlanner()
    request_case = planner.plan(request_archive)[0].payload["context_object"]["metadata"]
    scene_case = planner.plan(scene_archive)[0].payload["context_object"]["metadata"]
    single_case = planner.plan(single_archive)[0].payload["context_object"]["metadata"]

    assert request_case["feedback_type"] == "execution_success"
    assert request_case["reward"] == 1.0
    assert scene_case["feedback_type"] == "execution_failure"
    assert scene_case["reward"] == -1.0
    assert single_case["feedback_type"] == "implicit_positive"
    assert single_case["reward"] == 0.5


def test_same_scene_different_context_tags_do_not_generate_pattern(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    _seed_case(source, index, tmp_path, "h1", 1, location="office")
    _seed_case(source, index, tmp_path, "h2", 2, location="office")

    operations = BehaviorCommitPlanner(index, source).plan(_archive(location="home"))

    assert all(operation.context_type.value != "behavior_pattern" for operation in operations)
    assert all(operation.context_type.value != "behavior_cluster" for operation in operations)


def test_stale_single_short_behavior_archives_instead_of_pattern(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    operations = BehaviorCommitPlanner(index, source).plan(_archive(older_than_days=4))

    assert any(operation.action == OperationAction.ARCHIVE for operation in operations)
    assert all(operation.context_type.value != "behavior_pattern" for operation in operations)


def test_memory_commit_planner_does_not_anchor_same_scene_with_different_context_tags() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s",
        archive_uri="memoryos://user/u1/sessions/history/s",
        observations=[
            {"scene_key": "hot_room", "location": "home", "environment": {"temperature": 30}},
            {"scene_key": "hot_room", "location": "office", "environment": {"temperature": 30}},
        ],
    )

    operations = MemoryCommitPlanner().plan(archive)

    assert operations == []
