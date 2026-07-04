from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.model.opportunity import OpportunityStats
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.workers.cooling_worker import CoolingWorker


def _stores(tmp_path):
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    return source, index, committer


def _seed_policy_and_pattern(source, index, policy: ActionPolicy, stats: OpportunityStats) -> BehaviorPattern:
    source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    index.upsert_index(policy.to_context_object(), content=f"{policy.scene_key} {policy.action}")
    pattern = BehaviorPattern(
        user_id=policy.user_id,
        scene_key=policy.scene_key,
        trigger_conditions={
            "context_tags": ["home", "hot_environment"],
            "related_policy_uris": [policy.uri],
        },
        memory_anchor_uri=policy.memory_anchor_uri,
        case_refs=["c1", "c2", "c3"],
        action_distribution=[{"action": policy.action, "count": 3}],
        opportunity=stats,
        hotness=0.8,
        confidence=0.8,
    )
    source.write_object(pattern.to_context_object(), content="hot room behavior pattern")
    index.upsert_index(pattern.to_context_object(), content="hot room behavior pattern home")
    return pattern


def test_cooling_worker_no_opportunity_does_not_penalize_action_policy(tmp_path) -> None:
    source, index, committer = _stores(tmp_path)
    policy = ActionPolicy(user_id="u1", scene_key="hot_room", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
    pattern = _seed_policy_and_pattern(source, index, policy, OpportunityStats())

    result = CoolingWorker(source, index, committer).process_behavior_patterns("u1", [Observation(user_id="u1", raw_text="hot room", location="office")])

    assert result["operations"] == []
    assert source.read_object(policy.uri).metadata["failure_count"] == 0
    assert source.read_object(pattern.uri).uri == pattern.uri


def test_cooling_worker_activated_opportunity_refreshes_pattern(tmp_path) -> None:
    source, index, committer = _stores(tmp_path)
    policy = ActionPolicy(user_id="u1", scene_key="hot_room", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
    pattern = _seed_policy_and_pattern(source, index, policy, OpportunityStats(activation_count=2, missed_opportunity_count=1))

    result = CoolingWorker(source, index, committer).process_behavior_patterns(
        "u1", [Observation(user_id="u1", raw_text="hot room", location="home", signals=["action_executed"], environment={"temperature": 30})]
    )

    assert result["operations"][0]["action"] == "refresh_layers"
    assert source.read_object(pattern.uri).uri == pattern.uri


def test_cooling_worker_missed_opportunity_penalizes_policy_once(tmp_path) -> None:
    source, index, committer = _stores(tmp_path)
    policy = ActionPolicy(user_id="u1", scene_key="hot_room", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
    _seed_policy_and_pattern(source, index, policy, OpportunityStats(activation_count=0, missed_opportunity_count=2))

    result = CoolingWorker(source, index, committer).process_behavior_patterns("u1", [Observation(user_id="u1", raw_text="hot room", location="home", environment={"temperature": 30})])

    assert result["operations"][0]["action"] == "penalize"
    updated = source.read_object(policy.uri).metadata
    assert updated["failure_count"] == 1
    assert updated["penalty_score"] > 0


def test_cooling_worker_negative_feedback_penalizes_or_disables(tmp_path) -> None:
    source, index, committer = _stores(tmp_path)
    policy = ActionPolicy(user_id="u1", scene_key="hot_room", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot", auto_execute_allowed=True)
    _seed_policy_and_pattern(source, index, policy, OpportunityStats())

    result = CoolingWorker(source, index, committer).process_behavior_patterns("u1", [Observation(user_id="u1", raw_text="hot room", location="home", signals=["negative_feedback"], environment={"temperature": 30})])

    assert result["operations"][0]["action"] == "penalize"
    assert source.read_object(policy.uri).metadata["failure_count"] == 1

    result = CoolingWorker(source, index, committer).process_behavior_patterns("u1", [Observation(user_id="u1", raw_text="hot room", location="home", signals=["explicit_negative_rule"], environment={"temperature": 30})])
    assert result["operations"][0]["action"] == "disable"
    assert source.read_object(policy.uri).metadata["status"] == ActionPolicyStatus.DISABLED_AUTO_EXECUTE.value
