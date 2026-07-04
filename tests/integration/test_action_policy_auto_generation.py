from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.update.behavior_case_writer import BehaviorCaseWriter
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.operations.model.operation_action import OperationAction
from memoryos.prediction.model.prediction_request import PredictionRequest

NOW = datetime(2026, 7, 4, tzinfo=timezone.utc)


def _observation() -> Observation:
    return Observation(user_id="u1", raw_text="hot room", location="home", activity="resting", environment={"temperature": 30})


def _seed_case(client: MemoryOSClient, case_id: str, days_ago: int) -> None:
    obs = _observation()
    case = BehaviorCase(
        user_id="u1",
        scene_key=obs.scene_key,
        observation=obs.__dict__,
        selected_action="turn_on_ac",
        case_id=case_id,
        created_at=(NOW - timedelta(days=days_ago)).isoformat(),
    )
    client.committer.commit("u1", [BehaviorCaseWriter().add_case(case)])


def _archive(session_id: str) -> SessionArchive:
    obs = _observation()
    return SessionArchive(
        user_id="u1",
        session_id=session_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        observations=[{"scene_key": obs.scene_key, **obs.__dict__}],
        predictions=[
            {
                "observation": {"scene_key": obs.scene_key},
                "decision": {"action": "turn_on_ac"},
                "candidates": [{"action": "turn_on_ac", "score": 0.9}],
            }
        ],
        feedback=[{"scene_key": obs.scene_key, "action": "turn_on_ac", "reward": 1.0, "feedback_type": "implicit_positive"}],
    )


def test_behavior_windows_auto_generate_and_update_action_policy(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path))
    _seed_case(client, "h1", 2)
    _seed_case(client, "h2", 6)

    result = client.context_db.commit_session(_archive("s1"), async_commit=True)
    assert result.done

    obs = _observation()
    policy_uri = f"memoryos://user/u1/action_policies/{obs.scene_key}/turn_on_ac"
    policy = client.context_db.read_object(policy_uri)
    assert policy.context_type == ContextType.ACTION_POLICY
    assert policy.metadata["memory_anchor_uri"]
    assert policy.metadata["supported_behavior_pattern_uris"]
    assert policy.metadata["auto_execute_allowed"] is False

    patterns = client.context_db.search(obs.scene_key, owner_user_id="u1", context_type=ContextType.BEHAVIOR_PATTERN)
    clusters = client.context_db.search(obs.scene_key, owner_user_id="u1", context_type=ContextType.BEHAVIOR_CLUSTER)
    assert patterns
    assert clusters

    first_ops = client.session_commit_service.action_policy_planner.plan(_archive("s2"))
    assert any(operation.action == OperationAction.UPDATE and operation.target_uri == policy_uri for operation in first_ops)
    assert not any(operation.action == OperationAction.ADD and operation.target_uri == policy_uri for operation in first_ops)
    client.context_db.commit_session(_archive("s2"), async_commit=True)

    prediction = client.predict(
        PredictionRequest(
            user_id="u1",
            episode_id="p1",
            observation=obs,
            available_actions=["turn_on_ac", "ask_user", "do_nothing"],
        )
    )
    assert prediction.candidates[0].policy_uri == policy_uri
    assert prediction.memory_operations == []
