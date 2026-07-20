from __future__ import annotations

from datetime import datetime, timedelta, timezone

from behavior.core.model.behavior_case import BehaviorCase
from behavior.core.model.observation import Observation
from behavior.projection.behavior_case import BehaviorCaseWriter
from infrastructure.store.model.context.context_type import ContextType
from openApi.sdk.client import MemoryOSClient
from policy.action_policy.decision.request import PredictionRequest
from pre.connect import ConnectMetadata
from pre.session import SessionArchive
from transaction.model.operation_action import OperationAction

NOW = datetime.now(timezone.utc)


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
    client.runtime.transaction.committer.commit("u1", [BehaviorCaseWriter().add_case(case)])


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

    result = client.runtime.session.commit_service.commit_session(_archive("s1"), async_commit=True)
    assert result.done

    obs = _observation()
    policy_uri = f"memoryos://user/u1/action_policies/{obs.scene_key}/turn_on_ac"
    policy = client.runtime.stores.source.read_object(policy_uri)
    assert policy.context_type == ContextType.ACTION_POLICY
    assert policy.metadata["support_anchor_uri"]
    assert policy.metadata["supported_behavior_pattern_uris"]
    assert policy.metadata["auto_execute_allowed"] is False

    patterns = client.runtime.stores.index.search(
        obs.scene_key,
        tenant_id="default",
        filters={"owner_user_id": "u1", "context_type": ContextType.BEHAVIOR_PATTERN.value},
    )
    clusters = client.runtime.stores.index.search(
        obs.scene_key,
        tenant_id="default",
        filters={"owner_user_id": "u1", "context_type": ContextType.BEHAVIOR_CLUSTER.value},
    )
    assert patterns
    assert clusters
    assert policy.metadata["support_anchor_uri"] == patterns[0].metadata["support_anchor_uri"]

    first_ops = client.runtime.session.commit_service.action_policy_planner.plan(_archive("s2"))
    assert any(operation.action == OperationAction.UPDATE and operation.target_uri == policy_uri for operation in first_ops)
    assert not any(operation.action == OperationAction.ADD and operation.target_uri == policy_uri for operation in first_ops)
    client.runtime.session.commit_service.commit_session(_archive("s2"), async_commit=True)

    prediction = client.predict(
        PredictionRequest(
            user_id="u1",
            episode_id="p1",
            observation=obs,
            available_actions=["turn_on_ac", "ask_user", "do_nothing"],
            connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
        )
    )
    assert prediction.candidates[0].policy_uri == policy_uri
    assert "memory_operations" not in prediction.to_dict()
