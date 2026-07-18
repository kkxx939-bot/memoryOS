from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.update.behavior_case_writer import BehaviorCaseWriter
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.prediction.model.prediction_request import PredictionRequest

NOW = datetime.now(timezone.utc)


def _observation() -> Observation:
    return Observation(user_id="u1", raw_text="hot room", location="home", activity="resting", signals=["user_present"], environment={"temperature": 30})


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


def test_behavior_to_action_policy_auto_flow(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path))
    obs = _observation()
    resource_uri = "memoryos://resources/devices/living-room-ac"
    skill_uri = "memoryos://skills/smart-home/ac-control"
    client.context_db.seed_object(
        ContextObject(uri=resource_uri, context_type=ContextType.RESOURCE, title="Living room AC", metadata={"available": True}),
        content="living room AC available",
    )
    client.context_db.seed_object(
        ContextObject(uri=skill_uri, context_type=ContextType.SKILL, title="AC control skill", metadata={"executable": True, "risk_level": "low"}),
        content="turn_on_ac skill available",
    )
    _seed_case(client, "h1", 2)
    _seed_case(client, "h2", 6)

    archive = SessionArchive(
        user_id="u1",
        session_id="auto-flow",
        archive_uri="memoryos://user/u1/sessions/history/auto-flow",
        observations=[{"scene_key": obs.scene_key, **obs.__dict__}],
        predictions=[
            {
                "observation": {"scene_key": obs.scene_key},
                "decision": {"action": "turn_on_ac"},
                "candidates": [{"action": "turn_on_ac", "score": 0.9}],
            }
        ],
        feedback=[{"scene_key": obs.scene_key, "action": "turn_on_ac", "reward": 1.0, "feedback_type": "implicit_positive"}],
        used_contexts=[{"uri": resource_uri}],
        used_skills=[{"uri": skill_uri}],
    )

    result = client.context_db.commit_session(archive, async_commit=True)
    assert result.done

    policy_uri = f"memoryos://user/u1/action_policies/{obs.scene_key}/turn_on_ac"
    policy = client.context_db.read_object(policy_uri)
    assert policy.metadata["support_anchor_uri"]
    assert policy.metadata["supported_behavior_pattern_uris"]
    assert policy.metadata["required_resource_uris"] == [resource_uri]
    assert policy.metadata["required_skill_uris"] == [skill_uri]
    assert client.context_db.search(obs.scene_key, owner_user_id="u1", context_type=ContextType.BEHAVIOR_CLUSTER)
    assert client.context_db.search(obs.scene_key, owner_user_id="u1", context_type=ContextType.BEHAVIOR_PATTERN)
    assert (
        client.context_db.read_object(policy.metadata["support_anchor_uri"]).context_type
        == ContextType.BEHAVIOR_SUPPORT
    )

    prediction = client.predict(
        PredictionRequest(
            user_id="u1",
            episode_id="after-auto-flow",
            observation=obs,
            available_actions=["turn_on_ac", "ask_user", "do_nothing"],
            token_budget=2000,
            connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
        )
    )
    assert prediction.candidates[0].policy_uri == policy_uri
    assert prediction.decision.mode in {"execute", "ask_user"}
    assert prediction.memory_operations == []
    source_uris = set(prediction.action_context.source_uris)
    assert policy.metadata["support_anchor_uri"] in source_uris
    assert policy.metadata["supported_behavior_pattern_uris"][0] in source_uris
    assert resource_uri in source_uris
    assert skill_uri in source_uris

    persisted = client.session_archive_store.read_archive(archive.archive_uri)
    outputs = client.session_archive_store.read_async_outputs(persisted)
    memory_output = outputs["memory_diff"]
    assert memory_output["status"] == "committed"
    assert set(memory_output) >= {
        "edit_proposal_count",
        "memory_document_change_count",
        "effects",
    }
    for output_name in ("behavior_diff", "action_policy_diff", "context_diff"):
        payload = outputs[output_name]
        assert payload["status"] == "committed"
        assert "operation_ids" in payload
