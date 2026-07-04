from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.prediction.model.prediction_request import PredictionRequest


def test_predictive_context_hot_room_flow_uses_production_entrypoint(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path))
    observation = Observation(
        user_id="u1",
        raw_text="Room temperature is 30C and the user is home.",
        location="home",
        activity="resting",
        signals=["user_present"],
        environment={"temperature": 30},
    )
    anchor_uri = "memoryos://user/u1/memories/anchors/home_comfort"
    resource_uri = "memoryos://resources/devices/living-room-ac"
    skill_uri = "memoryos://skills/smart_home/ac-control"

    anchor = ContextObject(
        uri=anchor_uri,
        context_type=ContextType.MEMORY,
        title="home comfort",
        owner_user_id="u1",
        metadata={"summary": "User comfort memory anchor for hot room behavior."},
    )
    client.context_db.write_object(anchor, content="User comfort memory anchor for hot room behavior.")

    pattern = BehaviorPattern(
        user_id="u1",
        scene_key=observation.scene_key,
        trigger_conditions={"scene_key": observation.scene_key, "context_tags": ["home", "hot_environment"]},
        memory_anchor_uri=anchor_uri,
        case_refs=["case-1", "case-2", "case-3"],
        action_distribution=[{"action": "turn_on_ac", "count": 3}],
        hotness=0.95,
        confidence=0.95,
    )
    pattern_obj = pattern.to_context_object()
    client.context_db.write_object(pattern_obj, content="hot room home user_present turn_on_ac behavior pattern")

    resource = ContextObject(uri=resource_uri, context_type=ContextType.RESOURCE, title="Living room AC", metadata={"available": True})
    skill = ContextObject(uri=skill_uri, context_type=ContextType.SKILL, title="AC control skill", metadata={"executable": True})
    client.context_db.write_object(resource, content="living room AC is available")
    client.context_db.write_object(skill, content="skill can execute turn_on_ac")

    policy = ActionPolicy(
        user_id="u1",
        scene_key=observation.scene_key,
        action="turn_on_ac",
        memory_anchor_uri=anchor_uri,
        q_value=0.95,
        confidence=0.95,
        reward_score=10.0,
        auto_execute_allowed=True,
        required_resource_uris=[resource_uri],
        required_skill_uris=[skill_uri],
        supported_behavior_pattern_uris=[pattern_obj.uri],
    )
    client.context_db.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    for relation_type, target_uri in (
        ("anchored_by", anchor_uri),
        ("supported_by", pattern_obj.uri),
        ("requires_resource", resource_uri),
        ("requires_skill", skill_uri),
    ):
        client.context_db.add_relation(
            ContextRelation(
                source_uri=policy.uri,
                relation_type=relation_type,
                target_uri=target_uri,
                metadata={"owner_user_id": "u1", "tenant_id": "default"},
            )
        )

    request = PredictionRequest(
        user_id="u1",
        episode_id="ep-hot-room-production",
        observation=observation,
        available_actions=["turn_on_ac", "turn_on_fan", "ask_user", "do_nothing"],
        token_budget=2000,
    )
    result = client.process_observation(request, [policy], archive_session=True, async_commit=True)

    assert result.candidates[0].action == "turn_on_ac"
    assert result.decision.mode == "execute"
    assert result.memory_operations == []
    assert result.action_context.packed_context["load_plan"]
    assert "dropped_contexts" in result.action_context.packed_context

    archive_dir = ContextURI.parse("memoryos://user/u1/sessions/history/ep-hot-room-production").to_source_path(tmp_path)
    assert (archive_dir / "observations.jsonl").exists()
    for filename in ("memory_diff.json", "behavior_diff.json", "action_policy_diff.json", "context_diff.json"):
        payload = json.loads((archive_dir / filename).read_text(encoding="utf-8"))
        assert payload["status"] == "committed"
