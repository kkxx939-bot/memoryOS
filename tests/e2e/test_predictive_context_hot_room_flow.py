from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.skill.tool_registry import ToolRegistry


def test_predictive_context_hot_room_flow_uses_production_entrypoint(tmp_path) -> None:
    registry = ToolRegistry()
    registry.register(
        "ac.turn_on",
        lambda args: {"device_id": args["device_id"], "temperature": args["temperature"], "status": "on"},
        input_schema={
            "type": "object",
            "required": ["device_id", "temperature"],
            "properties": {"device_id": {"type": "string"}, "temperature": {"type": "number"}},
        },
    )
    client = MemoryOSClient(str(tmp_path), tool_registry=registry)
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
        metadata={
            "memory_kind": "anchor_memory",
            "admission": {"decision": "accept"},
            "summary": "User comfort memory anchor for hot room behavior.",
        },
    )
    client.context_db.seed_object(anchor, content="User comfort memory anchor for hot room behavior.")

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
    client.context_db.seed_object(pattern_obj, content="hot room home user_present turn_on_ac behavior pattern")

    resource = ContextObject(
        uri=resource_uri,
        context_type=ContextType.RESOURCE,
        title="Living room AC",
        metadata={"available": True, "device_id": "living-room-ac", "temperature": 24},
    )
    skill = ContextObject(
        uri=skill_uri,
        context_type=ContextType.SKILL,
        title="AC control skill",
        metadata={
            "executable": True,
            "tool_name": "ac.turn_on",
            "input_schema": {
                "type": "object",
                "required": ["device_id", "temperature"],
                "properties": {"device_id": {"type": "string"}, "temperature": {"type": "number"}},
            },
            "risk_level": "low",
            "dry_run_supported": True,
        },
    )
    client.context_db.seed_object(resource, content="living room AC is available")
    client.context_db.seed_object(skill, content="skill can execute turn_on_ac")

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
    client.context_db.seed_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
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
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
    )
    result = client.process_observation(request, archive_session=True, async_commit=True)
    prediction = result.prediction_result

    assert prediction.candidates[0].action == "turn_on_ac"
    assert prediction.decision.mode == "execute"
    assert prediction.memory_operations == []
    assert prediction.action_context.packed_context["load_plan"]
    assert "dropped_contexts" in prediction.action_context.packed_context

    archived = client.session_archive_store.read_archive("memoryos://user/u1/sessions/history/ep-hot-room-production")
    assert archived.observations
    action_result = archived.action_results[0]["action_result"]
    assert action_result["status"] == "success"
    assert action_result["tool_name"] == "ac.turn_on"
    outputs = client.session_archive_store.read_async_outputs(archived)
    for output_name in ("memory_diff", "behavior_diff", "action_policy_diff", "context_diff"):
        payload = outputs[output_name]
        assert payload["status"] == "committed"
