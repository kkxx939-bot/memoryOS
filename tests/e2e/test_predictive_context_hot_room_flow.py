from __future__ import annotations

import json

from behavior.core.model.behavior_pattern import BehaviorPattern
from behavior.core.model.observation import Observation
from behavior.projection import behavior_pattern_to_context_object
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from openApi.sdk.client import MemoryOSClient
from policy.action_policy.decision.request import PredictionRequest
from policy.action_policy.execution.tool_registry import ToolRegistry
from policy.action_policy.model.action_policy import ActionPolicy
from pre.connect import ConnectMetadata
from tests.support.persistence import seed_context_object


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
    anchor_uri = "memoryos://user/u1/support/behavior/home_comfort"
    resource_uri = "memoryos://resources/devices/living-room-ac"
    skill_uri = "memoryos://skills/smart_home/ac-control"

    anchor = ContextObject(
        uri=anchor_uri,
        context_type=ContextType.BEHAVIOR_SUPPORT,
        title="home comfort",
        owner_user_id="u1",
        metadata={
            "support_anchor_kind": "behavior",
            "summary": "User comfort support anchor for hot room behavior.",
        },
    )
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        anchor,
        content="User comfort memory anchor for hot room behavior.",
    )

    pattern = BehaviorPattern(
        user_id="u1",
        scene_key=observation.scene_key,
        trigger_conditions={"scene_key": observation.scene_key, "context_tags": ["home", "hot_environment"]},
        support_anchor_uri=anchor_uri,
        case_refs=["case-1", "case-2", "case-3"],
        action_distribution=[{"action": "turn_on_fan", "count": 3}],
        hotness=0.95,
        confidence=0.95,
    )
    pattern_obj = behavior_pattern_to_context_object(pattern)
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        pattern_obj,
        content="hot room home user_present turn_on_fan behavior pattern",
    )

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
    seed_context_object(client.runtime.stores.source, client.runtime.stores.index, resource, content="living room AC is available")
    seed_context_object(client.runtime.stores.source, client.runtime.stores.index, skill, content="skill can execute turn_on_fan")

    policy = ActionPolicy(
        user_id="u1",
        scene_key=observation.scene_key,
        action="turn_on_fan",
        support_anchor_uri=anchor_uri,
        q_value=0.95,
        confidence=0.95,
        reward_score=10.0,
        auto_execute_allowed=True,
        required_resource_uris=[resource_uri],
        required_skill_uris=[skill_uri],
        supported_behavior_pattern_uris=[pattern_obj.uri],
    )
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        policy.to_context_object(),
        content=json.dumps(policy.to_dict()),
    )
    for relation_type, target_uri in (
        ("anchored_by", anchor_uri),
        ("supported_by", pattern_obj.uri),
        ("requires_resource", resource_uri),
        ("requires_skill", skill_uri),
    ):
        client.runtime.stores.relation.add_relation(
            ContextRelation(
                source_uri=policy.uri,
                relation_type=relation_type,
                target_uri=target_uri,
                metadata={"owner_user_id": "u1", "tenant_id": "default"},
            ),
            tenant_id="default",
        )

    request = PredictionRequest(
        user_id="u1",
        episode_id="ep-hot-room-production",
        observation=observation,
        available_actions=["turn_on_fan", "ask_user", "do_nothing"],
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
    )
    result = client.process_observation(request, archive_session=True, async_commit=True)
    prediction = result.prediction_result

    assert result.archive_error is None, result.archive_error
    assert prediction.candidates[0].action == "turn_on_fan"
    assert prediction.decision.mode == "execute"
    assert "memory_operations" not in prediction.to_dict()
    assert prediction.action_context.packed_context["load_plan"]
    assert "dropped_contexts" in prediction.action_context.packed_context

    archived = client.runtime.session.archive_store.read_archive("memoryos://user/u1/sessions/history/ep-hot-room-production")
    assert archived.observations
    action_result = archived.action_results[0]["action_result"]
    assert action_result["status"] == "success"
    assert action_result["tool_name"] == "ac.turn_on"
    outputs = client.runtime.session.archive_store.read_async_outputs(archived)
    for output_name in ("memory_diff", "behavior_diff", "action_policy_diff", "context_diff"):
        payload = outputs[output_name]
        assert payload["status"] == "committed"
