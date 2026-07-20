from __future__ import annotations

import json

from behavior.core.model.behavior_pattern import BehaviorPattern
from behavior.core.support import BehaviorSupportAnchor
from behavior.projection import behavior_pattern_to_context_object, behavior_support_to_context_object
from infrastructure.context.operation_effects import InfrastructureContextOperationEffects
from infrastructure.store.model.context import ContextObject, ContextType
from openApi.sdk.client import MemoryOSClient
from policy.action_policy.decision.context_builder import ActionContextBuilder
from policy.action_policy.decision.gate import PolicyGate
from policy.action_policy.decision.request import PredictionRequest
from policy.action_policy.execution.tool_registry import ToolRegistry
from policy.action_policy.integration.commit_registration import build_action_policy_transaction_extensions
from policy.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from pre.connect import ConnectMetadata
from tests.support.persistence import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
    seed_context_object,
)
from tests.support.transaction import build_test_operation_committer as OperationCommitter
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def test_resource_and_skill_required_by_action_policy_gate_execution(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(
        source,
        index,
        tmp_path,
        relation_store=relations,
        context_effects=InfrastructureContextOperationEffects(),
        domain_extensions=build_action_policy_transaction_extensions(),
    )
    resource_uri = "memoryos://resources/devices/ac-living-room"
    skill_uri = "memoryos://skills/smart_home/ac-control"

    seed_context_object(
        source,
        index,
        ContextObject(
            uri=resource_uri,
            context_type=ContextType.RESOURCE,
            title="AC",
            metadata={"resource_type": "device"},
        ),
        content="available",
    )
    seed_context_object(
        source,
        index,
        ContextObject(
            uri=skill_uri,
            context_type=ContextType.SKILL,
            title="AC control",
            metadata={"tool_name": "ac.set", "risk_level": "low", "executable": True},
        ),
        content="executable",
    )
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_fan",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
        auto_execute_allowed=True,
        q_value=0.9,
        confidence=0.9,
        required_resource_uris=[resource_uri],
        required_skill_uris=[skill_uri],
    )
    source.write_object(
        behavior_support_to_context_object(
            BehaviorSupportAnchor(
                uri=policy.support_anchor_uri,
                user_id="u1",
                title="hot anchor",
                content="verified hot-room behavior anchor",
                anchor_key="hot",
            )
        ),
        content="verified hot-room behavior anchor",
    )
    committer.commit(
        "u1",
        [
            ContextOperation(
                user_id="u1",
                context_type=policy.to_context_object().context_type,
                action=OperationAction.ADD,
                target_uri=policy.uri,
                payload={"context_object": policy.to_context_object().to_dict(), "content": "policy"},
            )
        ],
    )

    candidate = ActionCandidate(action=policy.action, score=0.92, policy_uri=policy.uri, reason="test")
    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1", [candidate], [policy]
    )
    assert context.packed_context["slices"]["resource"]["items"]
    assert context.packed_context["slices"]["skill"]["items"]
    assert PolicyGate().evaluate(candidate, context, policy, 0.92).mode == "execute"

    source.soft_delete(skill_uri, "removed")
    context_without_skill = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1", [candidate], [policy]
    )
    assert PolicyGate().evaluate(candidate, context_without_skill, policy, 0.92).reason == "required skill unavailable"


def test_direct_request_resource_and_skill_are_archived_and_learned_by_action_policy(tmp_path) -> None:
    calls: list[dict] = []
    registry = ToolRegistry()

    def ac_tool(args: dict) -> dict:
        calls.append(args)
        return {"ok": True}

    registry.register("ac_tool", ac_tool)
    client = MemoryOSClient(str(tmp_path), tool_registry=registry)
    scene_key = "hot_room"
    resource_uri = "memoryos://resources/ac"
    skill_uri = "memoryos://skills/ac"
    anchor_uri = f"memoryos://user/u1/support/behavior/{scene_key}"
    pattern = BehaviorPattern(
        user_id="u1",
        scene_key=scene_key,
        trigger_conditions={"scene_key": scene_key},
        support_anchor_uri=anchor_uri,
        case_refs=["case-1"],
        action_distribution=[{"action": "turn_on_fan", "count": 1}],
    )
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        behavior_pattern_to_context_object(pattern),
        content="hot_room turn_on_fan behavior",
    )
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        behavior_support_to_context_object(
            BehaviorSupportAnchor(
                uri=anchor_uri,
                user_id="u1",
                title="hot room anchor",
                content="verified hot-room behavior anchor",
                anchor_key=scene_key,
            )
        ),
        content="verified hot-room behavior anchor",
    )
    policy = ActionPolicy(
        user_id="u1",
        scene_key=scene_key,
        action="turn_on_fan",
        support_anchor_uri=anchor_uri,
        auto_execute_allowed=True,
        q_value=0.95,
        confidence=0.95,
        reward_score=10.0,
    )
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        policy.to_context_object(),
        content=json.dumps(policy.to_dict()),
    )
    request = PredictionRequest(
        user_id="u1",
        episode_id="direct-resource-skill",
        observation={"raw_text": "room is hot", "location": "home", "scene_key": scene_key},
        available_actions=["turn_on_fan", "ask_user", "do_nothing"],
        request_id="req-direct",
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
        resources=[
            {
                "uri": resource_uri,
                "title": "AC",
                "metadata": {
                    "tool_name": "ac_tool",
                    "supported_actions": ["turn_on_fan"],
                    "device_id": "ac",
                },
            }
        ],
        skills=[
            {
                "uri": skill_uri,
                "title": "AC Skill",
                "metadata": {
                    "tool_name": "ac_tool",
                    "executable": True,
                    "supported_actions": ["turn_on_fan"],
                },
            }
        ],
    )

    result = client.process_observation(request, policies=[policy], async_commit=True)

    assert result.action_result is not None
    assert result.action_result.status == "success"
    assert result.action_result.resource_uris == [resource_uri]
    assert result.action_result.skill_uris == [skill_uri]
    assert calls
    action_context = result.prediction_result.action_context.packed_context["slices"]
    assert [item["uri"] for item in action_context["resource"]["items"]] == [resource_uri]
    assert [item["uri"] for item in action_context["skill"]["items"]] == [skill_uri]

    assert result.archive_uri is not None
    archived = client.runtime.session.archive_store.read_archive(result.archive_uri)
    assert resource_uri in {item["uri"] for item in archived.used_contexts}
    assert skill_uri in {item["uri"] for item in archived.used_skills}

    learned_policy = client.runtime.stores.source.read_object(f"memoryos://user/u1/action_policies/{scene_key}/turn_on_fan")
    assert learned_policy.metadata["required_resource_uris"] == [resource_uri]
    assert learned_policy.metadata["required_skill_uris"] == [skill_uri]


def test_persistent_context_skill_is_executable_when_explicitly_enabled(tmp_path) -> None:
    calls: list[dict] = []
    registry = ToolRegistry()

    def ac_tool(args: dict) -> dict:
        calls.append(args)
        return {"ok": True}

    registry.register("ac_tool", ac_tool)
    client = MemoryOSClient(str(tmp_path), tool_registry=registry)
    resource_uri = "memoryos://resources/devices/ac-living-room"
    skill_uri = "memoryos://skills/smart_home/ac-control"
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        ContextObject(
            uri=resource_uri,
            context_type=ContextType.RESOURCE,
            title="AC",
            metadata={
                "resource_type": "device",
                "tool_name": "ac_tool",
                "supported_actions": ["turn_on_fan"],
                "device_id": "ac",
            },
        ),
        content="AC resource available",
    )
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        ContextObject(
            uri=skill_uri,
            context_type=ContextType.SKILL,
            title="AC control",
            metadata={"tool_name": "ac_tool", "risk_level": "low", "executable": True},
        ),
        content="turn_on_fan skill available",
    )
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_fan",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
        auto_execute_allowed=True,
        q_value=0.95,
        confidence=0.95,
        reward_score=10.0,
        required_resource_uris=[resource_uri],
        required_skill_uris=[skill_uri],
    )
    seed_context_object(
        client.runtime.stores.source,
        client.runtime.stores.index,
        behavior_support_to_context_object(
            BehaviorSupportAnchor(
                uri=policy.support_anchor_uri,
                user_id="u1",
                title="hot anchor",
                content="verified hot-room behavior anchor",
                anchor_key="hot",
            )
        ),
        content="verified hot-room behavior anchor",
    )
    client.runtime.transaction.committer.commit(
        "u1",
        [
            ContextOperation(
                user_id="u1",
                context_type=policy.to_context_object().context_type,
                action=OperationAction.ADD,
                target_uri=policy.uri,
                payload={
                    "context_object": policy.to_context_object().to_dict(),
                    "content": json.dumps(policy.to_dict()),
                },
            )
        ],
    )

    result = client.process_observation(
        PredictionRequest(
            user_id="u1",
            episode_id="registered-skill",
            observation={"scene_key": "hot", "raw_text": "room is hot", "location": "home"},
            available_actions=["turn_on_fan", "ask_user", "do_nothing"],
            connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
        ),
        async_commit=False,
    )

    assert result.action_result is not None
    assert result.action_result.status == "success"
    assert result.action_result.skill_uris == [skill_uri]
    assert calls
