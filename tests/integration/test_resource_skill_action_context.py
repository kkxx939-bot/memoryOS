from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.resource.resource_importer import ResourceImporter
from memoryos.contextdb.skill.skill_model import Skill
from memoryos.contextdb.skill.skill_registry import SkillRegistry
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder
from memoryos.prediction.pipeline.policy_gate import PolicyGate
from memoryos.skill.tool_registry import ToolRegistry


def test_resource_and_skill_required_by_action_policy_gate_execution(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(source, index, tmp_path, relation_store=relations)
    resource_uri = "memoryos://resources/devices/ac-living-room"
    skill_uri = "memoryos://skills/smart_home/ac-control"

    ResourceImporter(source, index).import_text(resource_uri, "AC", "device", "available")
    SkillRegistry(source, index).register(Skill(uri=skill_uri, title="AC control", tool_name="ac.set", risk_level="low"), content="executable")
    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot", auto_execute_allowed=True, q_value=0.9, confidence=0.9, required_resource_uris=[resource_uri], required_skill_uris=[skill_uri])
    committer.commit("u1", [ContextOperation(user_id="u1", context_type=policy.to_context_object().context_type, action=OperationAction.ADD, target_uri=policy.uri, payload={"context_object": policy.to_context_object().to_dict(), "content": "policy"})])

    candidate = ActionCandidate(action=policy.action, score=0.92, policy_uri=policy.uri, reason="test")
    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build("u1", [candidate], [policy], token_budget=1000)
    assert context.packed_context["slices"]["resource"]["items"]
    assert context.packed_context["slices"]["skill"]["items"]
    assert PolicyGate().evaluate(candidate, context, policy, 0.92).mode == "execute"

    source.soft_delete(skill_uri, "removed")
    context_without_skill = ActionContextBuilder(index, source_store=source, relation_store=relations).build("u1", [candidate], [policy], token_budget=1000)
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
    anchor_uri = f"memoryos://user/u1/memories/anchors/{scene_key}"
    pattern = BehaviorPattern(
        user_id="u1",
        scene_key=scene_key,
        trigger_conditions={"scene_key": scene_key},
        memory_anchor_uri=anchor_uri,
        case_refs=["case-1"],
        action_distribution=[{"action": "turn_on_ac", "count": 1}],
    )
    client.context_db.seed_object(pattern.to_context_object(), content="hot_room turn_on_ac behavior")
    policy = ActionPolicy(
        user_id="u1",
        scene_key=scene_key,
        action="turn_on_ac",
        memory_anchor_uri=anchor_uri,
        auto_execute_allowed=True,
        q_value=0.95,
        confidence=0.95,
    )
    client.context_db.seed_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    request = PredictionRequest(
        user_id="u1",
        episode_id="direct-resource-skill",
        observation={"raw_text": "room is hot", "location": "home", "scene_key": scene_key},
        available_actions=["turn_on_ac", "ask_user", "do_nothing"],
        request_id="req-direct",
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
        resources=[
            {
                "uri": resource_uri,
                "title": "AC",
                "metadata": {
                    "tool_name": "ac_tool",
                    "supported_actions": ["turn_on_ac"],
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
                    "supported_actions": ["turn_on_ac"],
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
    archive_dir = ContextURI.parse(result.archive_uri).to_source_path(tmp_path)
    used_contexts = json.loads((archive_dir / "used_contexts.json").read_text(encoding="utf-8"))
    used_skills = json.loads((archive_dir / "used_skills.json").read_text(encoding="utf-8"))
    assert resource_uri in {item["uri"] for item in used_contexts}
    assert skill_uri in {item["uri"] for item in used_skills}

    learned_policy = client.context_db.read_object(f"memoryos://user/u1/action_policies/{scene_key}/turn_on_ac")
    assert learned_policy.metadata["required_resource_uris"] == [resource_uri]
    assert learned_policy.metadata["required_skill_uris"] == [skill_uri]


def test_registered_persistent_skill_is_executable_by_default(tmp_path) -> None:
    calls: list[dict] = []
    registry = ToolRegistry()

    def ac_tool(args: dict) -> dict:
        calls.append(args)
        return {"ok": True}

    registry.register("ac_tool", ac_tool)
    client = MemoryOSClient(str(tmp_path), tool_registry=registry)
    resource_uri = "memoryos://resources/devices/ac-living-room"
    skill_uri = "memoryos://skills/smart_home/ac-control"
    ResourceImporter(client.source_store, client.index_store).import_text(
        resource_uri,
        "AC",
        "device",
        json.dumps({"tool_name": "ac_tool", "supported_actions": ["turn_on_ac"], "device_id": "ac"}),
    )
    SkillRegistry(client.source_store, client.index_store).register(
        Skill(uri=skill_uri, title="AC control", tool_name="ac_tool"),
        content="turn_on_ac skill available",
    )
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_ac",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        auto_execute_allowed=True,
        q_value=0.95,
        confidence=0.95,
        required_resource_uris=[resource_uri],
        required_skill_uris=[skill_uri],
    )
    client.committer.commit(
        "u1",
        [
            ContextOperation(
                user_id="u1",
                context_type=policy.to_context_object().context_type,
                action=OperationAction.ADD,
                target_uri=policy.uri,
                payload={"context_object": policy.to_context_object().to_dict(), "content": json.dumps(policy.to_dict())},
            )
        ],
    )

    result = client.process_observation(
        PredictionRequest(
            user_id="u1",
            episode_id="registered-skill",
            observation={"scene_key": "hot", "raw_text": "room is hot", "location": "home"},
            available_actions=["turn_on_ac", "ask_user", "do_nothing"],
            connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
        ),
        async_commit=False,
    )

    assert result.action_result is not None
    assert result.action_result.status == "success"
    assert result.action_result.skill_uris == [skill_uri]
    assert calls


def test_registered_skill_preserves_explicit_non_executable_metadata() -> None:
    skill = Skill(
        uri="memoryos://skills/smart_home/ac-control-disabled",
        title="AC control disabled",
        tool_name="ac_tool",
        metadata={"executable": False},
    )

    assert skill.to_context_object().metadata["executable"] is False
