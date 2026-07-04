from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.observation import Observation
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.skill.tool_registry import ToolRegistry


def _seed_client(tmp_path, handler):
    registry = ToolRegistry()
    registry.register(
        "ac_tool",
        handler,
        input_schema={
            "type": "object",
            "required": ["device_id", "temperature"],
            "properties": {"device_id": {"type": "string"}, "temperature": {"type": "number"}},
        },
    )
    client = MemoryOSClient(str(tmp_path), tool_registry=registry)
    observation = Observation(user_id="u1", raw_text="hot room", location="home", environment={"temperature": 30})
    anchor_uri = "memoryos://user/u1/memories/anchors/hot"
    resource_uri = "memoryos://resources/ac"
    skill_uri = "memoryos://skills/ac"
    client.context_db.seed_object(ContextObject(uri=anchor_uri, context_type=ContextType.MEMORY, title="hot anchor", owner_user_id="u1"), content="hot anchor")
    client.context_db.seed_object(ContextObject(uri=resource_uri, context_type=ContextType.RESOURCE, title="AC", metadata={"available": True, "device_id": "ac", "temperature": 24}), content="available")
    client.context_db.seed_object(
        ContextObject(
            uri=skill_uri,
            context_type=ContextType.SKILL,
            title="AC skill",
            metadata={
                "tool_name": "ac_tool",
                "executable": True,
                "input_schema": {
                    "type": "object",
                    "required": ["device_id", "temperature"],
                    "properties": {"device_id": {"type": "string"}, "temperature": {"type": "number"}},
                },
                "risk_level": "low",
                "dry_run_supported": True,
            },
        ),
        content="tool",
    )
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
    )
    client.context_db.seed_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    for relation_type, target_uri in (
        ("anchored_by", anchor_uri),
        ("requires_resource", resource_uri),
        ("requires_skill", skill_uri),
    ):
        client.context_db.add_relation(ContextRelation(source_uri=policy.uri, relation_type=relation_type, target_uri=target_uri, metadata={"owner_user_id": "u1"}))
    request = PredictionRequest(
        user_id="u1",
        episode_id="ep-execute",
        observation=observation,
        available_actions=["turn_on_ac", "ask_user", "do_nothing"],
        token_budget=2000,
    )
    return client, policy, request


def test_execute_success_writes_action_result_and_rewards_policy(tmp_path) -> None:
    client, policy, request = _seed_client(tmp_path, lambda payload: {"device": "on", "action": payload["action"]})

    result = client.process_observation(request, archive_session=True, async_commit=True)

    assert result.decision.mode == "execute"
    archive_dir = ContextURI.parse("memoryos://user/u1/sessions/history/ep-execute").to_source_path(tmp_path)
    action_result = json.loads((archive_dir / "action_results.jsonl").read_text(encoding="utf-8").splitlines()[0])["action_result"]
    assert action_result["status"] == "success"
    assert client.source_store.read_object(policy.uri).metadata["success_count"] == 1


def test_execute_failure_writes_failed_action_result_and_penalizes_policy(tmp_path) -> None:
    def fail(_payload):
        raise RuntimeError("device offline")

    client, policy, request = _seed_client(tmp_path, fail)

    client.process_observation(request, archive_session=True, async_commit=True)

    archive_dir = ContextURI.parse("memoryos://user/u1/sessions/history/ep-execute").to_source_path(tmp_path)
    action_result = json.loads((archive_dir / "action_results.jsonl").read_text(encoding="utf-8").splitlines()[0])["action_result"]
    assert action_result["status"] == "failed"
    assert client.source_store.read_object(policy.uri).metadata["failure_count"] == 1


def test_missing_required_skill_keeps_policy_gate_from_execute(tmp_path) -> None:
    client, policy, request = _seed_client(tmp_path, lambda payload: {"ok": True})
    client.relation_store.delete_relation(policy.uri, "requires_skill", policy.required_skill_uris[0])

    result = client.process_observation(request, archive_session=True, async_commit=True)

    assert result.decision.mode == "ask_user"
    archive_dir = ContextURI.parse("memoryos://user/u1/sessions/history/ep-execute").to_source_path(tmp_path)
    action_result = json.loads((archive_dir / "action_results.jsonl").read_text(encoding="utf-8").splitlines()[0])["action_result"]
    assert action_result["status"] == "skipped"
