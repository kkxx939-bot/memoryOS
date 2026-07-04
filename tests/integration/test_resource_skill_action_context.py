from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.resource.resource_importer import ResourceImporter
from memoryos.contextdb.skill.skill_model import Skill
from memoryos.contextdb.skill.skill_registry import SkillRegistry
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder
from memoryos.prediction.pipeline.policy_gate import PolicyGate


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

