from __future__ import annotations

import tempfile
import unittest

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.support import SupportAnchor, SupportAnchorKind


class RelationPopulationTest(unittest.TestCase):
    def test_action_policy_behavior_pattern_and_policy_support_relations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            index = InMemoryIndexStore()
            relations = InMemoryRelationStore()
            committer = OperationCommitter(source, index, tmp, relation_store=relations)
            policy = ActionPolicy(
                user_id="u1",
                scene_key="hot",
                action="turn_on_ac",
                support_anchor_uri="memoryos://user/u1/support/behavior/hot",
                required_resource_uris=["memoryos://resources/devices/ac"],
                required_skill_uris=["memoryos://skills/ac-control"],
                supported_behavior_pattern_uris=["memoryos://user/u1/behavior/patterns/hot/p1"],
                constrained_by_support_uris=["memoryos://user/u1/support/action-policy/no-auto"],
            )
            committer.commit("u1", [ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.ADD, target_uri=policy.uri, payload={"context_object": policy.to_context_object().to_dict(), "content": "policy"})])
            relation_types = {
                (r.relation_type, r.target_uri)
                for r in relations.relations_of(
                    policy.uri,
                    tenant_id="default",
                    owner_user_id="u1",
                )
            }
            self.assertIn(("anchored_by", policy.support_anchor_uri), relation_types)
            self.assertIn(("requires_skill", "memoryos://skills/ac-control"), relation_types)
            self.assertIn(("requires_resource", "memoryos://resources/devices/ac"), relation_types)
            pattern = BehaviorPattern(user_id="u1", scene_key="hot", trigger_conditions={}, support_anchor_uri=policy.support_anchor_uri, case_refs=["case-1"], action_distribution=[])
            committer.commit("u1", [ContextOperation(user_id="u1", context_type=ContextType.BEHAVIOR_PATTERN, action=OperationAction.ADD, target_uri=pattern.uri, payload={"context_object": pattern.to_context_object().to_dict(), "content": "pattern"})])
            self.assertTrue(
                any(
                    r.relation_type == "anchored_by"
                    for r in relations.relations_of(
                        pattern.uri,
                        tenant_id="default",
                        owner_user_id="u1",
                    )
                )
            )
            support = SupportAnchor(uri="memoryos://user/u1/support/action-policy/no-auto", user_id="u1", title="no auto", content="以后别自动开空调", anchor_key="no-auto", kind=SupportAnchorKind.ACTION_POLICY, constrains_policy_uris=[policy.uri], policy_rule_type="action_auto_execute", policy_rule_value="forbidden", related_action=policy.action)
            committer.commit("u1", [ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY_SUPPORT, action=OperationAction.ADD, target_uri=support.uri, payload={"context_object": support.to_context_object().to_dict(), "content": support.content})])
            self.assertTrue(
                any(
                    r.relation_type == "constrained_by" and r.target_uri == support.uri
                    for r in relations.relations_of(
                        policy.uri,
                        tenant_id="default",
                        owner_user_id="u1",
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
