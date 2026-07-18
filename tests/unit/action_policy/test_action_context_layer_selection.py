from __future__ import annotations

import tempfile
import unittest

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder


class ActionContextLayerSelectionTest(unittest.TestCase):
    def test_prefers_l1_then_l0_and_structured_action_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            index = InMemoryIndexStore()
            relations = InMemoryRelationStore()
            anchor_uri = "memoryos://user/u1/support/behavior/hot"
            anchor = ContextObject(
                uri=anchor_uri,
                context_type=ContextType.BEHAVIOR_SUPPORT,
                title="anchor",
                owner_user_id="u1",
                metadata={"support_anchor_kind": "behavior"},
                layers=ContextLayers(l0_uri=f"{anchor_uri}/.abstract.md", l1_uri=f"{anchor_uri}/.overview.md", l2_uri=f"{anchor_uri}/content.md"),
            )
            source.write_object(anchor, content="L2 should not be used")
            assert anchor.layers.l0_uri is not None
            assert anchor.layers.l1_uri is not None
            source.write_content(anchor.layers.l0_uri, "L0 text")
            source.write_content(anchor.layers.l1_uri, "L1 text")
            policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", support_anchor_uri=anchor_uri)
            source.write_object(policy.to_context_object(), content="very long policy content")
            relations.add_relation(
                ContextRelation(
                    source_uri=policy.uri,
                    relation_type="anchored_by",
                    target_uri=anchor_uri,
                    metadata={"owner_user_id": "u1"},
                ),
                tenant_id="default",
            )
            candidate = ActionCandidate(action="turn_on_ac", score=0.9, policy_uri=policy.uri, reason="test")
            context = ActionContextBuilder(index, source_store=source, relation_store=relations).build("u1", [candidate], [policy], token_budget=1000)
            anchor_item = context.packed_context["slices"]["support_anchor"]["items"][0]
            self.assertEqual(anchor_item["content"], "L1 text")
            policy_item = context.packed_context["slices"]["action_policy"]["items"][0]
            self.assertIsInstance(policy_item["content"], dict)
            self.assertNotEqual(policy_item["content"], "very long policy content")

    def test_falls_back_to_summary_when_layers_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            index = InMemoryIndexStore()
            relations = InMemoryRelationStore()
            anchor_uri = "memoryos://user/u1/support/behavior/hot"
            anchor = ContextObject(
                uri=anchor_uri,
                context_type=ContextType.BEHAVIOR_SUPPORT,
                title="anchor",
                owner_user_id="u1",
                metadata={"support_anchor_kind": "behavior", "summary": "summary text"},
            )
            source.write_object(anchor)
            policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", support_anchor_uri=anchor_uri)
            relations.add_relation(
                ContextRelation(
                    source_uri=policy.uri,
                    relation_type="anchored_by",
                    target_uri=anchor_uri,
                    metadata={"owner_user_id": "u1"},
                ),
                tenant_id="default",
            )
            candidate = ActionCandidate(action="turn_on_ac", score=0.9, policy_uri=policy.uri, reason="test")
            context = ActionContextBuilder(index, source_store=source, relation_store=relations).build("u1", [candidate], [policy], token_budget=100)
            self.assertEqual(context.packed_context["slices"]["support_anchor"]["items"][0]["content"], "summary text")


if __name__ == "__main__":
    unittest.main()
