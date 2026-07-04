from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder


class ActionContextBuilderRelationFirstTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = FileSystemSourceStore(self.root)
        self.index = InMemoryIndexStore()
        self.relations = InMemoryRelationStore()
        self.policy = ActionPolicy(
            user_id="u1",
            scene_key="hot_room",
            action="turn_on_ac",
            memory_anchor_uri="memoryos://user/u1/memories/anchors/home_comfort",
        )
        self.source.write_object(self.policy.to_context_object(), content="action policy")
        self._write("memoryos://user/u1/memories/anchors/home_comfort", ContextType.MEMORY, "Home comfort anchor", "anchor text")
        self._write("memoryos://user/u1/memories/policies/no_auto", ContextType.MEMORY, "No auto AC", "policy memory text")
        self._write("memoryos://user/u1/behavior/patterns/hot_room/p1", ContextType.BEHAVIOR_PATTERN, "Hot room pattern", "pattern text")
        self._write("memoryos://resources/devices/ac-living-room", ContextType.RESOURCE, "Living room AC", "resource text", owner=None)
        self._write("memoryos://skills/smart_home/ac-control", ContextType.SKILL, "AC control", "skill text", owner=None)
        for relation_type, target in (
            ("anchored_by", "memoryos://user/u1/memories/anchors/home_comfort"),
            ("constrained_by", "memoryos://user/u1/memories/policies/no_auto"),
            ("supported_by", "memoryos://user/u1/behavior/patterns/hot_room/p1"),
            ("requires_resource", "memoryos://resources/devices/ac-living-room"),
            ("requires_skill", "memoryos://skills/smart_home/ac-control"),
        ):
            self.relations.add_relation(ContextRelation(source_uri=self.policy.uri, relation_type=relation_type, target_uri=target))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, uri: str, context_type: ContextType, title: str, content: str, owner: str | None = "u1") -> None:
        obj = ContextObject(uri=uri, context_type=context_type, title=title, owner_user_id=owner)
        self.source.write_object(obj, content=content)

    def build(self, budget: int = 2000):
        builder = ActionContextBuilder(self.index, source_store=self.source, relation_store=self.relations)
        candidate = ActionCandidate(action=self.policy.action, score=0.9, policy_uri=self.policy.uri, reason="test")
        return builder.build("u1", [candidate], [self.policy], token_budget=budget)

    def test_relation_first_fetches_anchor_policy_resource_and_skill(self) -> None:
        context = self.build()
        slices = context.packed_context["slices"]
        self.assertTrue(any(item["uri"].endswith("/home_comfort") for item in slices["memory_anchor"]["items"]))
        self.assertTrue(any(item["uri"].endswith("/no_auto") for item in slices["memory_rules"]["items"]))
        self.assertTrue(any(item["uri"].startswith("memoryos://skills/") for item in slices["skill"]["items"]))
        self.assertTrue(any(item["uri"].startswith("memoryos://resources/") for item in slices["resource"]["items"]))
        self.assertTrue(any(item["context_type"] == ContextType.BEHAVIOR_PATTERN.value for item in slices["behavior_pattern"]["items"]))

    def test_fallback_search_when_relation_missing(self) -> None:
        empty_relations = InMemoryRelationStore()
        anchor = ContextObject(uri="memoryos://user/u1/memories/anchors/fallback", context_type=ContextType.MEMORY, title="fallback anchor", owner_user_id="u1")
        self.index.upsert_index(anchor, content=self.policy.memory_anchor_uri)
        builder = ActionContextBuilder(self.index, source_store=self.source, relation_store=empty_relations)
        candidate = ActionCandidate(action=self.policy.action, score=0.9, policy_uri=self.policy.uri, reason="test")
        context = builder.build("u1", [candidate], [self.policy], token_budget=2000)
        self.assertTrue(context.packed_context["slices"]["memory_anchor"]["items"])

    def test_token_budget_limits_context(self) -> None:
        context = self.build(budget=160)
        self.assertLessEqual(context.packed_context["used"], 160)

    def test_cross_user_relation_target_is_not_read(self) -> None:
        self._write("memoryos://user/u2/memories/policies/private", ContextType.MEMORY, "private", "private text", owner="u2")
        self.relations.add_relation(ContextRelation(source_uri=self.policy.uri, relation_type="constrained_by", target_uri="memoryos://user/u2/memories/policies/private"))
        context = self.build()
        uris = [item["uri"] for item in context.packed_context["slices"]["memory_rules"]["items"]]
        self.assertNotIn("memoryos://user/u2/memories/policies/private", uris)


if __name__ == "__main__":
    unittest.main()
