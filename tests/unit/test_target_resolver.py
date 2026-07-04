from __future__ import annotations

import unittest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import InMemoryIndexStore
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.target_resolver import TargetResolver


class TargetResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = InMemoryIndexStore()
        self.resolver = TargetResolver(self.index)

    def test_target_uri_is_resolved(self) -> None:
        op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, target_uri="memoryos://user/u1/memories/profile/a", payload={})
        result = self.resolver.resolve(op, user_id="u1")
        self.assertTrue(result.resolved)
        self.assertEqual(op.status, OperationStatus.RESOLVED)

    def test_update_memory_without_target_resolves_by_query(self) -> None:
        obj = ContextObject(uri="memoryos://user/u1/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature preference", owner_user_id="u1")
        self.index.upsert_index(obj, content="prefers 26 degree room temperature")
        op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, payload={"query": "26 degree room temperature"})
        result = self.resolver.resolve(op, user_id="u1")
        self.assertTrue(result.resolved)
        self.assertEqual(op.target_uri, obj.uri)

    def test_delete_action_policy_without_target_resolves_by_scene_and_action(self) -> None:
        obj = ContextObject(uri="memoryos://user/u1/action_policies/hot_room/turn_on_ac", context_type=ContextType.ACTION_POLICY, title="hot_room turn_on_ac", owner_user_id="u1")
        self.index.upsert_index(obj, content="hot_room turn_on_ac policy")
        op = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.DELETE, payload={"scene_key": "hot_room", "action": "turn_on_ac"})
        result = self.resolver.resolve(op, user_id="u1")
        self.assertTrue(result.resolved)
        self.assertEqual(op.target_uri, obj.uri)

    def test_low_confidence_candidates_go_pending(self) -> None:
        class LowConfidenceIndex:
            def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
                return [IndexHit(uri="memoryos://user/u1/memories/profile/name", score=0.5, context_type="memory", title="name")]

            def upsert_index(self, obj: ContextObject, content: str = "") -> None:
                return None

            def delete_index(self, uri: str) -> None:
                return None

        resolver = TargetResolver(LowConfidenceIndex())
        op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, payload={"query": "name"})
        result = resolver.resolve(op, user_id="u1")
        self.assertFalse(result.resolved)
        self.assertEqual(op.status, OperationStatus.PENDING)
        self.assertIn("target_candidates", op.payload)

    def test_cross_user_namespace_is_not_resolved(self) -> None:
        obj = ContextObject(uri="memoryos://user/u2/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature preference", owner_user_id="u2")
        self.index.upsert_index(obj, content="prefers 26 degree room temperature")
        op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, payload={"query": "26 degree room temperature"})
        result = self.resolver.resolve(op, user_id="u1")
        self.assertFalse(result.resolved)
        self.assertIsNone(op.target_uri)


if __name__ == "__main__":
    unittest.main()
