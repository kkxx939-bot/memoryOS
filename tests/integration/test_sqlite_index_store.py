from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore


class SQLiteIndexStoreTest(unittest.TestCase):
    def test_upsert_search_delete_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteIndexStore(Path(tmp) / "index.sqlite3")
            u1 = ContextObject(uri="memoryos://user/u1/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature preference", owner_user_id="u1")
            u2 = ContextObject(uri="memoryos://user/u2/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature preference", owner_user_id="u2")
            policy = ContextObject(uri="memoryos://user/u1/action_policies/hot/turn_on_ac", context_type=ContextType.ACTION_POLICY, title="turn on ac policy", owner_user_id="u1")
            store.upsert_index(u1, content="prefers 26 degree")
            store.upsert_index(u2, content="prefers 18 degree")
            store.upsert_index(policy, content="hot room turn_on_ac")
            self.assertEqual(store.search("26", filters={"owner_user_id": "u1"})[0].uri, u1.uri)
            self.assertFalse(store.search("18", filters={"owner_user_id": "u1"}))
            self.assertEqual(store.search("turn_on_ac", filters={"owner_user_id": "u1", "context_type": "action_policy"})[0].uri, policy.uri)
            store.delete_index(u1.uri)
            self.assertFalse(store.search("26", filters={"owner_user_id": "u1"}))

    def test_applicability_filter_treats_query_scopes_as_available_superset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteIndexStore(Path(tmp) / "index.sqlite3")
            obj = ContextObject(
                uri="memoryos://user/u1/memories/preferences/music",
                context_type=ContextType.MEMORY,
                title="quiet hours",
                owner_user_id="u1",
                metadata={
                    "scope": {
                        "applicability": {
                            "all_of": [
                                {"namespace": "memoryos", "kind": "principal", "id": "u1"},
                                {"namespace": "memoryos", "kind": "environment", "id": "home"},
                            ]
                        }
                    }
                },
            )
            store.upsert_index(obj, content="quiet music")
            available = [
                "memoryos:principal:u1",
                "memoryos:environment:home",
                "memoryos:asset:reachy_01",
                "memoryos:location:kitchen",
            ]
            assert store.search("quiet", filters={"applicability_scope_keys": available})
            assert not store.search(
                "quiet",
                filters={"applicability_scope_keys": ["memoryos:principal:u1"]},
            )


if __name__ == "__main__":
    unittest.main()
