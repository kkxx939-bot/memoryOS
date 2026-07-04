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


if __name__ == "__main__":
    unittest.main()
