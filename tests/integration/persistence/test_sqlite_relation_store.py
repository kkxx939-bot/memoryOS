from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore


class SQLiteRelationStoreTest(unittest.TestCase):
    def test_add_relations_of_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRelationStore(Path(tmp) / "relations.sqlite3")
            relation = ContextRelation(source_uri="policy", relation_type="anchored_by", target_uri="anchor", metadata={"summary": "anchor"})
            store.add_relation(relation)
            self.assertEqual(len(store.relations_of("policy")), 1)
            self.assertEqual(len(store.relations_of("anchor")), 1)
            store.delete_relation("policy", "anchored_by", "anchor")
            self.assertFalse(store.relations_of("policy"))


if __name__ == "__main__":
    unittest.main()
