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
            store.add_relation(relation, tenant_id="tenant-a")
            self.assertEqual(len(store.relations_of("policy", tenant_id="tenant-a")), 1)
            self.assertEqual(len(store.relations_of("anchor", tenant_id="tenant-a")), 1)
            store.delete_relation("policy", "anchored_by", "anchor", tenant_id="tenant-a")
            self.assertFalse(store.relations_of("policy", tenant_id="tenant-a"))

    def test_memory_document_relation_cleanup_is_tenant_and_owner_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRelationStore(Path(tmp) / "relations.sqlite3")
            target = "memoryos://user/alice/memory/documents/memdoc_target"
            for tenant, owner, source in (
                ("tenant-a", "alice", "source-a"),
                ("tenant-a", "bob", "source-b"),
                ("tenant-b", "alice", "source-c"),
            ):
                store.add_relation(
                    ContextRelation(
                        source_uri=source,
                        relation_type="links_to",
                        target_uri=target,
                        metadata={"owner_user_id": owner},
                    ),
                    tenant_id=tenant,
                )

            removed = store.delete_memory_document_relations(
                target,
                tenant_id="tenant-a",
                owner_user_id="alice",
                limit=100,
            )

            self.assertEqual(removed, 1)
            self.assertEqual(
                {item.source_uri for item in store.relations_of(target, tenant_id="tenant-a")},
                {"source-b"},
            )
            self.assertEqual(
                {item.source_uri for item in store.relations_of(target, tenant_id="tenant-b")},
                {"source-c"},
            )


if __name__ == "__main__":
    unittest.main()
