from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.contextdb.layers import ContextPacker
from memoryos.contextdb.model import ContextObject, ContextType, ContextURI
from memoryos.contextdb.store import FileSystemSourceStore, IndexConsistencyService, InMemoryIndexStore
from memoryos.core.errors import InvalidContextURI


class ContextDBCoreTest(unittest.TestCase):
    def test_context_uri_maps_user_namespace_inside_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uri = ContextURI.parse("memoryos://user/gulf/memories/anchors/home-comfort")
            path = uri.to_source_path(Path(tmp))
            self.assertTrue(str(path).startswith(str(Path(tmp).resolve())))
            self.assertIn("tenants/default/users/gulf/memories/anchors/home-comfort", str(path))

    def test_context_uri_rejects_traversal_and_unknown_authority(self) -> None:
        with self.assertRaises(InvalidContextURI):
            ContextURI.parse("memoryos://user/gulf/../secret")
        with self.assertRaises(InvalidContextURI):
            ContextURI.parse("memoryos://other/x")

    def test_source_store_is_fact_source_and_index_is_derived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemSourceStore(tmp)
            index = InMemoryIndexStore()
            obj = ContextObject(
                uri="memoryos://user/gulf/memories/anchors/home-comfort",
                context_type=ContextType.MEMORY,
                title="Home comfort anchor",
                owner_user_id="gulf",
            )
            store.write_object(obj, content="User has a home comfort behavior theme.")
            loaded = store.read_object(obj.uri)
            self.assertEqual(loaded.title, "Home comfort anchor")

            self.assertEqual(index.search("comfort", filters={"owner_user_id": "gulf"}), [])
            index.upsert_index(loaded, store.read_content(obj.uri))
            self.assertEqual(index.search("comfort", filters={"owner_user_id": "gulf"})[0].uri, obj.uri)

    def test_index_consistency_can_rebuild_from_source_and_respects_user_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemSourceStore(tmp)
            index = InMemoryIndexStore()
            gulf_obj = ContextObject(
                uri="memoryos://user/gulf/memories/anchors/home-comfort",
                context_type=ContextType.MEMORY,
                title="Gulf comfort",
                owner_user_id="gulf",
            )
            other_obj = ContextObject(
                uri="memoryos://user/other/memories/anchors/home-comfort",
                context_type=ContextType.MEMORY,
                title="Other comfort",
                owner_user_id="other",
            )
            store.write_object(gulf_obj, content="comfort")
            store.write_object(other_obj, content="comfort")
            verify = IndexConsistencyService(store, index).verify()
            self.assertFalse(verify.consistent)
            rebuilt = IndexConsistencyService(store, index).rebuild()
            self.assertTrue(rebuilt.consistent)
            gulf_hits = index.search("comfort", filters={"owner_user_id": "gulf"})
            self.assertEqual([hit.uri for hit in gulf_hits], [gulf_obj.uri])

    def test_context_packer_respects_section_budget(self) -> None:
        packed = ContextPacker(100, allocations={"memory_rules": 20}).pack(
            {"memory_rules": [{"content": "x" * 200, "token_estimate": 50}, {"content": "small", "token_estimate": 5}]}
        )
        self.assertLessEqual(packed["slices"]["memory_rules"]["used"], 50)
        self.assertEqual(packed["total_budget"], 100)


if __name__ == "__main__":
    unittest.main()
