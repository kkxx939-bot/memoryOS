from __future__ import annotations

import tempfile
import unittest

from infrastructure.context.layers import LayerRefresher
from infrastructure.store.filesystem import FileSystemSourceStore
from infrastructure.store.model.context import ContextObject, ContextType
from tests.support.persistence import InMemoryVectorStore


class ContextDBFinalComponentsTest(unittest.TestCase):
    def test_layer_refresher_writes_l0_l1_l2_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            obj = ContextObject(
                uri="memoryos://user/gulf/resources/home-comfort",
                context_type=ContextType.RESOURCE,
                title="Home comfort",
                owner_user_id="gulf",
            )
            result = LayerRefresher(source).refresh(obj, "User prefers comfort around 26C.", ["prefers 26C"])
            self.assertIn(".abstract.md", result.l0_uri)
            self.assertIn(".overview.md", result.l1_uri)
            loaded = source.read_object(obj.uri)
            self.assertEqual(loaded.layers.l2_uri, result.l2_uri)

    def test_vector_store(self) -> None:
        obj = ContextObject(
            uri="memoryos://user/gulf/behavior/patterns/hot-weather",
            context_type=ContextType.BEHAVIOR_PATTERN,
            title="hot weather home behavior",
            owner_user_id="gulf",
        )
        vector = InMemoryVectorStore()
        vector.upsert_vector(obj.uri, [1.0, 0.0])
        self.assertEqual(vector.search_vector([1.0, 0.0], "memoryos://user/gulf")[0].uri, obj.uri)


if __name__ == "__main__":
    unittest.main()
