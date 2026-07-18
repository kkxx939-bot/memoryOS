from __future__ import annotations

import tempfile
import unittest

from memoryos.adapters.sqlite import SqliteIndexStore, SqliteQueueStore, SqliteRelationStore
from memoryos.contextdb.layers import LayerRefresher
from memoryos.contextdb.model import ContextObject, ContextRelation, ContextType
from memoryos.contextdb.resource import ResourceImporter
from memoryos.contextdb.skill import Skill, SkillContextBuilder, SkillRegistry
from memoryos.contextdb.store import FileSystemSourceStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore


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

    def test_sqlite_index_relation_queue_are_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = SqliteIndexStore(f"{tmp}/index.sqlite3")
            relation = SqliteRelationStore(f"{tmp}/relation.sqlite3")
            queue = SqliteQueueStore(f"{tmp}/queue.sqlite3")
            obj = ContextObject(
                uri="memoryos://user/gulf/resources/hot",
                context_type=ContextType.RESOURCE,
                title="Hot anchor",
                owner_user_id="gulf",
            )
            index.upsert_index(obj, "hot weather comfort", tenant_id="default")
            self.assertEqual(
                index.search(
                    "hot",
                    tenant_id="default",
                    filters={"owner_user_id": "gulf"},
                )[0].uri,
                obj.uri,
            )
            relation.add_relation(
                ContextRelation(
                    source_uri=obj.uri,
                    relation_type="requires_skill",
                    target_uri="memoryos://skills/ac/control",
                ),
                tenant_id="default",
            )
            self.assertEqual(
                relation.relations_of(obj.uri, tenant_id="default")[0].relation_type,
                "requires_skill",
            )
            from memoryos.contextdb.store import QueueJob

            queue.enqueue(QueueJob(job_id="j1", queue_name="semantic", action="refresh", target_uri=obj.uri))
            leased = queue.lease("semantic", lease_owner="test", limit=1)
            self.assertEqual(leased[0].job_id, "j1")
            queue.ack(leased[0])
            self.assertEqual(queue.lease("semantic", lease_owner="test", limit=1), [])

    def test_resource_skill_and_vector_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            resource = ResourceImporter(source).import_text(
                "memoryos://resources/devices/ac-living-room",
                "Living room AC",
                "device",
                '{"device_id":"ac1","capability":"cooling"}',
            )
            self.assertEqual(resource.metadata["device_id"], "ac1")
            registry = SkillRegistry()
            skill = Skill(uri="memoryos://skills/smart_home/ac-control", title="AC Control", tool_name="ac.set_temperature")
            registry.register(skill)
            self.assertEqual(SkillContextBuilder(registry).build([skill.uri])[0]["context_type"], "skill")
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
