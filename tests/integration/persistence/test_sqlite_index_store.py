from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import InMemoryIndexStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.core.types import ScopeRef


class SQLiteIndexStoreTest(unittest.TestCase):
    def test_upsert_search_delete_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteIndexStore(Path(tmp) / "index.sqlite3")
            u1 = ContextObject(
                uri="memoryos://user/u1/resources/preferences/temp",
                context_type=ContextType.RESOURCE,
                title="temperature preference",
                owner_user_id="u1",
            )
            u2 = ContextObject(
                uri="memoryos://user/u2/resources/preferences/temp",
                context_type=ContextType.RESOURCE,
                title="temperature preference",
                owner_user_id="u2",
            )
            policy = ContextObject(
                uri="memoryos://user/u1/action_policies/hot/turn_on_ac",
                context_type=ContextType.ACTION_POLICY,
                title="turn on ac policy",
                owner_user_id="u1",
            )
            store.upsert_index(u1, content="prefers 26 degree", tenant_id="default")
            store.upsert_index(u2, content="prefers 18 degree", tenant_id="default")
            store.upsert_index(policy, content="hot room turn_on_ac", tenant_id="default")
            self.assertEqual(
                store.search("26", tenant_id="default", filters={"owner_user_id": "u1"})[0].uri,
                u1.uri,
            )
            self.assertFalse(store.search("18", tenant_id="default", filters={"owner_user_id": "u1"}))
            self.assertEqual(
                store.search(
                    "turn_on_ac",
                    tenant_id="default",
                    filters={"owner_user_id": "u1", "context_type": "action_policy"},
                )[0].uri,
                policy.uri,
            )
            store.delete_index(u1.uri, tenant_id="default")
            self.assertFalse(store.search("26", tenant_id="default", filters={"owner_user_id": "u1"}))

    def test_applicability_filter_treats_query_scopes_as_available_superset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteIndexStore(Path(tmp) / "index.sqlite3")
            obj = ContextObject(
                uri="memoryos://user/u1/resources/preferences/music",
                context_type=ContextType.RESOURCE,
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
            store.upsert_index(obj, content="quiet music", tenant_id="default")
            available = [
                "memoryos:principal:u1",
                "memoryos:environment:home",
                "memoryos:asset:reachy_01",
                "memoryos:location:kitchen",
            ]
            assert store.search("quiet", tenant_id="default", filters={"applicability_scope_keys": available})
            assert not store.search(
                "quiet",
                tenant_id="default",
                filters={"applicability_scope_keys": ["memoryos:principal:u1"]},
            )

    def test_parent_aware_scope_keys_are_indexed_without_flat_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.sqlite3"
            store = SQLiteIndexStore(path)

            def scoped(uri: str, parent: str) -> ContextObject:
                return ContextObject(
                    uri=uri,
                    context_type=ContextType.RESOURCE,
                    title="camera calibration",
                    owner_user_id="u1",
                    metadata={
                        "scope": {
                            "applicability": {
                                "all_of": [
                                    {
                                        "namespace": "memoryos",
                                        "kind": "asset",
                                        "id": "camera",
                                        "parent_path": [parent],
                                    }
                                ]
                            }
                        }
                    },
                )

            first = scoped("memoryos://user/u1/resources/camera-a", "workspace-a")
            second = scoped("memoryos://user/u1/resources/camera-b", "workspace-b")
            malformed = scoped("memoryos://user/u1/resources/malformed", "workspace-c")
            malformed.title = "malformed legacy scope"
            store.upsert_index(first, content="camera calibration", tenant_id="default")
            store.upsert_index(second, content="camera calibration", tenant_id="default")
            store.upsert_index(malformed, content="malformed scope", tenant_id="default")
            first_key = ScopeRef("memoryos", "asset", "camera", parent_path=("workspace-a",)).key
            assert [
                hit.uri
                for hit in store.search(
                    "camera",
                    tenant_id="default",
                    filters={"owner_user_id": "u1", "applicability_scope_keys": [first_key]},
                )
            ] == [first.uri]

    def test_fts_disabled_does_not_restore_python_row_scan_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteIndexStore(Path(tmp) / "index.sqlite3")
            related = ContextObject(
                uri="memoryos://user/u1/resources/redis",
                context_type=ContextType.RESOURCE,
                title="Redis cache",
                owner_user_id="u1",
            )
            unrelated = ContextObject(
                uri="memoryos://user/u1/resources/redistribution",
                context_type=ContextType.RESOURCE,
                title="redistribution guide",
                owner_user_id="u1",
                hotness=1.0,
                semantic_hotness=1.0,
                behavior_support_hotness=1.0,
            )
            chinese = ContextObject(
                uri="memoryos://user/u1/resources/chinese",
                context_type=ContextType.RESOURCE,
                title="数据库继续使用PostgreSQL",
                owner_user_id="u1",
            )
            store.upsert_index(related, content="Redis is the cache backend", tenant_id="default")
            store.upsert_index(unrelated, content="redistribution strategy", tenant_id="default")
            store.upsert_index(chinese, content="生产数据库继续使用PostgreSQL", tenant_id="default")
            store.fts_enabled = False

            assert store.search("Redis", tenant_id="default", filters={"owner_user_id": "u1"}) == []
            assert store.search("数据库继续使用", tenant_id="default", filters={"owner_user_id": "u1"}) == []

    def test_runtime_fts_storage_failure_is_not_reported_as_empty_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteIndexStore(Path(tmp) / "index.sqlite3")
            with store._connect() as conn:  # noqa: SLF001 - failure-injection boundary.
                conn.execute("DROP TABLE contexts_fts")

            with self.assertRaises(sqlite3.OperationalError):
                store.search_catalog("must remain observable", tenant_id="default")

    def test_inmemory_lexical_matching_matches_sqlite_token_semantics(self) -> None:
        store = InMemoryIndexStore()
        related = ContextObject(
            uri="memoryos://user/u1/resources/redis",
            context_type=ContextType.RESOURCE,
            title="Redis cache",
            owner_user_id="u1",
        )
        unrelated = ContextObject(
            uri="memoryos://user/u1/resources/redistribution",
            context_type=ContextType.RESOURCE,
            title="redistribution guide",
            owner_user_id="u1",
            hotness=1.0,
        )
        chinese = ContextObject(
            uri="memoryos://user/u1/resources/chinese",
            context_type=ContextType.RESOURCE,
            title="数据库继续使用PostgreSQL",
            owner_user_id="u1",
        )
        store.upsert_index(related, content="Redis cache backend", tenant_id="default")
        store.upsert_index(unrelated, content="redistribution strategy", tenant_id="default")
        store.upsert_index(chinese, content="生产数据库继续使用PostgreSQL", tenant_id="default")

        assert [
            hit.uri
            for hit in store.search("Redis", tenant_id="default", filters={"owner_user_id": "u1"})
        ] == [related.uri]
        assert chinese.uri in {
            hit.uri
            for hit in store.search("数据库继续使用", tenant_id="default", filters={"owner_user_id": "u1"})
        }


if __name__ == "__main__":
    unittest.main()
