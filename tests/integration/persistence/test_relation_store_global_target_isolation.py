from __future__ import annotations

from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore


def test_global_resource_reverse_lookup_filters_owner_user_id(tmp_path) -> None:
    store = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    resource_uri = "memoryos://resources/devices/ac"
    store.add_relation(
        ContextRelation(
            source_uri="memoryos://user/u1/action_policies/hot/turn_on_ac",
            relation_type="requires_resource",
            target_uri=resource_uri,
            metadata={"owner_user_id": "u1", "tenant_id": "default"},
        )
    )
    store.add_relation(
        ContextRelation(
            source_uri="memoryos://user/u2/action_policies/hot/turn_on_ac",
            relation_type="requires_resource",
            target_uri=resource_uri,
            metadata={"owner_user_id": "u2", "tenant_id": "default"},
        )
    )

    rows = store.relations_of(resource_uri, owner_user_id="u1")

    assert [row.source_uri for row in rows] == ["memoryos://user/u1/action_policies/hot/turn_on_ac"]
