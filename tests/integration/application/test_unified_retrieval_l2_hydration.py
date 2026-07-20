from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from infrastructure.context.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent
from infrastructure.store.model.catalog import CatalogRecord
from infrastructure.store.model.context.context_layer import ContextLayers
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.sqlite.index_store import SQLiteIndexStore
from openApi.sdk.client import MemoryOSClient
from tests.support.persistence import seed_context_object


def _resource_options() -> RetrievalOptions:
    return RetrievalOptions(
        context_types=(ContextType.RESOURCE,),
        source_kinds=("resource",),
        tenant_id="default",
        owner_user_id="u1",
        workspace_ids=("project-a",),
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        candidate_limit=10,
        final_limit=1,
    )


def _assemble_resource(client: MemoryOSClient) -> dict[str, Any]:
    return client.assemble_context(
        "bounded l2 hydration marker",
        options=_resource_options(),
        user_id="u1",
        project_id="project-a",
        limit=1,
    )


def test_public_sdk_hydrates_only_bounded_resource_l2(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    uri = "memoryos://user/u1/resources/repository/l2-hydration-proof"
    l0_text = "bounded l2 hydration marker"
    l1_text = "bounded l2 hydration marker overview " + ("summary detail " * 12)
    full_only_marker = "full-source-evidence-only"
    content = (
        f"bounded l2 hydration marker {full_only_marker}\n"
        + ("production implementation evidence " * 100)
        + "\nAPI_KEY=must-not-leak /Users/u1/private/credentials.txt"
    )
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.RESOURCE,
        title="Bounded L2 hydration marker resource",
        owner_user_id="u1",
        tenant_id="default",
        layers=ContextLayers(l2_uri=f"{uri}/content.md"),
        metadata={
            "source_kind": "resource",
            "workspace_id": "project-a",
            "resource_location": "repository",
            "tree_paths": ["projects/project-a", "resources/repository"],
            "primary_tree_path": "projects/project-a",
        },
    )
    seed_context_object(client.runtime.stores.source, client.runtime.stores.index, obj, content=content)

    # Represent the normal semantic-layer projection: Source keeps complete
    # L2 while the rebuildable Catalog holds bounded L0/L1 serving text.
    record = CatalogRecord.from_context_object(obj, content=content)
    cast(SQLiteIndexStore, client.runtime.stores.index).upsert_catalog(
        replace(record, l0_text=l0_text, l1_text=l1_text),
        tenant_id="default",
    )

    l2 = _assemble_resource(client)
    assert len(l2["contexts"]) == 1
    assert l2["contexts"][0]["selected_layer"] == "L2"
    assert full_only_marker in l2["contexts"][0]["content"]
    assert "must-not-leak" not in l2["contexts"][0]["content"]
    assert "/Users/u1" not in l2["contexts"][0]["content"]
    assert l2["metrics"]["source_reads"] == 2
    assert l2["metrics"]["source_reads"] <= l2["query_plan"]["candidate_limit"] + 2

def test_message_and_tool_result_never_read_raw_source_or_archive_during_recall(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = client.commit_agent_session(
        user_id="u1",
        session_id="no-raw-atomic-session",
        messages=[
            {
                "role": "user",
                "content": "atomic message projection marker",
                "occurred_at": "2026-07-14T09:00:00+08:00",
            }
        ],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": "atomic tool projection marker API_KEY=archive-secret",
                "path": "/Users/u1/Desktop/atomic-proof.txt",
                "occurred_at": "2026-07-14T09:01:00+08:00",
            }
        ],
        async_commit=False,
        project_id="project-a",
    )
    assert committed.session_projection_status == "projected"

    def forbidden_raw_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("atomic Session nodes must not read raw evidence during online recall")

    monkeypatch.setattr(client.runtime.stores.source, "read_object", forbidden_raw_read)
    monkeypatch.setattr(client.runtime.stores.source, "read_content", forbidden_raw_read)
    monkeypatch.setattr(client.runtime.session.archive_store, "read_archive", forbidden_raw_read)

    for source_kind, query in (
        ("message", "atomic message projection marker"),
        ("tool_result", "atomic tool projection marker"),
    ):
        result = client.assemble_context(
            query,
            options=RetrievalOptions(
                context_types=(ContextType.SESSION,),
                source_kinds=(source_kind,),
                tenant_id="default",
                owner_user_id="u1",
                workspace_ids=("project-a",),
                query_intent=RetrievalQueryIntent.OPEN_RECALL,
                candidate_limit=10,
                final_limit=2,
            ),
            user_id="u1",
            project_id="project-a",
            limit=2,
        )
        assert result["contexts"]
        assert all(item["selected_layer"] != "L2" for item in result["contexts"])
        assert result["metrics"]["source_reads"] == 0
        assert "archive-secret" not in result["packed_context"]
        assert "/Users/u1" not in result["packed_context"]
