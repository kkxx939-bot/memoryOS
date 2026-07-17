from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.query_plan import (
    CanonicalResolutionMode,
    RetrievalOptions,
    RetrievalQueryIntent,
)
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore


def _resource_options(*, token_budget: int) -> RetrievalOptions:
    return RetrievalOptions(
        context_types=(ContextType.RESOURCE,),
        source_kinds=("resource",),
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_ids=("project-a",),
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        canonical_resolution_mode=CanonicalResolutionMode.DISABLED,
        candidate_limit=10,
        final_limit=1,
        token_budget=token_budget,
    )


def _assemble_resource(client: MemoryOSClient, *, token_budget: int) -> dict[str, Any]:
    return client.assemble_context(
        "bounded l2 hydration marker",
        options=_resource_options(token_budget=token_budget),
        user_id="u1",
        project_id="project-a",
        tenant_id="tenant-a",
        limit=1,
        token_budget=token_budget,
    )


def test_public_sdk_hydrates_only_bounded_resource_l2_and_degrades_by_budget(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
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
        tenant_id="tenant-a",
        layers=ContextLayers(l2_uri=f"{uri}/content.md"),
        metadata={
            "source_kind": "resource",
            "workspace_id": "project-a",
            "resource_location": "repository",
            "tree_paths": ["projects/project-a", "resources/repository"],
            "primary_tree_path": "projects/project-a",
        },
    )
    client.context_db.seed_object(obj, content=content)

    # Represent the normal semantic-layer projection: Source keeps complete
    # L2 while the rebuildable Catalog holds bounded L0/L1 serving text.
    record = CatalogRecord.from_context_object(obj, content=content)
    cast(SQLiteIndexStore, client.index_store).upsert_catalog(
        replace(record, l0_text=l0_text, l1_text=l1_text)
    )

    l2 = _assemble_resource(client, token_budget=1_000)
    assert len(l2["contexts"]) == 1
    assert l2["contexts"][0]["selected_layer"] == "L2"
    assert full_only_marker in l2["contexts"][0]["content"]
    assert "must-not-leak" not in l2["contexts"][0]["content"]
    assert "/Users/u1" not in l2["contexts"][0]["content"]
    assert l2["contexts"][0]["metadata"]["l2_hydrated"] is True
    assert l2["metrics"]["source_reads"] == 2
    assert l2["metrics"]["source_reads"] <= l2["query_plan"]["candidate_limit"] + 2

    l1 = _assemble_resource(client, token_budget=80)
    assert len(l1["contexts"]) == 1
    assert l1["contexts"][0]["selected_layer"] == "L1"
    assert l1["contexts"][0]["content"] == l1_text
    assert l1["metrics"]["source_reads"] == 2

    l0 = _assemble_resource(client, token_budget=10)
    assert len(l0["contexts"]) == 1
    assert l0["contexts"][0]["selected_layer"] == "L0"
    assert l0["contexts"][0]["content"] == l0_text
    assert l0["metrics"]["source_reads"] == 2

    # A partially backfilled Catalog row can lack both summaries.  The same
    # bounded L2 attempt must still finish at the final URI reference when the
    # full sanitized content does not fit.
    cast(SQLiteIndexStore, client.index_store).upsert_catalog(
        replace(record, l0_text="", l1_text="")
    )
    uri_only = _assemble_resource(client, token_budget=40)
    assert len(uri_only["contexts"]) == 1
    assert uri_only["contexts"][0]["selected_layer"] == "URI"
    assert uri_only["contexts"][0]["content"] == uri
    assert uri_only["metrics"]["source_reads"] == 2


def test_message_and_tool_result_never_read_raw_source_or_archive_during_recall(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
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
        tenant_id="tenant-a",
    )
    assert committed.session_projection_status == "projected"

    def forbidden_raw_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("atomic Session nodes must not read raw evidence during online recall")

    monkeypatch.setattr(client.source_store, "read_object", forbidden_raw_read)
    monkeypatch.setattr(client.source_store, "read_content", forbidden_raw_read)
    monkeypatch.setattr(client.session_archive_store, "read_archive", forbidden_raw_read)

    for source_kind, query in (
        ("message", "atomic message projection marker"),
        ("tool_result", "atomic tool projection marker"),
    ):
        result = client.assemble_context(
            query,
            options=RetrievalOptions(
                context_types=(ContextType.SESSION,),
                source_kinds=(source_kind,),
                tenant_id="tenant-a",
                owner_user_id="u1",
                workspace_ids=("project-a",),
                query_intent=RetrievalQueryIntent.OPEN_RECALL,
                canonical_resolution_mode=CanonicalResolutionMode.DISABLED,
                candidate_limit=10,
                final_limit=2,
                token_budget=256,
            ),
            user_id="u1",
            project_id="project-a",
            tenant_id="tenant-a",
            limit=2,
            token_budget=256,
        )
        assert result["contexts"]
        assert all(item["selected_layer"] != "L2" for item in result["contexts"])
        assert result["metrics"]["source_reads"] == 0
        assert "archive-secret" not in result["packed_context"]
        assert "/Users/u1" not in result["packed_context"]
