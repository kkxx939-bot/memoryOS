from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from infrastructure.context.candidate import CandidateGenerator
from infrastructure.context.orchestrator import UnifiedRetrievalOrchestrator
from infrastructure.context.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent, RetrievalQueryPlan
from infrastructure.store.contracts.vector import VectorHit, vector_row_id
from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind, catalog_vector_metadata
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.sqlite.index_store import SQLiteIndexStore
from openApi.sdk.client import MemoryOSClient
from tests.support.persistence import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
    InMemoryVectorStore,
)

_TIME = "2026-07-14T03:30:00+00:00"


def test_candidate_generator_binds_tenant_on_generic_index_search() -> None:
    index = InMemoryIndexStore()
    shared_uri = "memoryos://user/u1/resources/shared"
    for tenant_id, content in (
        ("tenant-a", "tenant-bound-query visible-a"),
        ("tenant-b", "tenant-bound-query hidden-b"),
    ):
        index.upsert_index(
            ContextObject(
                uri=shared_uri,
                context_type=ContextType.RESOURCE,
                title=content,
                tenant_id=tenant_id,
                owner_user_id="u1",
            ),
            content=content,
            tenant_id=tenant_id,
        )
    generated = CandidateGenerator(index).generate(
        RetrievalQueryPlan(
            semantic_query="tenant-bound-query",
            tenant_id="tenant-a",
            owner_user_id="u1",
            candidate_limit=10,
            final_limit=10,
        )
    )
    assert generated.fts_candidates == 1
    assert generated.branches["lexical"][0].tenant_id == "tenant-a"
    assert generated.branches["lexical"][0].title.endswith("visible-a")


def test_generic_in_memory_index_preserves_source_digest_for_bounded_l2(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path / "embedded-source", tenant_id="tenant-a")
    index = InMemoryIndexStore()
    obj = ContextObject(
        uri="memoryos://user/u1/resources/embedded",
        context_type=ContextType.RESOURCE,
        title="embedded resource",
        tenant_id="tenant-a",
        owner_user_id="u1",
    )
    content = "embedded-l2-marker exact source bytes"
    source.write_object(obj, content=content)
    index.upsert_index(obj, content=content, tenant_id="tenant-a")
    relations = InMemoryRelationStore()

    result = UnifiedRetrievalOrchestrator(
        index,
        source_store=source,
        relation_store=relations,
        queue_store=None,
        session_archive_store=None,
    ).execute(
        RetrievalQueryPlan(
            semantic_query="embedded-l2-marker",
            context_types=(ContextType.RESOURCE,),
            tenant_id="tenant-a",
            owner_user_id="u1",
            candidate_limit=10,
            final_limit=10,
        )
    )

    assert result.contexts[0]["selected_layer"] == "L2"
    assert result.contexts[0]["content"] == content


def _session_root(session_index: int) -> CatalogRecord:
    session_id = f"session-{session_index:04d}"
    return CatalogRecord(
        record_key=f"session:{session_id}:root",
        uri=f"memoryos://user/u1/sessions/history/{session_id}/context/root",
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_id="memoryOS",
        session_id=session_id,
        adapter_id="codex",
        context_type="session",
        source_kind="session_root",
        record_kind=CatalogRecordKind.SESSION_ROOT.value,
        tree_paths=(f"sessions/{session_id}", "projects/memoryOS", "timeline/2026/07/14"),
        created_at=_TIME,
        updated_at=_TIME,
        event_time=_TIME,
        ingested_at=_TIME,
        transaction_time=_TIME,
        title=f"Session {session_id}",
        l0_text=f"Session {session_index}",
        l1_text=f"Bounded serving overview for session {session_index}",
        source_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        source_digest=f"session-digest-{session_index:04d}",
        metadata={"vector_eligible": True},
    )


def _tool_result(tool_index: int) -> CatalogRecord:
    session_index = tool_index // 10
    session_id = f"session-{session_index:04d}"
    name = f"report-{tool_index:05d}.txt"
    return CatalogRecord(
        record_key=f"session:{session_id}:tool:{tool_index:05d}",
        uri=f"memoryos://user/u1/sessions/history/{session_id}/context/tool/{tool_index:05d}",
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_id="memoryOS",
        session_id=session_id,
        adapter_id="codex",
        context_type="session",
        source_kind="tool_result",
        record_kind=CatalogRecordKind.TOOL_RESULT.value,
        tree_paths=("timeline/2026/07/14", f"sessions/{session_id}", "resources/repository"),
        created_at=_TIME,
        updated_at=_TIME,
        event_time=_TIME,
        ingested_at=_TIME,
        transaction_time=_TIME,
        title=name,
        l0_text=f"Repository result {name}",
        l1_text=f"boundedneedle{tool_index:05d} from {name}",
        source_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        source_digest=f"tool-digest-{tool_index:05d}",
        metadata={"resource_name": name, "resource_location": "repository", "vector_eligible": False},
    )


def _memory_document(document_index: int) -> CatalogRecord:
    document_id = f"document-{document_index:04d}"
    uri = f"memoryos://user/u1/memories/documents/{document_id}"
    return CatalogRecord(
        record_key=f"memory-document:u1:{document_id}",
        uri=uri,
        tenant_id="tenant-a",
        owner_user_id="u1",
        context_type="memory",
        source_kind="markdown_memory_document",
        record_kind=CatalogRecordKind.MEMORY_DOCUMENT.value,
        tree_paths=(f"memories/knowledge/topics/user-{document_index:04d}",),
        primary_tree_path=f"memories/knowledge/topics/user-{document_index:04d}",
        created_at=_TIME,
        updated_at=_TIME,
        transaction_time=_TIME,
        title=f"Topic document {document_index}",
        l0_text=f"Topic {document_index}",
        l1_text=f"Markdown topic document {document_index}",
        l2_uri=uri,
        source_uri=uri,
        source_digest=f"document-digest-{document_index:04d}",
        source_revision=3,
        document_id=document_id,
        document_kind="topic",
        document_revision=3,
        projection_generation=3,
        projection_effect_hash=f"document-digest-{document_index:04d}",
        metadata={"relative_path": f"knowledge/topics/user-{document_index:04d}.md"},
    )


def _memory_block(document_index: int, block_index: int) -> CatalogRecord:
    document = _memory_document(document_index)
    block_id = f"block-{document_index:04d}-{block_index}"
    return CatalogRecord(
        **{
            **document.__dict__,
            "record_key": f"memory-block:u1:{block_id}",
            "uri": f"{document.uri}/blocks/{block_id}",
            "record_kind": CatalogRecordKind.MEMORY_BLOCK.value,
            "parent_uri": document.uri,
            "block_id": block_id,
            "title": f"Topic section {block_index}",
            "l0_text": f"Section {block_index}",
            "l1_text": f"Markdown block {document_index}-{block_index}",
        }
    )


def test_sqlite_catalog_handles_1000_sessions_10000_tools_and_1000_memory_documents(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = SQLiteIndexStore(tmp_path / "scale.sqlite3")

    records = [
        *(_session_root(index) for index in range(1_000)),
        *(_tool_result(index) for index in range(10_000)),
    ]
    for start in range(0, len(records), 500):
        assert store.upsert_catalog_batch(
            records[start : start + 500], tenant_id="tenant-a"
        ) == len(records[start : start + 500])
    for index in range(1_000):
        document = _memory_document(index)
        store.replace_memory_document_projection(
            document,
            tuple(_memory_block(index, block_index) for block_index in range(3)),
            None,
            tenant_id="tenant-a",
            owner_user_id="u1",
        )
    # Keep a hard serving ceiling while allowing this deliberately oversized
    # fixture to exercise FTS plus ACL and path predicates in one query.
    store.online_vm_step_limit = 5_000_000

    filters = {
        "tenant_id": "tenant-a",
        "principal_owner_id": "u1",
        "workspace_access_ids": ("", "memoryOS"),
        "context_types": ("session",),
        "record_kinds": (CatalogRecordKind.TOOL_RESULT.value,),
        "target_paths": ("resources/repository",),
        "event_time_from": "2026-07-14T00:00:00+00:00",
        "event_time_to": "2026-07-15T00:00:00+00:00",
    }
    hits = store.search_catalog("boundedneedle09999", tenant_id="tenant-a", filters=filters, limit=25)
    assert hits
    assert hits[0].metadata["catalog_record_key"] == "session:session-0999:tool:09999"
    assert len(hits) <= 25

    with sqlite3.connect(store.path) as conn:
        counts = dict(conn.execute("SELECT record_kind, count(*) FROM contexts GROUP BY record_kind"))
        document_duplicates = conn.execute(
            "SELECT count(*) FROM (SELECT document_id FROM contexts "
            "WHERE record_kind = 'memory_document' "
            "GROUP BY tenant_id, owner_user_id, document_id HAVING count(*) > 1)"
        ).fetchone()[0]

        def explain(sql: str, parameters: Sequence[Any]) -> str:
            return " ".join(str(row[3]) for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}", tuple(parameters)))

        index_plans = {
            "tenant": explain(
                "SELECT record_key FROM contexts WHERE tenant_id = ? LIMIT 25",
                ("tenant-a",),
            ),
            "owner": explain(
                "SELECT record_key FROM contexts WHERE tenant_id = ? AND owner_user_id = ? LIMIT 25",
                ("tenant-a", "u1"),
            ),
            "context_type": explain(
                "SELECT record_key FROM context_paths WHERE tenant_id = ? AND context_type = ? "
                "AND path >= ? AND path < ? LIMIT 25",
                ("tenant-a", "session", "resources/repository", "resources/repository/\uffff"),
            ),
            "path": explain(
                "SELECT record_key FROM context_paths WHERE tenant_id = ? AND path >= ? AND path < ? LIMIT 25",
                ("tenant-a", "resources/repository", "resources/repository/\uffff"),
            ),
            "event_time": explain(
                "SELECT record_key FROM contexts WHERE tenant_id = ? AND event_time >= ? AND event_time < ? LIMIT 25",
                ("tenant-a", "2026-07-14T00:00:00+00:00", "2026-07-15T00:00:00+00:00"),
            ),
            "transaction_time": explain(
                "SELECT record_key FROM contexts WHERE tenant_id = ? AND transaction_time >= ? "
                "AND transaction_time < ? LIMIT 25",
                ("tenant-a", "2026-07-14T00:00:00+00:00", "2026-07-15T00:00:00+00:00"),
            ),
        }
    assert counts[CatalogRecordKind.SESSION_ROOT.value] == 1_000
    assert counts[CatalogRecordKind.TOOL_RESULT.value] == 10_000
    assert counts[CatalogRecordKind.MEMORY_DOCUMENT.value] == 1_000
    assert counts[CatalogRecordKind.MEMORY_BLOCK.value] == 3_000
    assert document_duplicates == 0

    query_plan = " ".join(
        store.explain_structured_query(tenant_id="tenant-a", filters=filters, limit=25)
    )
    assert "idx_contexts_tenant_event_time" in query_plan
    assert "idx_context_path" in query_plan
    assert "context_acl_grants" in query_plan
    assert "INDEX" in index_plans["tenant"]
    assert "INDEX" in index_plans["owner"]
    assert "idx_context_paths_tenant_path" in index_plans["context_type"]
    assert "idx_context_paths_tenant_path" in index_plans["path"]
    assert "idx_contexts_tenant_event_time" in index_plans["event_time"]
    assert "idx_contexts_tenant_transaction_time" in index_plans["transaction_time"]
    document_plan = " ".join(
        store.explain_structured_query(
            tenant_id="tenant-a",
            filters={
                "tenant_id": "tenant-a",
                "document_ids": ("document-0001",),
                "record_kinds": (CatalogRecordKind.MEMORY_DOCUMENT.value,),
            },
            limit=1,
        )
    )
    assert "idx_contexts_tenant_document_id" in document_plan
    exact_document = store.list_catalog(
        tenant_id="tenant-a",
        filters={
            "tenant_id": "tenant-a",
            "document_ids": ("document-0001",),
            "record_kinds": (CatalogRecordKind.MEMORY_DOCUMENT.value,),
        },
        limit=1,
    )
    assert [record.record_key for record in exact_document] == ["memory-document:u1:document-0001"]

    source = FileSystemSourceStore(tmp_path / "source", tenant_id="tenant-a")
    vector = _NoEnumerationVector()
    vector.reject_enumeration = True

    def prohibited(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("large online retrieval attempted a prohibited full scan")

    monkeypatch.setattr(source, "list_objects", prohibited)
    monkeypatch.setattr(Path, "glob", prohibited)
    monkeypatch.setattr(Path, "rglob", prohibited)
    relations = InMemoryRelationStore()
    orchestrator = UnifiedRetrievalOrchestrator(
        store,
        source_store=source,
        relation_store=relations,
        queue_store=None,
        session_archive_store=None,
        vector_store=vector,
        embedding_provider=_Embedding(),
    )
    result = orchestrator.execute(
        RetrievalQueryPlan(
            semantic_query="boundedneedle09999",
            target_paths=("resources/repository",),
            context_types=(ContextType.SESSION,),
            record_kinds=(CatalogRecordKind.TOOL_RESULT.value,),
            tenant_id="tenant-a",
            owner_user_id="u1",
            workspace_ids=("memoryOS",),
            event_time_from="2026-07-14T00:00:00+00:00",
            event_time_to="2026-07-15T00:00:00+00:00",
            query_intent=RetrievalQueryIntent.OPEN_RECALL,
            candidate_limit=25,
            final_limit=10,
        )
    )
    assert result.contexts
    assert result.contexts[0]["metadata"]["catalog_record_key"] == "session:session-0999:tool:09999"
    assert result.metrics.structured_candidates == 0
    assert result.metrics.exact_candidates == 0
    assert result.metrics.fts_candidates == 1
    assert result.metrics.selected_count <= 25
    assert result.metrics.source_reads <= 28
    assert result.metrics.vector_overfetch <= 200
    assert result.metrics.selected_count <= 10
    assert vector.max_candidate_count <= 25


def test_document_identity_bounds_10000_blocks_before_document_and_block_top_k(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = SQLiteIndexStore(tmp_path / "large-memory-document.sqlite3")
    document = _memory_document(9_999)
    blocks = tuple(_memory_block(9_999, block_index) for block_index in range(10_000))
    store.replace_memory_document_projection(
        document,
        blocks,
        None,
        tenant_id="tenant-a",
        owner_user_id="u1",
    )

    real_list_catalog = store.list_catalog
    catalog_calls: list[tuple[dict[str, Any], int]] = []

    def recording_list_catalog(
        *,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        copied = dict(filters or {})
        if copied.get("target_identity_uris") or copied.get("document_ids"):
            catalog_calls.append((copied, limit))
        return real_list_catalog(tenant_id=tenant_id, filters=copied, limit=limit)

    monkeypatch.setattr(store, "list_catalog", recording_list_catalog)
    # Materializing every block exceeds this ceiling. Both the live document
    # lookup and the block stream must remain indexed, branch-local Top-K.
    store.online_vm_step_limit = 50_000
    document_result = CandidateGenerator(store).generate(
        RetrievalQueryPlan(
            semantic_query="",
            target_uris=(document.uri,),
            context_types=(ContextType.MEMORY,),
            record_kinds=(CatalogRecordKind.MEMORY_DOCUMENT.value,),
            document_ids=(document.document_id,),
            tenant_id="tenant-a",
            owner_user_id="u1",
            query_intent=RetrievalQueryIntent.EXACT,
            candidate_limit=3,
            final_limit=3,
        )
    )
    document_filters, document_limit = catalog_calls[-1]
    assert [candidate.record_key for candidate in document_result.branches["exact"]] == [
        document.record_key
    ]
    assert document_result.exact_candidates == 1
    assert document_limit == 3
    assert document_filters["record_kinds"] == (CatalogRecordKind.MEMORY_DOCUMENT.value,)
    assert document_filters["document_ids"] == (document.document_id,)
    assert document_filters["_identity_candidate_limit"] == 3

    catalog_calls.clear()
    block_result = CandidateGenerator(store).generate(
        RetrievalQueryPlan(
            semantic_query="",
            context_types=(ContextType.MEMORY,),
            record_kinds=(CatalogRecordKind.MEMORY_BLOCK.value,),
            document_ids=(document.document_id,),
            tenant_id="tenant-a",
            owner_user_id="u1",
            query_intent=RetrievalQueryIntent.CURRENT,
            candidate_limit=7,
            final_limit=7,
        )
    )
    block_filters, block_limit = catalog_calls[-1]
    assert len(block_result.branches["structured"]) == 7
    assert all(
        candidate.record_kind == CatalogRecordKind.MEMORY_BLOCK.value
        and candidate.document_id == document.document_id
        for candidate in block_result.branches["structured"]
    )
    assert block_result.structured_candidates == 7
    assert block_result.exact_candidates == 0
    assert block_limit == 7
    assert block_filters["record_kinds"] == (CatalogRecordKind.MEMORY_BLOCK.value,)
    assert block_filters["document_ids"] == (document.document_id,)

    explain = store.explain_structured_query(
        tenant_id="tenant-a",
        filters=block_filters,
        limit=block_limit,
    )
    joined = " ".join(explain)
    assert "idx_contexts_tenant_document" in joined
    assert "context_acl_grants" in joined
    assert not any("SCAN c" in detail for detail in explain)


def test_metadata_exact_applies_acl_before_overflow_detection(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "exact-acl-before-bound.sqlite3")
    timestamp = "2026-07-14T01:00:00+00:00"
    crowded = tuple(
        CatalogRecord(
            record_key=f"unauthorized-exact-{index:04d}",
            uri=f"memoryos://user/u2/sessions/history/crowded-{index:04d}",
            tenant_id="tenant-a",
            owner_user_id="u2",
            workspace_id="other-project",
            context_type="session",
            source_kind="message",
            record_kind=CatalogRecordKind.MESSAGE.value,
            tree_paths=("timeline/2026/07/14",),
            created_at=timestamp,
            updated_at=timestamp,
            event_time=timestamp,
            ingested_at=timestamp,
            transaction_time=timestamp,
            title="Unauthorized exact identity",
            l0_text="unrelated",
            l1_text="unrelated",
            metadata={"scene_key": "shared-exact-identity"},
        )
        for index in range(901)
    )
    authorized = CatalogRecord(
        record_key="authorized-exact",
        uri="memoryos://user/u1/sessions/history/authorized",
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_id="memoryOS",
        context_type="session",
        source_kind="message",
        record_kind=CatalogRecordKind.MESSAGE.value,
        tree_paths=("timeline/2026/07/14",),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title="Authorized exact identity",
        l0_text="authorized",
        l1_text="authorized",
        metadata={"scene_key": "shared-exact-identity"},
    )
    assert store.upsert_catalog_batch((*crowded, authorized), tenant_id="tenant-a") == 902

    hits = store.search_catalog(
        "shared-exact-identity",
        tenant_id="tenant-a",
        filters={
            "tenant_id": "tenant-a",
            "principal_owner_id": "u1",
            "workspace_access_ids": ("", "memoryOS"),
            "context_types": ("session",),
            "source_kinds": ("message",),
            "record_kinds": (CatalogRecordKind.MESSAGE.value,),
            "target_paths": ("timeline/2026/07/14",),
            "event_time_from": "2026-07-14T00:00:00+00:00",
            "event_time_to": "2026-07-15T00:00:00+00:00",
        },
        limit=10,
    )

    assert [hit.uri for hit in hits] == [authorized.uri]
    assert hits[0].metadata["owner_user_id"] == "u1"


class _Embedding:
    model_name = "scale-test"
    dimension = 2

    def embed(self, text: str) -> list[float]:
        return [1.0, float(bool(text))]


class _NoEnumerationVector(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.reject_enumeration = False
        self.max_candidate_count = 0

    def vector_uris(self) -> list[str]:
        if self.reject_enumeration:
            raise AssertionError("online retrieval called vector_uris()")
        return super().vector_uris()

    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]:
        raise AssertionError("online retrieval used unfiltered vector search")

    def search_vector_candidates(
        self,
        embedding: list[float],
        candidate_uris: list[str] | tuple[str, ...],
        *,
        limit: int = 10,
    ) -> list[VectorHit]:
        self.max_candidate_count = max(self.max_candidate_count, len(candidate_uris))
        return super().search_vector_candidates(embedding, candidate_uris, limit=limit)


def test_online_sdk_chain_never_enumerates_sources_vectors_or_archive_tree(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    vector = _NoEnumerationVector()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vector,
        embedding_provider=_Embedding(),
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="bounded-session",
        messages=[{"role": "user", "content": "read repository report"}],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": "bounded online result",
                "path": "/Users/u1/repository/bounded-report.txt",
                "occurred_at": "2026-07-14T09:00:00+08:00",
            }
        ],
        async_commit=False,
        project_id="memoryOS",
    )
    tool = cast(SQLiteIndexStore, client.runtime.stores.index).list_catalog(
        tenant_id="default",
        filters={"tenant_id": "default", "record_kinds": (CatalogRecordKind.TOOL_RESULT.value,)},
        limit=10,
    )[0]
    vector.upsert_vector(
        vector_row_id("default", tool.record_key),
        [1.0, 1.0],
        catalog_vector_metadata(tool),
    )
    vector.reject_enumeration = True

    def prohibited(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("online retrieval attempted a prohibited full scan")

    monkeypatch.setattr(client.runtime.stores.source, "list_objects", prohibited)
    monkeypatch.setattr(Path, "glob", prohibited)
    monkeypatch.setattr(Path, "rglob", prohibited)

    options = RetrievalOptions(
        target_paths=("resources/repository",),
        context_types=(ContextType.SESSION,),
        record_kinds=(CatalogRecordKind.TOOL_RESULT.value,),
        tenant_id="default",
        owner_user_id="u1",
        workspace_ids=("memoryOS",),
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        candidate_limit=25,
        final_limit=10,
    )
    hits = client.search_context(
        "bounded-report.txt",
        options=options,
        user_id="u1",
        project_id="memoryOS",
    )
    trace = client.recall_trace(client.last_recall_trace_id)

    assert hits
    assert vector.max_candidate_count <= options.candidate_limit
    assert int(trace["memory_validated"]) <= options.candidate_limit
    assert int(trace["source_reads"]) <= options.candidate_limit
    assert int(trace["vector_overfetch"]) <= 200
    assert int(trace["selected_count"]) <= options.final_limit
    assert len([item for item in hits if dict(item["metadata"]).get("session_id") == "bounded-session"]) <= 5
