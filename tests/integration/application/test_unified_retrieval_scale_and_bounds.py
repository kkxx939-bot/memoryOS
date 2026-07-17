from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.candidate_generator import CandidateGenerator
from memoryos.contextdb.retrieval.orchestrator import UnifiedRetrievalOrchestrator
from memoryos.contextdb.retrieval.query_plan import (
    CanonicalResolutionMode,
    RetrievalOptions,
    RetrievalQueryIntent,
    RetrievalQueryPlan,
)
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryRelationStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore, VectorHit

_TIME = "2026-07-14T03:30:00+00:00"


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


def _current_slot(slot_index: int) -> CatalogRecord:
    slot_id = f"slot-{slot_index:04d}"
    claim_id = f"claim-{slot_index:04d}-b"
    return CatalogRecord(
        record_key=f"slot:{slot_id}:current",
        uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}",
        tenant_id="tenant-a",
        owner_user_id="u1",
        context_type="memory",
        source_kind="canonical_projection",
        record_kind=CatalogRecordKind.CURRENT_SLOT.value,
        tree_paths=(f"memories/preferences/user-{slot_index:04d}/dimension",),
        created_at=_TIME,
        updated_at=_TIME,
        transaction_time=_TIME,
        valid_from=_TIME,
        title=f"Current slot {slot_index}",
        l0_text=f"Current preference {slot_index}",
        l1_text=f"Current canonical value {slot_index}",
        source_uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}/claims/{claim_id}",
        source_digest=f"current-digest-{slot_index:04d}",
        source_revision=3,
        canonical_slot_id=slot_id,
        canonical_slot_uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}",
        canonical_claim_id=claim_id,
        canonical_claim_uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}/claims/{claim_id}",
        canonical_revision=1,
        canonical_state="ACTIVE",
        canonical_head_digest=f"head-{slot_index:04d}",
        receipt_digest=f"receipt-{slot_index:04d}",
        projection_effect_hash=f"effect-{slot_index:04d}",
        metadata={"memory_type": "preference", "canonical_value": slot_index},
    )


def _claim_revision(slot_index: int, claim_suffix: str, revision: int) -> CatalogRecord:
    slot_id = f"slot-{slot_index:04d}"
    claim_id = f"claim-{slot_index:04d}-{claim_suffix}"
    state = "ACTIVE" if claim_suffix == "b" else "SUPERSEDED"
    return CatalogRecord(
        record_key=f"claim:{claim_id}:revision:{revision}",
        uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}/claims/{claim_id}",
        tenant_id="tenant-a",
        owner_user_id="u1",
        context_type="memory",
        source_kind="canonical_projection",
        record_kind=CatalogRecordKind.CLAIM_REVISION.value,
        tree_paths=(f"memories/preferences/user-{slot_index:04d}/dimension",),
        created_at=_TIME,
        updated_at=_TIME,
        transaction_time=_TIME,
        valid_from=_TIME,
        title=f"Claim {claim_id} revision {revision}",
        l0_text=f"Claim {claim_suffix}",
        l1_text=f"Canonical history value {slot_index}-{claim_suffix}-{revision}",
        source_uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}/claims/{claim_id}",
        source_digest=f"claim-digest-{slot_index:04d}-{claim_suffix}-{revision}",
        source_revision=revision,
        canonical_slot_id=slot_id,
        canonical_slot_uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}",
        canonical_claim_id=claim_id,
        canonical_claim_uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}/claims/{claim_id}",
        canonical_revision=revision,
        canonical_state=state,
        metadata={"memory_type": "preference", "canonical_value": f"{slot_index}-{claim_suffix}"},
    )


def test_sqlite_catalog_handles_1000_sessions_10000_tools_and_1000_slot_histories(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = SQLiteIndexStore(tmp_path / "scale.sqlite3")

    records: list[CatalogRecord] = []
    records.extend(_session_root(index) for index in range(1_000))
    records.extend(_tool_result(index) for index in range(10_000))
    for index in range(1_000):
        records.extend(
            (
                _current_slot(index),
                _claim_revision(index, "a", 1),
                _claim_revision(index, "a", 2),
                _claim_revision(index, "b", 1),
            )
        )
    for start in range(0, len(records), 500):
        assert store.upsert_catalog_batch(records[start : start + 500]) == len(records[start : start + 500])

    filters = {
        "tenant_id": "tenant-a",
        "principal_owner_id": "u1",
        "workspace_access_ids": ("", "memoryOS"),
        "context_types": ("session",),
        "source_kinds": ("tool_result",),
        "target_paths": ("resources/repository",),
        "event_time_from": "2026-07-14T00:00:00+00:00",
        "event_time_to": "2026-07-15T00:00:00+00:00",
    }
    hits = store.search_catalog("boundedneedle09999", filters=filters, limit=25)
    assert hits
    assert hits[0].metadata["catalog_record_key"] == "session:session-0999:tool:09999"
    assert len(hits) <= 25

    with sqlite3.connect(store.path) as conn:
        counts = dict(conn.execute("SELECT record_kind, count(*) FROM contexts GROUP BY record_kind"))
        current_duplicates = conn.execute(
            "SELECT count(*) FROM (SELECT canonical_slot_id FROM contexts "
            "WHERE record_kind = 'current_slot' GROUP BY tenant_id, canonical_slot_id HAVING count(*) > 1)"
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
    assert counts[CatalogRecordKind.CURRENT_SLOT.value] == 1_000
    assert counts[CatalogRecordKind.CLAIM_REVISION.value] == 3_000
    assert current_duplicates == 0

    query_plan = " ".join(store.explain_structured_query(filters, limit=25))
    assert "idx_context_path_acl_workspace_event" in query_plan
    assert "idx_contexts_record_key" in query_plan
    assert "INDEX" in index_plans["tenant"]
    assert "idx_contexts_tenant_owner_type" in index_plans["owner"]
    assert "idx_context_paths_type_path" in index_plans["context_type"]
    assert "idx_context_paths_path_valid" in index_plans["path"]
    assert "idx_contexts_tenant_event_time" in index_plans["event_time"]
    assert "idx_contexts_tenant_transaction_time" in index_plans["transaction_time"]
    current_slot_plan = " ".join(
        store.explain_structured_query(
            {
                "tenant_id": "tenant-a",
                "canonical_slot_ids": ("slot-0001",),
                "record_kinds": (CatalogRecordKind.CURRENT_SLOT.value,),
            },
            limit=1,
        )
    )
    assert "uq_contexts_current_slot" in current_slot_plan
    exact_current = store.list_catalog(
        filters={
            "tenant_id": "tenant-a",
            "canonical_slot_ids": ("slot-0001",),
            "record_kinds": (CatalogRecordKind.CURRENT_SLOT.value,),
        },
        limit=1,
    )
    assert [record.record_key for record in exact_current] == ["slot:slot-0001:current"]

    source = FileSystemSourceStore(tmp_path / "source", tenant_id="tenant-a")
    vector = _NoEnumerationVector()
    vector.reject_enumeration = True

    def prohibited(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("large online retrieval attempted a prohibited full scan")

    monkeypatch.setattr(source, "list_objects", prohibited)
    monkeypatch.setattr(Path, "glob", prohibited)
    monkeypatch.setattr(Path, "rglob", prohibited)
    orchestrator = UnifiedRetrievalOrchestrator(
        ContextDB(source, store, InMemoryRelationStore()),
        vector_store=vector,
        embedding_provider=_Embedding(),
    )
    result = orchestrator.execute(
        RetrievalQueryPlan(
            semantic_query="boundedneedle09999",
            target_paths=("resources/repository",),
            context_types=(ContextType.SESSION,),
            source_kinds=("tool_result",),
            tenant_id="tenant-a",
            owner_user_id="u1",
            workspace_ids=("memoryOS",),
            event_time_from="2026-07-14T00:00:00+00:00",
            event_time_to="2026-07-15T00:00:00+00:00",
            query_intent=RetrievalQueryIntent.OPEN_RECALL,
            canonical_resolution_mode=CanonicalResolutionMode.DISABLED,
            candidate_limit=25,
            final_limit=10,
            token_budget=1_024,
        )
    )
    assert result.contexts
    assert result.contexts[0]["metadata"]["catalog_record_key"] == "session:session-0999:tool:09999"
    assert result.metrics.structured_candidates == 0
    assert result.metrics.exact_candidates == 0
    assert result.metrics.fts_candidates == 1
    assert result.metrics.canonical_validated <= 25
    assert result.metrics.source_reads <= 28
    assert result.metrics.vector_overfetch <= 200
    assert result.metrics.selected_count <= 10
    assert vector.max_candidate_count <= 25


def test_exact_slot_identity_bounds_10000_revisions_before_current_and_history_top_k(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = SQLiteIndexStore(tmp_path / "same-slot-history.sqlite3")
    current = _current_slot(9_999)
    slot_uri = current.canonical_slot_uri
    private_scope = {
        "visibility": {
            "tenant_id": "tenant-a",
            "private": True,
            "allowed_principal_ids": ("u1",),
            "allowed_service_ids": (),
        }
    }
    current = replace(
        current,
        uri=f"{slot_uri}/serving/current",
        metadata={**dict(current.metadata), "scope": private_scope},
    )
    history = tuple(
        replace(
            record,
            metadata={**dict(record.metadata), "scope": private_scope},
        )
        for record in (
            _claim_revision(9_999, "history", revision) for revision in range(1, 10_001)
        )
    )
    store.upsert_catalog(current)
    for start in range(0, len(history), 500):
        assert store.upsert_catalog_batch(history[start : start + 500]) == len(
            history[start : start + 500]
        )

    real_list_catalog = store.list_catalog
    identity_calls: list[tuple[dict[str, Any], int]] = []

    def recording_list_catalog(
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        copied = dict(filters or {})
        if copied.get("target_identity_uris"):
            identity_calls.append((copied, limit))
        return real_list_catalog(filters=copied, limit=limit)

    monkeypatch.setattr(store, "list_catalog", recording_list_catalog)
    # An unbounded 10k-row identity materialization exceeds this ceiling;
    # the indexed branch-local Top-K remains comfortably below it.
    store.online_vm_step_limit = 50_000
    common: dict[str, Any] = {
        "semantic_query": "",
        "target_uris": (slot_uri,),
        "context_types": (ContextType.MEMORY,),
        "tenant_id": "tenant-a",
        "owner_user_id": "u1",
        "canonical_resolution_mode": CanonicalResolutionMode.DISABLED,
        "token_budget": 1_024,
    }
    current_result = CandidateGenerator(store).generate(
        RetrievalQueryPlan(
            **common,
            query_intent=RetrievalQueryIntent.CURRENT,
            candidate_limit=3,
            final_limit=3,
        )
    )
    current_filters, current_limit = identity_calls[-1]
    assert [candidate.record_key for candidate in current_result.branches["exact"]] == [
        current.record_key
    ]
    assert current_result.exact_candidates == 1
    assert current_limit == 3
    assert current_filters["record_kinds"] == (CatalogRecordKind.CURRENT_SLOT.value,)
    assert current_filters["_identity_candidate_limit"] == 3

    identity_calls.clear()
    history_result = CandidateGenerator(store).generate(
        RetrievalQueryPlan(
            **common,
            query_intent=RetrievalQueryIntent.HISTORY,
            candidate_limit=7,
            final_limit=7,
        )
    )
    history_filters, history_limit = identity_calls[-1]
    assert len(history_result.branches["exact"]) == 7
    assert all(
        candidate.record_kind == CatalogRecordKind.CLAIM_REVISION.value
        for candidate in history_result.branches["exact"]
    )
    assert history_result.exact_candidates == 7
    assert history_limit == 7
    assert history_filters["record_kinds"] == (CatalogRecordKind.CLAIM_REVISION.value,)
    assert history_filters["_identity_candidate_limit"] == 7

    for exact_filters, limit in ((current_filters, 3), (history_filters, 7)):
        explain = store.explain_structured_query(exact_filters, limit=limit)
        joined = " ".join(explain)
        assert "idx_contexts_tenant_canonical_slot_uri" in joined
        assert "idx_context_acl_grants_record" in joined
        assert not any("SCAN identity_" in detail for detail in explain)


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
    assert store.upsert_catalog_batch((*crowded, authorized)) == 902

    hits = store.search_catalog(
        "shared-exact-identity",
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
        tenant_id="tenant-a",
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
        tenant_id="tenant-a",
    )
    tool = cast(SQLiteIndexStore, client.index_store).list_catalog(
        filters={"tenant_id": "tenant-a", "record_kinds": (CatalogRecordKind.TOOL_RESULT.value,)},
        limit=10,
    )[0]
    vector.upsert_vector(tool.uri, [1.0, 1.0], {"tenant_id": "tenant-a"})
    vector.reject_enumeration = True

    def prohibited(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("online retrieval attempted a prohibited full scan")

    monkeypatch.setattr(client.source_store, "list_objects", prohibited)
    monkeypatch.setattr(Path, "glob", prohibited)
    monkeypatch.setattr(Path, "rglob", prohibited)

    options = RetrievalOptions(
        target_paths=("resources/repository",),
        context_types=(ContextType.SESSION,),
        source_kinds=("tool_result",),
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_ids=("memoryOS",),
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        canonical_resolution_mode=CanonicalResolutionMode.DISABLED,
        candidate_limit=25,
        final_limit=10,
        token_budget=1_024,
    )
    hits = client.search_context(
        "bounded-report.txt",
        options=options,
        user_id="u1",
        project_id="memoryOS",
        tenant_id="tenant-a",
    )
    trace = client.recall_trace(client.last_recall_trace_id)

    assert hits
    assert vector.max_candidate_count <= options.candidate_limit
    assert int(trace["canonical_validated"]) <= options.candidate_limit
    assert int(trace["source_reads"]) <= options.candidate_limit
    assert int(trace["vector_overfetch"]) <= 200
    assert int(trace["selected_count"]) <= options.final_limit
    assert len([item for item in hits if dict(item["metadata"]).get("session_id") == "bounded-session"]) <= 5
