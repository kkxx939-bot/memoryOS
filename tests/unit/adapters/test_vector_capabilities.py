from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from memoryos.adapters.vector.chroma_store import ChromaStore
from memoryos.adapters.vector.errors import VectorBackendUnavailableError
from memoryos.adapters.vector.milvus_store import MilvusStore
from memoryos.adapters.vector.qdrant_store import QdrantStore
from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind, catalog_vector_metadata
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.candidate_generator import CandidateGenerator
from memoryos.contextdb.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import (
    InMemoryVectorStore,
    VectorCapabilities,
    VectorHit,
    require_production_vector_capabilities,
    vector_capabilities,
    vector_row_id,
)


def test_local_vector_store_declares_bounded_fallback_not_native_filtering() -> None:
    store = InMemoryVectorStore()

    capabilities = vector_capabilities(store)

    assert capabilities.supports_namespace_filtering
    assert not capabilities.production_filtered_top_k_ready
    with pytest.raises(ValueError, match="supports_metadata_filtering"):
        require_production_vector_capabilities(store)


@pytest.mark.parametrize("store_type", [QdrantStore, MilvusStore, ChromaStore])
def test_unimplemented_named_adapter_selection_fails_fast(store_type: type) -> None:
    assert store_type is not InMemoryVectorStore
    with pytest.raises(VectorBackendUnavailableError, match="not implemented"):
        store_type()


def test_native_capability_contract_accepts_only_explicit_full_backend() -> None:
    class FilteredStore(InMemoryVectorStore):
        @property
        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(
                supports_metadata_filtering=True,
                supports_namespace_filtering=True,
                supports_time_filtering=True,
                supports_delete_by_filter=True,
            )

        def search_vector_filtered(
            self,
            embedding: list[float],
            *,
            namespace: str,
            filters: Mapping[str, object],
            limit: int = 10,
        ) -> list[VectorHit]:
            del embedding, namespace, filters, limit
            return []

    assert require_production_vector_capabilities(FilteredStore()).production_filtered_top_k_ready


def test_native_filtered_vector_branch_can_recall_without_lexical_seed(tmp_path) -> None:  # noqa: ANN001
    class Embedding:
        model_name = "test-filtered"
        dimension = 2

        def embed(self, text: str) -> list[float]:
            assert text == "zzvectorconcept"
            return [1.0, 0.0]

    class FilteredStore(InMemoryVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.seen_namespace = ""
            self.seen_filters: dict[str, object] = {}
            self.filtered_before_top_k: tuple[str, ...] = ()

        @property
        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(
                supports_metadata_filtering=True,
                supports_namespace_filtering=True,
                supports_time_filtering=True,
                supports_delete_by_filter=True,
            )

        @staticmethod
        def _string_values(value: object) -> tuple[str, ...]:
            if not isinstance(value, list | tuple | set | frozenset):
                return ()
            return tuple(str(item) for item in value)

        def search_vector_filtered(
            self,
            embedding: list[float],
            *,
            namespace: str,
            filters: Mapping[str, object],
            limit: int = 10,
        ) -> list[VectorHit]:
            self.seen_namespace = namespace
            self.seen_filters = dict(filters)
            allowed: list[str] = []
            for uri, (_stored_embedding, metadata) in self.rows.items():
                if str(metadata.get("tenant_id") or "") != namespace:
                    continue
                if str(metadata.get("tenant_id") or "") != str(filters.get("tenant_id") or ""):
                    continue
                principal = filters.get("principal_owner_id")
                if principal and str(metadata.get("owner_user_id") or "") != str(principal):
                    continue
                workspaces = self._string_values(filters.get("workspace_access_ids", ()))
                if workspaces and str(metadata.get("workspace_id") or "") not in workspaces:
                    continue
                scopes = set(self._string_values(filters.get("applicability_scope_keys", ())))
                row_scopes = set(str(item) for item in metadata.get("scope_keys", ()) or ())
                if scopes and not row_scopes.issubset(scopes):
                    continue
                target_paths = self._string_values(filters.get("target_paths", ()))
                row_paths = tuple(str(item) for item in metadata.get("tree_paths", ()) or ())
                if target_paths and not any(
                    path == target or path.startswith(f"{target}/") for path in row_paths for target in target_paths
                ):
                    continue
                rejected_by_time = False
                for field, lower_name, upper_name in (
                    ("event_time", "event_time_from", "event_time_to"),
                    ("transaction_time", "transaction_time_from", "transaction_time_to"),
                ):
                    value = str(metadata.get(field) or "")
                    lower = str(filters.get(lower_name) or "")
                    upper = str(filters.get(upper_name) or "")
                    if (lower and value < lower) or (upper and value >= upper):
                        rejected_by_time = True
                        break
                if rejected_by_time:
                    continue
                allowed.append(uri)
            self.filtered_before_top_k = tuple(allowed)
            return self.search_vector_candidates(embedding, tuple(allowed), limit=limit)

    index = SQLiteIndexStore(tmp_path / "filtered-vector.sqlite3")
    uri = "memoryos://user/u1/sessions/history/s1/context/root"
    scope: dict[str, Any] = {
        "applicability": {
            "all_of": [
                {
                    "namespace": "memoryos",
                    "kind": "workspace",
                    "id": "project-a",
                    "parent_id": None,
                    "attributes": {},
                    "confidence": 1.0,
                    "source": "explicit",
                    "inferred": False,
                }
            ]
        }
    }
    valid = CatalogRecord(
        record_key="session:s1:root",
        uri=uri,
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_id="project-a",
        session_id="s1",
        adapter_id="codex",
        context_type="session",
        source_kind="session_root",
        record_kind=CatalogRecordKind.SESSION_ROOT.value,
        tree_paths=("projects/project-a", "sessions/s1"),
        created_at="2026-07-14T01:00:00+00:00",
        updated_at="2026-07-14T01:00:00+00:00",
        event_time="2026-07-14T01:00:00+00:00",
        ingested_at="2026-07-14T01:00:00+00:00",
        transaction_time="2026-07-14T01:00:00+00:00",
        title="no lexical overlap",
        l0_text="unrelated words",
        l1_text="catalog phrase without matching terms",
        source_uri="memoryos://user/u1/sessions/history/s1",
        source_digest="digest-s1",
        metadata={"scope": scope},
    )
    records = (
        valid,
        replace(valid, record_key="wrong-tenant", uri=f"{uri}-wrong-tenant", tenant_id="tenant-b"),
        replace(valid, record_key="wrong-owner", uri=f"{uri}-wrong-owner", owner_user_id="u2"),
        replace(valid, record_key="wrong-workspace", uri=f"{uri}-wrong-workspace", workspace_id="project-b"),
        replace(
            valid,
            record_key="wrong-scope",
            uri=f"{uri}-wrong-scope",
            metadata={
                "scope": {
                    "applicability": {
                        "all_of": [
                            {
                                **scope["applicability"]["all_of"][0],
                                "id": "project-b",
                            }
                        ]
                    }
                }
            },
        ),
        replace(
            valid,
            record_key="wrong-path",
            uri=f"{uri}-wrong-path",
            primary_tree_path="projects/project-b",
            tree_paths=("projects/project-b", "sessions/s1"),
        ),
        replace(
            valid,
            record_key="wrong-event-time",
            uri=f"{uri}-wrong-event-time",
            event_time="2026-07-13T23:59:59+00:00",
        ),
        replace(
            valid,
            record_key="wrong-transaction-time",
            uri=f"{uri}-wrong-transaction-time",
            transaction_time="2026-07-16T00:00:00+00:00",
        ),
    )
    for record in records:
        index.upsert_catalog(record, tenant_id=record.tenant_id)
    vectors = FilteredStore()
    valid_row_id = vector_row_id(valid.tenant_id, valid.record_key)
    vectors.upsert_vector(valid_row_id, [0.8, 0.2], catalog_vector_metadata(valid))
    for record in records[1:]:
        vectors.upsert_vector(
            vector_row_id(record.tenant_id, record.record_key),
            [1.0, 0.0],
            catalog_vector_metadata(record),
        )
    plan = RetrievalQueryPlan(
        semantic_query="zzvectorconcept",
        target_paths=("projects/project-a",),
        context_types=(ContextType.SESSION,),
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_ids=("project-a",),
        event_time_from="2026-07-14T00:00:00+00:00",
        event_time_to="2026-07-15T00:00:00+00:00",
        transaction_time_from="2026-07-14T00:00:00+00:00",
        transaction_time_to="2026-07-15T00:00:00+00:00",
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        candidate_limit=1,
        final_limit=1,
        metadata_filters={
            "applicability_scope_keys": ("memoryos:workspace:project-a",),
        },
    )

    generated = CandidateGenerator(
        index,
        vector_store=vectors,
        embedding_provider=Embedding(),
    ).generate(plan)

    assert generated.branches["lexical"] == ()
    assert [candidate.record_key for candidate in generated.branches["vector"]] == ["session:s1:root"]
    assert vectors.seen_namespace == "tenant-a"
    assert vectors.seen_filters["principal_owner_id"] == "u1"
    assert vectors.seen_filters["workspace_access_ids"] == ("", "project-a")
    assert vectors.seen_filters["target_paths"] == ("projects/project-a",)
    assert vectors.seen_filters["event_time_from"] == "2026-07-14T00:00:00+00:00"
    assert vectors.seen_filters["transaction_time_to"] == "2026-07-15T00:00:00+00:00"
    assert vectors.seen_filters["applicability_scope_keys"] == ("memoryos:workspace:project-a",)
    assert vectors.filtered_before_top_k == (valid_row_id,)


def test_candidate_generation_reports_fts_unavailable(tmp_path) -> None:  # noqa: ANN001
    index = SQLiteIndexStore(tmp_path / "fts-unavailable.sqlite3")
    index.fts_enabled = False
    generated = CandidateGenerator(index).generate(
        RetrievalQueryPlan(
            semantic_query="semantic query",
            tenant_id="tenant-a",
            owner_user_id="u1",
            query_intent=RetrievalQueryIntent.OPEN_RECALL,
            candidate_limit=10,
            final_limit=5,
        )
    )

    assert generated.branches["lexical"] == ()
    assert "fts_unavailable" in generated.degraded_modes
