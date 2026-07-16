from __future__ import annotations

import json
import logging
from typing import Any, cast

import pytest

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.retrieval import ActionPolicyRetriever
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.retrieval.similar_behavior_retriever import SimilarBehaviorRetriever
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.contextdb.store.vector_store import InMemoryVectorStore, VectorHit
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine
from memoryos.providers.embedding import HashingEmbeddingProvider


class BrokenProvider(HashingEmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("provider down")


class BrokenVectorStore(InMemoryVectorStore):
    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10):
        raise RuntimeError("vector db down")


class FixedEmbeddingProvider:
    model_name = "fixed"
    dimension = 2

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [1.0, 0.0]


def test_hybrid_search_falls_back_to_index_without_vector(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    obj = ContextObject(
        uri="memoryos://user/u1/memories/m1", context_type=ContextType.MEMORY, title="hot", owner_user_id="u1"
    )
    source.write_object(obj, content="hot room")
    index.upsert_index(obj, content="hot room")

    hits = HybridSearch(index, source_store=source).search(
        "hot", filters={"owner_user_id": "u1"}, namespace="memoryos://user/u1/", context_type=ContextType.MEMORY
    )
    assert hits[0].uri == obj.uri
    assert hits[0].source == "index"


def test_hybrid_and_assembler_keep_same_asset_ids_isolated_by_parent_path(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()

    def scoped(uri: str, parent: str) -> ContextObject:
        return ContextObject(
            uri=uri,
            context_type=ContextType.MEMORY,
            title="camera calibration",
            owner_user_id="u1",
            metadata={
                "retrieval_views": ["test:camera"],
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
                },
            },
        )

    first = scoped("memoryos://user/u1/memories/camera-a", "workspace-a")
    second = scoped("memoryos://user/u1/memories/camera-b", "workspace-b")
    for obj in (first, second):
        source.write_object(obj, content="camera calibration")
        index.upsert_index(obj, content="camera calibration")
    first_key = ScopeRef("memoryos", "asset", "camera", parent_path=("workspace-a",)).key
    hybrid = HybridSearch(index, source_store=source)

    hits = hybrid.search(
        "camera",
        filters={"owner_user_id": "u1", "applicability_scope_keys": [first_key]},
        context_type=ContextType.MEMORY,
    )
    assert [hit.uri for hit in hits] == [first.uri]

    assembler = ContextAssembler(
        ContextDB(source, index, InMemoryRelationStore()),
        hybrid_search=hybrid,
    )
    assert [
        item["uri"]
        for item in assembler.search(
            "camera",
            user_id="u1",
            context_type=ContextType.MEMORY,
            limit=10,
            retrieval_views=["test:camera"],
            project_id="",
            tenant_id="default",
            applicability_scope_keys=[first_key],
        )
    ] == [first.uri]


def test_mixed_valid_and_malformed_scope_is_not_retrievable(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    obj = ContextObject(
        uri="memoryos://user/u1/memories/malformed-scope",
        context_type=ContextType.MEMORY,
        title="mixed malformed scope",
        owner_user_id="u1",
        metadata={
            "scope": {
                "applicability": {
                    "all_of": [
                        {"namespace": "memoryos", "kind": "workspace", "id": "w1"},
                        {"namespace": "memoryos", "kind": "location"},
                    ]
                }
            }
        },
    )
    source.write_object(obj, content="mixed malformed scope")
    index.upsert_index(obj, content="mixed malformed scope")
    filters = {
        "owner_user_id": "u1",
        "applicability_scope_keys": ["memoryos:workspace:w1"],
    }

    assert not index.search("mixed", filters=filters)
    assert not HybridSearch(index, source_store=source).search(
        "mixed",
        filters=filters,
        context_type=ContextType.MEMORY,
    )
    assembler = ContextAssembler(ContextDB(source, index, InMemoryRelationStore()))
    assert (
        assembler.search(
            "mixed",
            user_id="u1",
            context_type=ContextType.MEMORY,
            limit=10,
            project_id="",
            tenant_id="default",
            applicability_scope_keys=["memoryos:workspace:w1"],
        )
        == []
    )


def test_hybrid_search_logs_embedding_failure_and_returns_index_hits(tmp_path, caplog) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    obj = ContextObject(
        uri="memoryos://user/u1/memories/m1", context_type=ContextType.MEMORY, title="hot", owner_user_id="u1"
    )
    source.write_object(obj, content="hot room")
    index.upsert_index(obj, content="hot room")

    caplog.set_level(logging.WARNING, logger="memoryos.contextdb.retrieval.hybrid_search")
    hits = HybridSearch(index, vector, BrokenProvider(), source).search(
        "hot",
        filters={"owner_user_id": "u1"},
        namespace="memoryos://user/u1/",
        context_type=ContextType.MEMORY,
    )

    assert hits[0].uri == obj.uri
    assert hits[0].source == "index"
    assert "HybridSearch vector branch failed; falling back to lexical search: provider down" in caplog.text


def test_hybrid_search_logs_vector_store_failure_and_returns_index_hits(tmp_path, caplog) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    provider = HashingEmbeddingProvider()
    obj = ContextObject(
        uri="memoryos://user/u1/memories/m1", context_type=ContextType.MEMORY, title="hot", owner_user_id="u1"
    )
    source.write_object(obj, content="hot room")
    index.upsert_index(obj, content="hot room")

    caplog.set_level(logging.WARNING, logger="memoryos.contextdb.retrieval.hybrid_search")
    hits = HybridSearch(index, BrokenVectorStore(), provider, source).search(
        "hot",
        filters={"owner_user_id": "u1"},
        namespace="memoryos://user/u1/",
        context_type=ContextType.MEMORY,
    )

    assert hits[0].uri == obj.uri
    assert hits[0].source == "index"
    assert "HybridSearch vector branch failed; falling back to lexical search: vector db down" in caplog.text


def test_vector_only_hit_and_namespace_isolation(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = HashingEmbeddingProvider()
    obj = ContextObject(
        uri="memoryos://user/u1/memories/m1", context_type=ContextType.MEMORY, title="hot", owner_user_id="u1"
    )
    other = ContextObject(
        uri="memoryos://user/u2/memories/m1", context_type=ContextType.MEMORY, title="hot other", owner_user_id="u2"
    )
    source.write_object(obj, content="hot room")
    source.write_object(other, content="hot room")
    vector.upsert_vector(
        obj.uri, provider.embed("hot room"), metadata={"owner_user_id": "u1", "context_type": "memory", "title": "hot"}
    )
    vector.upsert_vector(
        other.uri,
        provider.embed("hot room"),
        metadata={"owner_user_id": "u2", "context_type": "memory", "title": "hot other"},
    )

    hits = HybridSearch(index, vector, provider, source).search(
        "hot room", filters={"owner_user_id": "u1"}, namespace="memoryos://user/u1/", context_type=ContextType.MEMORY
    )
    assert [hit.uri for hit in hits] == [obj.uri]
    assert hits[0].source == "vector"
    assert hits[0].metadata["retrieval_scores"]["vector"] > HybridSearch.DEFAULT_MIN_VECTOR_SIMILARITY


def test_real_canonical_projection_rebinds_hashed_vector_row_to_public_uri(tmp_path) -> None:
    vector = InMemoryVectorStore()
    provider = FixedEmbeddingProvider()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vector,
        embedding_provider=provider,
    )
    remembered = client.remember(
        user_id="u1",
        memory_type="preference",
        content="I like pistachio gelato",
        identity_fields={"subject": "food", "dimension": "gelato_flavor"},
    )
    claim_uri = str(remembered["uri"])
    assert any(row_id.startswith("memoryos-vector://v1/") for row_id in vector.vector_uris())
    hybrid = HybridSearch(
        client.index_store,
        vector,
        provider,
        client.source_store,
    )

    allowlisted = hybrid.search(
        "vector-only-query-with-no-lexical-overlap",
        filters={
            "tenant_id": "default",
            "owner_user_id": "u1",
            "allowed_uris": (claim_uri,),
        },
        namespace="memoryos://user/u1/",
        context_type=ContextType.MEMORY,
    )
    namespaced = hybrid.search(
        "another-vector-only-query",
        filters={"tenant_id": "default", "owner_user_id": "u1"},
        namespace="memoryos://user/u1/",
        context_type=ContextType.MEMORY,
    )

    assert [hit.uri for hit in allowlisted] == [claim_uri]
    assert claim_uri in {hit.uri for hit in namespaced}
    assert all(hit.uri.startswith("memoryos://") for hit in (*allowlisted, *namespaced))


def test_zero_or_below_threshold_vector_similarity_is_not_a_candidate(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = FixedEmbeddingProvider()
    orthogonal = ContextObject(
        uri="memoryos://user/u1/memories/orthogonal",
        context_type=ContextType.MEMORY,
        title="orthogonal",
        owner_user_id="u1",
        hotness=1.0,
        semantic_hotness=1.0,
        behavior_support_hotness=1.0,
    )
    weak = ContextObject(
        uri="memoryos://user/u1/memories/weak",
        context_type=ContextType.MEMORY,
        title="weak",
        owner_user_id="u1",
    )
    source.write_object(orthogonal, content="unrelated")
    source.write_object(weak, content="also unrelated")
    vector.upsert_vector(orthogonal.uri, [0.0, 1.0], metadata={"owner_user_id": "u1", "context_type": "memory"})
    vector.upsert_vector(weak.uri, [0.1, 0.995], metadata={"owner_user_id": "u1", "context_type": "memory"})

    hits = HybridSearch(index, vector, provider, source).search(
        "PostgreSQL",
        filters={"owner_user_id": "u1"},
        context_type=ContextType.MEMORY,
    )

    assert hits == []


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True, "invalid"])
def test_vector_similarity_threshold_must_be_finite_and_bounded(tmp_path, threshold) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="min_vector_similarity must be a finite number between 0 and 1"):
        HybridSearch(
            InMemoryIndexStore(), source_store=FileSystemSourceStore(tmp_path), min_vector_similarity=threshold
        )


def test_high_index_score_without_base_relevance_and_nan_scores_are_excluded(tmp_path) -> None:
    objects = []
    for suffix in ("hot-zero", "nan-score", "nan-base"):
        obj = ContextObject(
            uri=f"memoryos://user/u1/memories/{suffix}",
            context_type=ContextType.MEMORY,
            title=suffix,
            owner_user_id="u1",
        )
        objects.append(obj)

    class MalformedIndex(InMemoryIndexStore):
        def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
            return [
                IndexHit(
                    uri=objects[0].uri,
                    score=1000.0,
                    context_type="memory",
                    metadata={
                        "owner_user_id": "u1",
                        "context_type": "memory",
                        "retrieval_scores": {
                            "lexical": 0.0,
                            "vector": 0.0,
                            "identity": 0.0,
                            "hotness": 1.0,
                        },
                    },
                ),
                IndexHit(
                    uri=objects[1].uri,
                    score=float("nan"),
                    context_type="memory",
                    metadata={
                        "owner_user_id": "u1",
                        "context_type": "memory",
                        "retrieval_scores": {"lexical": 1.0, "vector": 0.0, "identity": 0.0},
                    },
                ),
                IndexHit(
                    uri=objects[2].uri,
                    score=1.0,
                    context_type="memory",
                    metadata={
                        "owner_user_id": "u1",
                        "context_type": "memory",
                        "retrieval_scores": {"lexical": float("nan"), "vector": 0.0, "identity": 0.0},
                    },
                ),
            ]

    hits = HybridSearch(MalformedIndex()).search(
        "PostgreSQL",
        filters={"owner_user_id": "u1"},
        context_type=ContextType.MEMORY,
    )

    assert hits == []


@pytest.mark.parametrize("malformed_score", [float("nan"), True])
def test_malformed_vector_score_is_excluded_even_when_metadata_is_hot(tmp_path, malformed_score) -> None:  # noqa: ANN001
    obj = ContextObject(
        uri="memoryos://user/u1/memories/nan-vector",
        context_type=ContextType.MEMORY,
        title="nan vector",
        owner_user_id="u1",
    )

    class MalformedVectorStore(InMemoryVectorStore):
        def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]:
            return [
                VectorHit(
                    uri=obj.uri,
                    score=malformed_score,
                    metadata={
                        "owner_user_id": "u1",
                        "context_type": "memory",
                        "hotness": 1.0,
                    },
                )
            ]

    hits = HybridSearch(InMemoryIndexStore(), MalformedVectorStore(), FixedEmbeddingProvider()).search(
        "PostgreSQL",
        filters={"owner_user_id": "u1"},
        context_type=ContextType.MEMORY,
    )

    assert hits == []


def test_user_owned_hit_without_owner_is_excluded_without_source_store() -> None:
    class MissingOwnerIndex(InMemoryIndexStore):
        def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
            return [
                IndexHit(
                    uri="memoryos://user/u2/memories/private",
                    score=1.0,
                    context_type="memory",
                    metadata={
                        "context_type": "memory",
                        "retrieval_scores": {"lexical": 1.0, "vector": 0.0, "identity": 0.0},
                    },
                )
            ]

    hits = HybridSearch(MissingOwnerIndex()).search(
        "private",
        filters={"owner_user_id": "u1"},
        context_type=ContextType.MEMORY,
    )

    assert hits == []


def test_context_assembler_without_principal_excludes_user_memory_but_keeps_explicit_global_resources(
    tmp_path,
) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    db = ContextDB(source, index, InMemoryRelationStore())
    u1 = ContextObject(
        uri="memoryos://user/u1/memories/private",
        context_type=ContextType.MEMORY,
        title="shared search term u1",
        owner_user_id="u1",
    )
    u2 = ContextObject(
        uri="memoryos://user/u2/memories/private",
        context_type=ContextType.MEMORY,
        title="shared search term u2",
        owner_user_id="u2",
    )
    resource = ContextObject(
        uri="memoryos://resources/shared-search-term",
        context_type=ContextType.RESOURCE,
        title="shared search term resource",
    )
    for obj in (u1, u2, resource):
        source.write_object(obj, content=obj.title)
        index.upsert_index(obj, content=obj.title)

    assembler = ContextAssembler(db)

    assert assembler.search("shared search term", context_type=ContextType.MEMORY) == []
    assert [
        hit["uri"]
        for hit in assembler.search(
            "shared search term",
            user_id="u1",
            context_type=ContextType.MEMORY,
        )
    ] == [u1.uri]
    assert [hit["uri"] for hit in assembler.search("shared search term", context_type=ContextType.RESOURCE)] == [
        resource.uri
    ]


def test_context_assembler_without_principal_does_not_call_unified_retrieval(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    assembler = ContextAssembler(ContextDB(source, index, InMemoryRelationStore()))

    class LeakOrchestrator:
        def __init__(self) -> None:
            self.called = False

        def execute(self, plan):  # noqa: ANN001, ANN201
            self.called = True
            raise AssertionError(f"unscoped memory query reached Unified retrieval: {plan!r}")

    leak_orchestrator = LeakOrchestrator()
    cast(Any, assembler).unified_retrieval = leak_orchestrator

    assert assembler.search("private", context_type=ContextType.MEMORY) == []
    assert leak_orchestrator.called is False


def test_source_store_metadata_overrides_stale_vector_metadata(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = FixedEmbeddingProvider()
    active = ContextObject(
        uri="memoryos://user/u1/memories/rule",
        context_type=ContextType.MEMORY,
        title="source title",
        owner_user_id="u1",
        metadata={
            "marker": "source",
            "memory_type": "project_rule",
            "admission": {"decision": "accept"},
            "scope": {"project_id": "alpha"},
        },
    )
    source.write_object(active, content="PostgreSQL")
    vector.upsert_vector(
        active.uri,
        [1.0, 0.0],
        metadata={
            "owner_user_id": "u1",
            "context_type": "memory",
            "marker": "stale-vector",
            "memory_type": "preference",
            "admission": {"decision": "reject"},
            "scope": {"project_id": "beta"},
        },
    )

    hits = HybridSearch(index, vector, provider, source).search(
        "PostgreSQL",
        filters={
            "owner_user_id": "u1",
            "project_id": "alpha",
            "memory_type": "project_rule",
        },
        context_type=ContextType.MEMORY,
    )

    assert [hit.uri for hit in hits] == [active.uri]
    assert hits[0].title == "source title"
    assert hits[0].metadata["marker"] == "source"
    assert hits[0].metadata["memory_type"] == "project_rule"
    assert hits[0].metadata["admission"] == {"decision": "accept"}


def test_non_active_source_is_excluded_from_stale_index_and_vector_hits(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    vector = InMemoryVectorStore()
    provider = FixedEmbeddingProvider()
    pending = ContextObject(
        uri="memoryos://user/u1/memories/pending/p1",
        context_type=ContextType.MEMORY,
        title="pending PostgreSQL",
        owner_user_id="u1",
        lifecycle_state=LifecycleState.PENDING,
        metadata={"canonical_kind": "pending_proposal", "admission": {"decision": "pending"}},
    )
    source.write_object(pending, content="PostgreSQL")

    class StaleIndex(InMemoryIndexStore):
        def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
            return [
                IndexHit(
                    uri=pending.uri,
                    score=1.0,
                    context_type="memory",
                    title=pending.title,
                    metadata={
                        "admission": {"decision": "accept"},
                        "retrieval_scores": {"lexical": 1.0, "vector": 0.0, "identity": 0.0},
                    },
                )
            ]

    vector.upsert_vector(
        pending.uri,
        [1.0, 0.0],
        metadata={
            "owner_user_id": "u1",
            "context_type": "memory",
            "admission": {"decision": "accept"},
        },
    )

    hits = HybridSearch(StaleIndex(), vector, provider, source).search(
        "PostgreSQL",
        filters={"owner_user_id": "u1"},
        context_type=ContextType.MEMORY,
    )

    assert hits == []


def test_vector_allowed_uri_filter_overfetches_before_limit(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = FixedEmbeddingProvider()
    blocked = ContextObject(
        uri="memoryos://user/u1/memories/blocked",
        context_type=ContextType.MEMORY,
        title="blocked",
        owner_user_id="u1",
    )
    allowed = ContextObject(
        uri="memoryos://user/u1/memories/allowed",
        context_type=ContextType.MEMORY,
        title="allowed",
        owner_user_id="u1",
    )
    source.write_object(blocked, content="blocked")
    source.write_object(allowed, content="allowed")
    vector.upsert_vector(blocked.uri, [1.0, 0.0], metadata={"owner_user_id": "u1", "context_type": "memory"})
    vector.upsert_vector(allowed.uri, [0.9, 0.1], metadata={"owner_user_id": "u1", "context_type": "memory"})

    hits = HybridSearch(index, vector, provider, source).search(
        "anything",
        filters={"owner_user_id": "u1", "allowed_uris": [allowed.uri]},
        context_type=ContextType.MEMORY,
        limit=1,
    )

    assert [hit.uri for hit in hits] == [allowed.uri]


def test_index_and_vector_same_uri_merge_score(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = HashingEmbeddingProvider()
    obj = ContextObject(
        uri="memoryos://user/u1/memories/m1", context_type=ContextType.MEMORY, title="hot", owner_user_id="u1"
    )
    source.write_object(obj, content="hot room")
    index.upsert_index(obj, content="hot room")
    vector.upsert_vector(
        obj.uri, provider.embed("hot room"), metadata={"owner_user_id": "u1", "context_type": "memory", "title": "hot"}
    )

    hit = HybridSearch(index, vector, provider, source).search(
        "hot room", filters={"owner_user_id": "u1"}, namespace="memoryos://user/u1/", context_type=ContextType.MEMORY
    )[0]
    assert hit.uri == obj.uri
    assert hit.source == "hybrid"
    assert hit.score > 0


def test_behavior_and_action_policy_retrievers_return_vector_hits(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = HashingEmbeddingProvider()
    hybrid = HybridSearch(index, vector, provider, source)
    pattern = BehaviorPattern(
        user_id="u1",
        scene_key="hot_room",
        trigger_conditions={"context_tags": ["home"]},
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        case_refs=["c1", "c2", "c3"],
        action_distribution=[{"action": "turn_on_ac", "count": 3}],
    )
    policy = ActionPolicy(
        user_id="u1", scene_key="hot_room", action="turn_on_ac", memory_anchor_uri=pattern.memory_anchor_uri
    )
    source.write_object(pattern.to_context_object(), content="hot room behavior")
    source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    vector.upsert_vector(
        pattern.uri,
        provider.embed("hot room behavior"),
        metadata={"owner_user_id": "u1", "context_type": "behavior_pattern", "title": "pattern"},
    )
    vector.upsert_vector(
        policy.uri,
        provider.embed("hot room turn_on_ac"),
        metadata={"owner_user_id": "u1", "context_type": "action_policy", "title": "policy"},
    )

    similar = SimilarBehaviorRetriever(index, source_store=source, hybrid_search=hybrid).retrieve(
        "u1", Observation(user_id="u1", raw_text="hot room", location="home")
    )
    assert similar["patterns"][0]["uri"] == pattern.uri
    policies = ActionPolicyRetriever(index, source, hybrid_search=hybrid).retrieve(
        "u1", ["turn_on_ac"], scene_key="hot_room"
    )
    assert policies[0].uri == policy.uri


def test_prediction_engine_ignores_vector_provider_failure(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot_room",
        action="turn_on_ac",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
    )
    source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    index.upsert_index(policy.to_context_object(), content="hot_room turn_on_ac")

    result = PredictionEngine(
        index, PredictionLedger(tmp_path), source_store=source, vector_store=vector, embedding_provider=BrokenProvider()
    ).process(
        PredictionRequest(
            user_id="u1",
            episode_id="e1",
            observation={"scene_key": "hot_room", "raw_text": "hot room"},
            available_actions=["turn_on_ac"],
        )
    )

    assert result.candidates[0].action == "turn_on_ac"
    assert result.memory_operations == []
