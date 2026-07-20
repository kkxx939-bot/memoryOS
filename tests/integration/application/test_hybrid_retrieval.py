from __future__ import annotations

import json
import logging

import pytest

from behavior.core.model.behavior_pattern import BehaviorPattern
from behavior.core.model.observation import Observation
from behavior.projection import behavior_pattern_to_context_object
from behavior.retrieval.similar_behavior_retriever import SimilarBehaviorRetriever
from foundation.identity import LocalUserContext
from infrastructure.context.retrieval.hybrid_search import HybridSearch
from infrastructure.store.action_policy import ActionPolicyDecisionLedger
from infrastructure.store.contracts.index import IndexHit
from infrastructure.store.contracts.vector import VectorHit, vector_row_id
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from openApi.sdk.client import MemoryOSClient
from policy.action_policy.decision.engine import PredictionEngine
from policy.action_policy.decision.request import PredictionRequest
from policy.action_policy.model.action_policy import ActionPolicy
from policy.action_policy.retrieval import ActionPolicyRetriever
from pre.evidence import ScopeRef
from tests.support.embedding import DeterministicEmbeddingProvider
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore, InMemoryVectorStore


class BrokenProvider(DeterministicEmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("provider down")


class BrokenVectorStore(InMemoryVectorStore):
    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10):  # noqa: ANN201
        raise RuntimeError("vector db down")


class FixedEmbeddingProvider:
    model_name = "fixed"
    dimension = 2

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [1.0, 0.0]


def _resource(uri: str, title: str, *, owner: str | None = "u1", metadata: dict | None = None) -> ContextObject:
    return ContextObject(
        uri=uri,
        context_type=ContextType.RESOURCE,
        title=title,
        owner_user_id=owner,
        metadata=dict(metadata or {}),
    )


def _upsert(index: InMemoryIndexStore, obj: ContextObject, content: str) -> None:
    index.upsert_index(obj, content=content, tenant_id="default")


def _upsert_vector(
    vector: InMemoryVectorStore,
    obj: ContextObject,
    embedding: list[float],
    *,
    metadata: dict | None = None,
) -> str:
    record_key = f"test:{obj.context_type.value}:{obj.uri}"
    row_id = vector_row_id("default", record_key)
    vector.upsert_vector(
        row_id,
        embedding,
        metadata={
            "tenant_id": "default",
            "catalog_record_key": record_key,
            "record_key": record_key,
            "public_uri": obj.uri,
            "source_uri": obj.uri,
            "owner_user_id": str(obj.owner_user_id or ""),
            "context_type": obj.context_type.value,
            "title": obj.title,
            "namespace": f"memoryos://user/{obj.owner_user_id}/" if obj.owner_user_id else "",
            **dict(metadata or {}),
        },
    )
    return row_id


def test_hybrid_search_uses_explicit_tenant_and_falls_back_to_lexical_index(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    obj = _resource("memoryos://user/u1/resources/hot-room", "hot room")
    source.write_object(obj, content="hot room")
    _upsert(index, obj, "hot room")

    with pytest.raises(ValueError, match="explicit tenant_id"):
        HybridSearch(index, source_store=source).search("hot", filters={"owner_user_id": "u1"})

    hits = HybridSearch(index, source_store=source).search(
        "hot",
        filters={"tenant_id": "default", "owner_user_id": "u1"},
        context_type=ContextType.RESOURCE,
    )
    assert [hit.uri for hit in hits] == [obj.uri]
    assert hits[0].source == "index"


def test_hybrid_scope_filter_keeps_same_asset_id_isolated_by_parent_path(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()

    def scoped(suffix: str, parent: str) -> ContextObject:
        return _resource(
            f"memoryos://user/u1/resources/{suffix}",
            "camera calibration",
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

    first = scoped("camera-a", "workspace-a")
    second = scoped("camera-b", "workspace-b")
    for obj in (first, second):
        source.write_object(obj, content="camera calibration")
        _upsert(index, obj, "camera calibration")
    first_key = ScopeRef("memoryos", "asset", "camera", parent_path=("workspace-a",)).key

    hits = HybridSearch(index, source_store=source).search(
        "camera",
        filters={
            "tenant_id": "default",
            "owner_user_id": "u1",
            "applicability_scope_keys": [first_key],
        },
        context_type=ContextType.RESOURCE,
    )
    assert [hit.uri for hit in hits] == [first.uri]


@pytest.mark.parametrize(
    ("vector_store", "provider", "message"),
    [
        (InMemoryVectorStore(), BrokenProvider(), "provider down"),
        (BrokenVectorStore(), DeterministicEmbeddingProvider(), "vector db down"),
    ],
)
def test_vector_failure_logs_and_returns_lexical_hits(tmp_path, caplog, vector_store, provider, message) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    obj = _resource("memoryos://user/u1/resources/fallback", "hot room")
    source.write_object(obj, content="hot room")
    _upsert(index, obj, "hot room")

    caplog.set_level(logging.WARNING, logger="infrastructure.context.retrieval.hybrid_search")
    hits = HybridSearch(index, vector_store, provider, source).search(
        "hot",
        filters={"tenant_id": "default", "owner_user_id": "u1"},
        context_type=ContextType.RESOURCE,
    )

    assert [hit.uri for hit in hits] == [obj.uri]
    assert hits[0].source == "index"
    assert message in caplog.text


def test_vector_rows_use_hashed_storage_identity_and_public_uri_owner_isolation(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = FixedEmbeddingProvider()
    own = _resource("memoryos://user/u1/resources/vector", "own vector")
    other = _resource("memoryos://user/u2/resources/vector", "other vector", owner="u2")
    for obj in (own, other):
        source.write_object(obj, content=obj.title)
        _upsert_vector(vector, obj, [1.0, 0.0])

    hits = HybridSearch(index, vector, provider, source).search(
        "anything",
        filters={"tenant_id": "default", "owner_user_id": "u1"},
        context_type=ContextType.RESOURCE,
    )

    assert [hit.uri for hit in hits] == [own.uri]
    assert hits[0].source == "vector"
    assert hits[0].metadata["vector_storage_id"].startswith("memoryos-vector://v1/")


def test_document_projection_vector_rebinds_to_public_block_uri(tmp_path) -> None:
    vector = InMemoryVectorStore()
    provider = FixedEmbeddingProvider()
    client = MemoryOSClient(str(tmp_path), vector_store=vector, embedding_provider=provider)
    caller = LocalUserContext(
        user_id="u1",
    )
    remembered = client.remember(
        "I like pistachio gelato",
        target_hint="preference",
        caller=caller,
    )
    run = client.runtime.memory.projection_worker.process_pending(limit=20)
    assert run.processed
    block_uri = next(
        str(metadata["public_uri"])
        for _, metadata in vector.rows.values()
        if str(metadata.get("public_uri") or "").startswith(f"{remembered['document_uri']}/blocks/")
    )

    hits = HybridSearch(client.runtime.stores.index, vector, provider).search(
        "vector-only-query",
        filters={
            "tenant_id": "default",
            "owner_user_id": "u1",
            "allowed_uris": (block_uri,),
        },
    )

    assert [hit.uri for hit in hits] == [block_uri]
    assert hits[0].metadata["vector_storage_id"].startswith("memoryos-vector://v1/")


@pytest.mark.parametrize("malformed_score", [float("nan"), True])
def test_malformed_vector_score_is_excluded(malformed_score) -> None:  # noqa: ANN001
    class MalformedVectorStore(InMemoryVectorStore):
        def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]:
            del embedding, namespace, limit
            return [
                VectorHit(
                    uri="not-a-valid-row",
                    score=malformed_score,
                    metadata={
                        "tenant_id": "default",
                        "record_key": "bad",
                        "public_uri": "memoryos://user/u1/resources/bad",
                        "owner_user_id": "u1",
                        "context_type": "resource",
                    },
                )
            ]

    hits = HybridSearch(InMemoryIndexStore(), MalformedVectorStore(), FixedEmbeddingProvider()).search(
        "anything",
        filters={"tenant_id": "default", "owner_user_id": "u1"},
        context_type=ContextType.RESOURCE,
    )
    assert hits == []


def test_non_active_source_is_excluded_from_stale_index_and_vector_hits(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    pending = _resource("memoryos://user/u1/resources/pending", "pending PostgreSQL")
    pending.lifecycle_state = LifecycleState.PENDING
    source.write_object(pending, content="PostgreSQL")
    _upsert(index, pending, "PostgreSQL")
    _upsert_vector(vector, pending, [1.0, 0.0])

    hits = HybridSearch(index, vector, FixedEmbeddingProvider(), source).search(
        "PostgreSQL",
        filters={"tenant_id": "default", "owner_user_id": "u1"},
        context_type=ContextType.RESOURCE,
    )
    assert hits == []


def test_high_index_score_without_base_relevance_is_excluded() -> None:
    class MalformedIndex(InMemoryIndexStore):
        def search(self, query: str, *, tenant_id: str, filters: dict | None = None, limit: int = 10):  # noqa: ANN201
            del query, tenant_id, filters, limit
            return [
                IndexHit(
                    uri="memoryos://resources/no-relevance",
                    score=1000.0,
                    context_type="resource",
                    metadata={
                        "tenant_id": "default",
                        "context_type": "resource",
                        "retrieval_scores": {"lexical": 0.0, "vector": 0.0, "identity": 0.0},
                    },
                )
            ]

    assert HybridSearch(MalformedIndex()).search(
        "query",
        filters={"tenant_id": "default"},
        context_type=ContextType.RESOURCE,
    ) == []


def test_behavior_and_action_policy_retrievers_accept_vector_candidates(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = DeterministicEmbeddingProvider()
    hybrid = HybridSearch(index, vector, provider, source)
    pattern = BehaviorPattern(
        user_id="u1",
        scene_key="hot_room",
        trigger_conditions={"context_tags": ["home"]},
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
        case_refs=["c1", "c2", "c3"],
        action_distribution=[{"action": "turn_on_ac", "count": 3}],
    )
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot_room",
        action="turn_on_ac",
        support_anchor_uri=pattern.support_anchor_uri,
    )
    pattern_obj = behavior_pattern_to_context_object(pattern)
    policy_obj = policy.to_context_object()
    source.write_object(pattern_obj, content="hot room behavior")
    source.write_object(policy_obj, content=json.dumps(policy.to_dict()))
    _upsert_vector(vector, pattern_obj, provider.embed("hot room behavior"))
    _upsert_vector(vector, policy_obj, provider.embed("hot room turn_on_ac"))

    similar = SimilarBehaviorRetriever(index, source_store=source, hybrid_search=hybrid).retrieve(
        "u1",
        Observation(user_id="u1", raw_text="hot room", location="home"),
    )
    policies = ActionPolicyRetriever(index, source, hybrid_search=hybrid).retrieve(
        "u1",
        ["turn_on_ac"],
        scene_key="hot_room",
    )

    assert similar["patterns"][0]["uri"] == pattern.uri
    assert policies[0].uri == policy.uri


def test_prediction_engine_ignores_vector_provider_failure(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot_room",
        action="turn_on_ac",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
    )
    obj = policy.to_context_object()
    source.write_object(obj, content=json.dumps(policy.to_dict()))
    _upsert(index, obj, "hot_room turn_on_ac")

    result = PredictionEngine(
        index,
        ActionPolicyDecisionLedger(tmp_path),
        source_store=source,
        vector_store=InMemoryVectorStore(),
        embedding_provider=BrokenProvider(),
    ).process(
        PredictionRequest(
            user_id="u1",
            episode_id="e1",
            observation={"scene_key": "hot_room", "raw_text": "hot room"},
            available_actions=["turn_on_ac"],
        )
    )

    assert result.candidates[0].action == "turn_on_ac"
