from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.retrieval import ActionPolicyRetriever
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.retrieval.similar_behavior_retriever import SimilarBehaviorRetriever
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine
from memoryos.providers.embedding import HashingEmbeddingProvider


class BrokenProvider(HashingEmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("provider down")


def test_hybrid_search_falls_back_to_index_without_vector(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    obj = ContextObject(uri="memoryos://user/u1/memories/m1", context_type=ContextType.MEMORY, title="hot", owner_user_id="u1")
    source.write_object(obj, content="hot room")
    index.upsert_index(obj, content="hot room")

    hits = HybridSearch(index, source_store=source).search("hot", filters={"owner_user_id": "u1"}, namespace="memoryos://user/u1/", context_type=ContextType.MEMORY)
    assert hits[0].uri == obj.uri
    assert hits[0].source == "index"


def test_vector_only_hit_and_namespace_isolation(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = HashingEmbeddingProvider()
    obj = ContextObject(uri="memoryos://user/u1/memories/m1", context_type=ContextType.MEMORY, title="hot", owner_user_id="u1")
    other = ContextObject(uri="memoryos://user/u2/memories/m1", context_type=ContextType.MEMORY, title="hot other", owner_user_id="u2")
    source.write_object(obj, content="hot room")
    source.write_object(other, content="hot room")
    vector.upsert_vector(obj.uri, provider.embed("hot room"), metadata={"owner_user_id": "u1", "context_type": "memory", "title": "hot"})
    vector.upsert_vector(other.uri, provider.embed("hot room"), metadata={"owner_user_id": "u2", "context_type": "memory", "title": "hot other"})

    hits = HybridSearch(index, vector, provider, source).search("hot room", filters={"owner_user_id": "u1"}, namespace="memoryos://user/u1/", context_type=ContextType.MEMORY)
    assert [hit.uri for hit in hits] == [obj.uri]
    assert hits[0].source == "vector"


def test_index_and_vector_same_uri_merge_score(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    provider = HashingEmbeddingProvider()
    obj = ContextObject(uri="memoryos://user/u1/memories/m1", context_type=ContextType.MEMORY, title="hot", owner_user_id="u1")
    source.write_object(obj, content="hot room")
    index.upsert_index(obj, content="hot room")
    vector.upsert_vector(obj.uri, provider.embed("hot room"), metadata={"owner_user_id": "u1", "context_type": "memory", "title": "hot"})

    hit = HybridSearch(index, vector, provider, source).search("hot room", filters={"owner_user_id": "u1"}, namespace="memoryos://user/u1/", context_type=ContextType.MEMORY)[0]
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
    policy = ActionPolicy(user_id="u1", scene_key="hot_room", action="turn_on_ac", memory_anchor_uri=pattern.memory_anchor_uri)
    source.write_object(pattern.to_context_object(), content="hot room behavior")
    source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    vector.upsert_vector(pattern.uri, provider.embed("hot room behavior"), metadata={"owner_user_id": "u1", "context_type": "behavior_pattern", "title": "pattern"})
    vector.upsert_vector(policy.uri, provider.embed("hot room turn_on_ac"), metadata={"owner_user_id": "u1", "context_type": "action_policy", "title": "policy"})

    similar = SimilarBehaviorRetriever(index, source_store=source, hybrid_search=hybrid).retrieve(
        "u1", Observation(user_id="u1", raw_text="hot room", location="home")
    )
    assert similar["patterns"][0]["uri"] == pattern.uri
    policies = ActionPolicyRetriever(index, source, hybrid_search=hybrid).retrieve("u1", ["turn_on_ac"], scene_key="hot_room")
    assert policies[0].uri == policy.uri


def test_prediction_engine_ignores_vector_provider_failure(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vector = InMemoryVectorStore()
    policy = ActionPolicy(user_id="u1", scene_key="hot_room", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
    source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    index.upsert_index(policy.to_context_object(), content="hot_room turn_on_ac")

    result = PredictionEngine(index, PredictionLedger(tmp_path), source_store=source, vector_store=vector, embedding_provider=BrokenProvider()).process(
        PredictionRequest(user_id="u1", episode_id="e1", observation={"scene_key": "hot_room", "raw_text": "hot room"}, available_actions=["turn_on_ac"])
    )

    assert result.candidates[0].action == "turn_on_ac"
    assert result.memory_operations == []
