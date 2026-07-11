"""预测模块里的预测引擎。"""

from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.ranking.action_policy_ranker import ActionPolicyRanker
from memoryos.action_policy.retrieval import ActionPolicyRetriever
from memoryos.behavior.retrieval.similar_behavior_retriever import SimilarBehaviorRetriever
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.source_store import IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder
from memoryos.prediction.pipeline.observation_normalizer import ObservationNormalizer
from memoryos.prediction.pipeline.policy_gate import PolicyGate
from memoryos.providers.embedding import EmbeddingProvider


class PredictionEngine:
    def __init__(
        self,
        index_store: IndexStore,
        ledger: PredictionLedger,
        source_store: SourceStore | None = None,
        relation_store: RelationStore | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        hybrid_search: HybridSearch | None = None,
    ) -> None:
        self.index_store = index_store
        self.ledger = ledger
        self.source_store = source_store
        self.relation_store = relation_store
        self.hybrid_search = hybrid_search or (
            HybridSearch(index_store, vector_store=vector_store, embedding_provider=embedding_provider, source_store=source_store)
            if vector_store is not None and embedding_provider is not None
            else None
        )
        self.observation_normalizer = ObservationNormalizer()
        self.similar_behavior_retriever = SimilarBehaviorRetriever(index_store, source_store=source_store, relation_store=relation_store, hybrid_search=self.hybrid_search)
        self.action_policy_retriever = ActionPolicyRetriever(index_store, source_store, hybrid_search=self.hybrid_search) if source_store is not None else None
        self.action_policy_ranker = ActionPolicyRanker()
        self.action_context_builder = ActionContextBuilder(index_store, source_store=source_store, relation_store=relation_store)
        self.policy_gate = PolicyGate()

    def process(self, request: PredictionRequest, policies: list[ActionPolicy] | None = None) -> PredictionResult:
        observation = self.observation_normalizer.normalize(request.user_id, request.observation)
        similar = self.similar_behavior_retriever.retrieve(request.user_id, observation)
        if policies is None:
            policies = (
                self.action_policy_retriever.retrieve(
                    request.user_id,
                    request.available_actions,
                    scene_key=observation.scene_key,
                )
                if self.action_policy_retriever is not None
                else []
            )
        available = {action for action in request.available_actions}
        scoped_policies = [policy for policy in policies if policy.action in available]
        candidates = self.action_policy_ranker.rank(
            scoped_policies,
            similarity_scores=similar["similarity_scores"],
        )
        action_context = self.action_context_builder.build(
            user_id=request.user_id,
            top_candidates=candidates[:4],
            policies=scoped_policies,
            token_budget=request.token_budget,
            resources=request.resources,
            skills=request.skills,
        )
        policy_by_uri = {policy.uri: policy for policy in scoped_policies}
        top = candidates[0] if candidates else None
        decision = self.policy_gate.evaluate(
            top,
            action_context,
            policy_by_uri.get(top.policy_uri) if top else None,
            prediction_confidence=top.score if top else 0.0,
        )
        result = PredictionResult(
            request_id=request.request_id,
            episode_id=request.episode_id,
            observation=observation,
            candidates=candidates,
            action_context=action_context,
            decision=decision,
            memory_operations=[],
        )
        self.ledger.record(result)
        return result
