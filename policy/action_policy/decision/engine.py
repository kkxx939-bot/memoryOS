"""ActionPolicy 的在线候选检索、排序、上下文构建与决策入口。"""

from __future__ import annotations

from behavior.retrieval.similar_behavior_retriever import SimilarBehaviorRetriever
from infrastructure.context.retrieval.embedding import EmbeddingProvider
from infrastructure.context.retrieval.hybrid_search import HybridSearch
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.contracts.vector import VectorStore
from policy.action_policy.decision.context_builder import ActionContextBuilder
from policy.action_policy.decision.gate import PolicyGate
from policy.action_policy.decision.ledger import DecisionLedger
from policy.action_policy.decision.observation_normalizer import ObservationNormalizer
from policy.action_policy.decision.request import PredictionRequest
from policy.action_policy.decision.result import PredictionResult
from policy.action_policy.model.action_policy import ActionPolicy
from policy.action_policy.ranking.action_policy_ranker import ActionPolicyRanker
from policy.action_policy.retrieval import ActionPolicyRetriever
from policy.action_policy.risk import canonical_action


class PredictionEngine:
    def __init__(
        self,
        index_store: IndexStore,
        ledger: DecisionLedger,
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
        available = {canonical_action(action) for action in request.available_actions if canonical_action(action)}
        scoped_policies = [
            policy
            for policy in policies
            if policy.user_id == request.user_id and canonical_action(policy.action) in available
        ]
        tenant_id = self._tenant_id()
        verified_anchors = self.action_context_builder.verified_support_anchor_uris(
            request.user_id,
            scoped_policies,
            tenant_id=tenant_id,
        )
        candidates = self.action_policy_ranker.rank(
            scoped_policies,
            similarity_scores=similar["similarity_scores"],
            verified_support_anchor_uris=verified_anchors,
        )
        action_context = self.action_context_builder.build(
            user_id=request.user_id,
            top_candidates=candidates[:4],
            policies=scoped_policies,
            resources=request.resources,
            skills=request.skills,
            tenant_id=tenant_id,
            verified_support_anchor_uris=verified_anchors,
        )
        policy_by_uri = {policy.uri: policy for policy in scoped_policies}
        top = candidates[0] if candidates else None
        decision = self.policy_gate.evaluate(
            top,
            action_context,
            policy_by_uri.get(top.policy_uri) if top else None,
            decision_confidence=top.score if top else 0.0,
        )
        result = PredictionResult(
            request_id=request.request_id,
            episode_id=request.episode_id,
            observation=observation,
            candidates=candidates,
            action_context=action_context,
            decision=decision,
        )
        self.ledger.record(result, tenant_id=tenant_id)
        return result

    def _tenant_id(self) -> str:
        return str(getattr(self.source_store, "tenant_id", "default") or "default")
