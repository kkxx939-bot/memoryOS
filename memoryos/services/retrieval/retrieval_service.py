from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from memoryos.ports.providers.rerank_provider import RerankProvider
from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.services.retrieval.behavior_context_builder import BehaviorContext, BehaviorContextBuilder
from memoryos.services.retrieval.memory_context_builder import MemoryContext, MemoryContextBuilder


@dataclass
class RetrievalResult:
    memory_context: MemoryContext
    behavior_context: BehaviorContext = field(default_factory=BehaviorContext)
    behavior_patterns: list[dict] = field(default_factory=list)
    behavior_distribution: list[dict] = field(default_factory=list)
    query_plan: dict = field(default_factory=dict)

    def memories_for_prediction(self) -> list[dict]:
        return self.memory_context.memories_for_prediction()

    def route_trace(self) -> list[dict]:
        return [
            *[route.to_dict() for route in self.memory_context.route_trace],
            *[route.to_dict() for route in self.behavior_context.route_trace],
        ]

    def source_summary(self) -> dict[str, dict]:
        summary = self.memory_context.source_summary()
        summary.update(self.behavior_context.source_summary())
        return summary

    def to_dict(self) -> dict:
        return {
            "memory_context": self.memory_context.to_dict(),
            "behavior_context": {
                "route_trace": [route.to_dict() for route in self.behavior_context.route_trace],
                "source_summary": self.behavior_context.source_summary(),
            },
            "query_plan": self.query_plan,
            "behavior_patterns": self.behavior_patterns,
            "behavior_distribution": self.behavior_distribution,
            "route_trace": self.route_trace(),
            "source_summary": self.source_summary(),
        }


class RetrievalOrchestrator:
    def __init__(
        self,
        store: MemoryRepository,
        behavior_stats_path: Path,
        rerank_provider: RerankProvider | None = None,
    ) -> None:
        self.store = store
        self.behavior_stats_path = behavior_stats_path
        self.rerank_provider = rerank_provider

    def retrieve(
        self,
        user_id: str,
        query: str,
        context_tags: list[str],
        retrieval_limit: int = 8,
    ) -> RetrievalResult:
        memory_context = MemoryContextBuilder(self.store).build(
            user_id=user_id,
            query=query,
            relevant_limit_per_type=max(1, retrieval_limit // 3),
        )
        behavior_context = BehaviorContextBuilder(
            self.store.root,
            self.behavior_stats_path,
            rerank_provider=self.rerank_provider or self.store.rerank_provider,
        ).build(
            user_id=user_id,
            query=query,
            context_tags=context_tags,
        )
        query_plan = {
            "query": query,
            "context_tags": context_tags,
            "mode": "directory_first_memory_then_behavior",
            "steps": [
                "stable_profile_policy_preference",
                "recent_feedback_intervention_event",
                "L0_L1_directory_route",
                "L2_memory_hybrid_search",
                "behavior_pattern_recall",
                "behavior_feedback_recall",
            ],
            "target_directories": [
                route.target_uri
                for route in memory_context.route_trace
                if route.strategy == "directory_first_relevant_memory"
            ],
            "behavior_routes": [route.to_dict() for route in behavior_context.route_trace],
        }
        return RetrievalResult(
            memory_context=memory_context,
            behavior_context=behavior_context,
            behavior_patterns=behavior_context.behavior_patterns,
            behavior_distribution=behavior_context.behavior_distribution,
            query_plan=query_plan,
        )
