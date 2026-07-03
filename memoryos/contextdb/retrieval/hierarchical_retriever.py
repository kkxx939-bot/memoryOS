from __future__ import annotations

from dataclasses import dataclass

from memoryos.contextdb.retrieval.context_selector import ContextSelector
from memoryos.contextdb.retrieval.query_plan import QueryPlan
from memoryos.contextdb.retrieval.reranker import ContextReranker
from memoryos.contextdb.store.source_store import IndexHit, IndexStore


@dataclass(frozen=True)
class HierarchicalRetrievalResult:
    plan: QueryPlan
    l0_hits: list[IndexHit]
    l1_hits: list[IndexHit]
    l2_uris: list[str]

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "l0_hits": [hit.__dict__ for hit in self.l0_hits],
            "l1_hits": [hit.__dict__ for hit in self.l1_hits],
            "l2_uris": self.l2_uris,
        }


class HierarchicalRetriever:
    def __init__(self, index_store: IndexStore) -> None:
        self.index_store = index_store
        self.selector = ContextSelector()
        self.reranker = ContextReranker()

    def retrieve(self, plan: QueryPlan, l0_limit: int = 12, l1_limit: int = 6, l2_limit: int = 2) -> HierarchicalRetrievalResult:
        hits = []
        for context_type in plan.context_types:
            hits.extend(
                self.index_store.search(
                    plan.query,
                    filters={"owner_user_id": plan.user_id, "context_type": context_type.value},
                    limit=l0_limit,
                )
            )
        l0_hits = self.selector.select(self.reranker.rerank(hits), l0_limit)
        l1_hits = self.selector.select(l0_hits, l1_limit)
        l2_uris = [hit.uri for hit in l1_hits[:l2_limit]]
        return HierarchicalRetrievalResult(plan=plan, l0_hits=l0_hits, l1_hits=l1_hits, l2_uris=l2_uris)
