from pathlib import Path

from memoryos.application.context.orchestrator import UnifiedRetrievalOrchestrator
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.fusion import RetrievalCandidate, RetrievalScore
from memoryos.contextdb.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
)


def test_memory_candidate_is_dropped_when_source_read_budget_is_exhausted(tmp_path: Path) -> None:
    orchestrator = UnifiedRetrievalOrchestrator(
        ContextDB(
            FileSystemSourceStore(tmp_path / "source"),
            InMemoryIndexStore(),
            InMemoryRelationStore(),
        )
    )
    candidate = RetrievalCandidate(
        record_key="memory-document:u1:memdoc_budget",
        uri="memoryos://user/u1/memory/documents/memdoc_budget",
        title="Budgeted memory",
        context_type=ContextType.MEMORY.value,
        record_kind="memory_document",
        source_uri="memoryos://user/u1/memory/documents/memdoc_budget",
        source_digest="a" * 64,
        tenant_id="default",
        owner_user_id="u1",
        document_id="memdoc_budget",
        l0_text="unverified serving summary",
        score=RetrievalScore(final_score=0.9),
        metadata={"relative_path": "knowledge/topics/budget.md"},
    )
    plan = RetrievalQueryPlan(
        semantic_query="budget",
        context_types=(ContextType.MEMORY,),
        tenant_id="default",
        owner_user_id="u1",
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        candidate_limit=10,
        final_limit=5,
        token_budget=256,
    )

    hydrated, reads, degraded_modes, dropped, memory_validated = orchestrator._hydrate(
        (candidate,),
        plan=plan,
        source_read_budget=0,
    )

    assert hydrated == ()
    assert reads == 0
    assert degraded_modes == ("memory_source_read_bound",)
    assert memory_validated == 0
    assert len(dropped) == 1
    assert dropped[0]["drop_reason"] == "memory_source_read_bound"
