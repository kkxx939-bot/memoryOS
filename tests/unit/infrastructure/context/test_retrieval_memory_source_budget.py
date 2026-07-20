"""上下文检索候选预算测试。"""

from pathlib import Path

from infrastructure.context.orchestrator import UnifiedRetrievalOrchestrator
from infrastructure.context.retrieval.fusion import RetrievalCandidate, RetrievalScore
from infrastructure.context.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from infrastructure.store.model.context.context_type import ContextType
from tests.support.persistence import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
)


def test_memory_candidate_is_dropped_when_source_read_budget_is_exhausted(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path / "source")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    orchestrator = UnifiedRetrievalOrchestrator(
        index,
        source_store=source,
        relation_store=relations,
        queue_store=None,
        session_archive_store=None,
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
    )

    hydrated, reads, degraded_modes, dropped, memory_validated = orchestrator.hydrator.hydrate(
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
