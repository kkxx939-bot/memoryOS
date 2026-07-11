from __future__ import annotations

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.local_stores import InMemoryIndexStore, InMemoryQueueStore, InMemoryRelationStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.memory.model.memory import Memory, MemoryCandidate, MemoryKind
from memoryos.memory.service.memory_updater import MemoryUpdater
from memoryos.operations.commit.operation_committer import OperationCommitter


def _db(tmp_path):
    from memoryos.contextdb.store.local_stores import FileSystemSourceStore

    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relation = InMemoryRelationStore()
    return ContextDB(
        source,
        index,
        relation,
        queue_store=InMemoryQueueStore(),
        committer=OperationCommitter(source, index, str(tmp_path), relation_store=relation),
    )


def _commit_memory(db: ContextDB, memory: Memory) -> str:
    operation = MemoryUpdater().add_memory(memory)
    db.commit_operation(operation)
    return memory.uri


def test_retrieval_view_default_scope_cross_adapter(tmp_path) -> None:
    db = _db(tmp_path)
    _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/rules/r1",
            user_id="u1",
            title="MemoryOS rule",
            content="MemoryOS must keep raw tool output private.",
            kind=MemoryKind.POLICY,
            memory_type="project_rule",
            retrieval_views=["project:memoryOS:rules"],
            admission={"decision": "accept"},
            source={"adapter_id": "codex"},
            merge_key="rule:raw",
        ),
    )

    hits = ContextAssembler(db).search(
        "raw tool output",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="default",
        project_id="memoryOS",
        adapter_id="openclaw",
    )

    assert [hit["uri"] for hit in hits] == ["memoryos://user/u1/memories/rules/r1"]


def test_agent_private_not_cross_adapter(tmp_path) -> None:
    db = _db(tmp_path)
    _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/private/p1",
            user_id="u1",
            title="Codex private",
            content="Codex scratchpad implementation note.",
            kind=MemoryKind.EXPLICIT,
            memory_type="agent_experience",
            retrieval_views=["agent:codex:private"],
            admission={"decision": "accept", "private": True},
            source={"adapter_id": "codex"},
            merge_key="private:1",
        ),
    )

    hits = ContextAssembler(db).search(
        "scratchpad",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="default",
        project_id="memoryOS",
        adapter_id="openclaw",
    )

    assert hits == []


def test_no_project_id_does_not_leak_all_project_rules(tmp_path) -> None:
    db = _db(tmp_path)
    _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/rules/r1",
            user_id="u1",
            title="Project rule",
            content="Project alpha must use strict source checks.",
            kind=MemoryKind.POLICY,
            memory_type="project_rule",
            retrieval_views=["project:alpha:rules"],
            admission={"decision": "accept"},
            merge_key="rule:alpha",
        ),
    )

    hits = ContextAssembler(db).search(
        "strict source",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="default",
        adapter_id="openclaw",
    )

    assert hits == []


def test_pending_candidate_not_shared_by_default(tmp_path) -> None:
    db = _db(tmp_path)
    _commit_memory(
        db,
        MemoryCandidate(
            uri="memoryos://user/u1/memories/candidates/c1",
            user_id="u1",
            title="Candidate rule",
            content="MemoryOS might adopt candidate review.",
            kind=MemoryKind.CANDIDATE,
            memory_type="project_rule",
            retrieval_views=["project:memoryOS:rules"],
            admission={"decision": "pending", "reason": "needs_review"},
            merge_key="candidate:1",
        ),
    )

    default_hits = ContextAssembler(db).search(
        "candidate review",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="default",
        project_id="memoryOS",
        adapter_id="openclaw",
    )
    candidate_hits = ContextAssembler(db).search(
        "candidate review",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="candidates",
        project_id="memoryOS",
        adapter_id="openclaw",
    )

    assert default_hits == []
    assert [hit["uri"] for hit in candidate_hits] == ["memoryos://user/u1/memories/candidates/c1"]


def test_source_view_scan_excludes_inactive_lifecycle_before_scoring_and_limit(tmp_path) -> None:  # noqa: ANN001
    db = _db(tmp_path)
    for index in range(20):
        memory = Memory(
            uri=f"memoryos://user/u1/memories/rules/inactive-{index}",
            user_id="u1",
            title=f"inactive matching rule {index}",
            content="target phrase",
            kind=MemoryKind.POLICY,
            memory_type="project_rule",
            retrieval_views=["project:memoryOS:rules"],
            admission={"decision": "accept"},
            merge_key=f"inactive-{index}",
        )
        uri = _commit_memory(db, memory)
        obj = db.source_store.read_object(uri)
        obj.lifecycle_state = LifecycleState.OBSOLETE
        db.source_store.write_object(obj, content=memory.content)
    active_uri = _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/rules/active-target",
            user_id="u1",
            title="active target",
            content="target phrase",
            kind=MemoryKind.POLICY,
            memory_type="project_rule",
            retrieval_views=["project:memoryOS:rules"],
            admission={"decision": "accept"},
            merge_key="active-target",
        ),
    )
    hits = ContextAssembler(db).search(
        "target phrase",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="default",
        project_id="memoryOS",
        limit=1,
    )
    assert [item["uri"] for item in hits] == [active_uri]


def test_source_view_scan_filters_tenant_before_limit(tmp_path) -> None:  # noqa: ANN001
    db = _db(tmp_path)
    other_uri = _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/rules/other-tenant",
            user_id="u1",
            title="target target target",
            content="target phrase",
            kind=MemoryKind.POLICY,
            memory_type="project_rule",
            retrieval_views=["project:memoryOS:rules"],
            admission={"decision": "accept"},
            merge_key="other-tenant",
        ),
    )
    other = db.source_store.read_object(other_uri)
    other.tenant_id = "other"
    db.source_store.write_object(other, content="target phrase")
    target_uri = _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/rules/default-tenant",
            user_id="u1",
            title="target",
            content="target phrase",
            kind=MemoryKind.POLICY,
            memory_type="project_rule",
            retrieval_views=["project:memoryOS:rules"],
            admission={"decision": "accept"},
            merge_key="default-tenant",
        ),
    )
    hits = ContextAssembler(db).search(
        "target",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="default",
        project_id="memoryOS",
        tenant_id="default",
        limit=1,
    )
    assert [item["uri"] for item in hits] == [target_uri]


def test_candidate_confirm_reject_promote(tmp_path) -> None:
    db = _db(tmp_path)
    updater = MemoryUpdater()
    uri = _commit_memory(
        db,
        MemoryCandidate(
            uri="memoryos://user/u1/memories/candidates/c1",
            user_id="u1",
            title="Candidate preference",
            content="User prefers source-confirmed reports.",
            kind=MemoryKind.CANDIDATE,
            memory_type="preference",
            retrieval_views=["user:u1:preferences"],
            admission={"decision": "pending", "reason": "needs_review"},
            merge_key="candidate:pref",
        ),
    )

    db.commit_operation(updater.confirm_candidate(user_id="u1", candidate_uri=uri))
    confirmed = ContextAssembler(db).search(
        "source-confirmed",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="default",
        adapter_id="codex",
    )
    db.commit_operation(updater.reject_candidate(user_id="u1", candidate_uri=uri, reason="bad"))
    rejected = ContextAssembler(db).search(
        "source-confirmed",
        user_id="u1",
        context_type=ContextType.MEMORY,
        search_scope="default",
        adapter_id="codex",
    )

    assert [hit["uri"] for hit in confirmed] == [uri]
    assert rejected == []


def test_embedding_provider_noop_fallback(tmp_path) -> None:
    db = _db(tmp_path)
    _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/preferences/p1",
            user_id="u1",
            title="Review preference",
            content="User prefers concise review findings.",
            kind=MemoryKind.EXPLICIT,
            memory_type="preference",
            retrieval_views=["user:u1:preferences"],
            admission={"decision": "accept"},
            merge_key="pref:review",
        ),
    )

    hits = ContextAssembler(db).search("concise review", user_id="u1", context_type=ContextType.MEMORY, limit=5)

    assert hits
    assert hits[0]["uri"] == "memoryos://user/u1/memories/preferences/p1"


def test_fake_embedding_provider_interface_called(tmp_path) -> None:
    db = _db(tmp_path)
    uri = _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/preferences/p1",
            user_id="u1",
            title="Vector preference",
            content="User prefers vector-ready tests.",
            kind=MemoryKind.EXPLICIT,
            memory_type="preference",
            retrieval_views=["user:u1:preferences"],
            admission={"decision": "accept"},
            merge_key="pref:vector",
        ),
    )

    class FakeEmbeddingProvider:
        model_name = "fake"
        dimension = 2

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> list[float]:  # noqa: ARG002
            self.calls += 1
            return [1.0, 0.0]

    provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    vector_store.upsert_vector(uri, [1.0, 0.0], metadata={"owner_user_id": "u1", "context_type": "memory"})
    search = HybridSearch(db.index_store, vector_store=vector_store, embedding_provider=provider, source_store=db.source_store)

    hits = search.search("anything", filters={"owner_user_id": "u1"}, context_type=ContextType.MEMORY, limit=5)

    assert provider.calls == 1
    assert hits[0].uri == uri


def test_fake_reranker_can_reorder_results(tmp_path) -> None:
    db = _db(tmp_path)
    _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/preferences/a",
            user_id="u1",
            title="A",
            content="review",
            kind=MemoryKind.EXPLICIT,
            memory_type="preference",
            retrieval_views=["user:u1:preferences"],
            admission={"decision": "accept"},
            merge_key="a",
        ),
    )
    _commit_memory(
        db,
        Memory(
            uri="memoryos://user/u1/memories/preferences/b",
            user_id="u1",
            title="B",
            content="review",
            kind=MemoryKind.EXPLICIT,
            memory_type="preference",
            retrieval_views=["user:u1:preferences"],
            admission={"decision": "accept"},
            merge_key="b",
        ),
    )

    class ReverseReranker:
        def __init__(self) -> None:
            self.called = False

        def rerank(self, query, items):  # noqa: ANN001, ANN201
            self.called = True
            return list(reversed(items))

    reranker = ReverseReranker()
    hits = ContextAssembler(db, reranker=reranker).search("review", user_id="u1", context_type=ContextType.MEMORY, limit=2)

    assert reranker.called is True
    assert [hit["uri"] for hit in hits] == [
        "memoryos://user/u1/memories/preferences/b",
        "memoryos://user/u1/memories/preferences/a",
    ]
