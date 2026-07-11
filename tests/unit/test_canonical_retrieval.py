from __future__ import annotations

import json

from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
)
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.memory.canonical import (
    CanonicalMemoryProjector,
    CanonicalMemoryQuery,
    CanonicalMemoryRetriever,
    CanonicalQueryIntent,
    EvidenceRef,
    MemoryClaim,
    MemoryRevision,
    TransitionProfile,
)
from memoryos.providers.embedding import HashingEmbeddingProvider


def _scope(*refs, tenant_id: str = "t1", principals=()):  # noqa: ANN001, ANN202
    return {
        "applicability": {
            "all_of": [
                {
                    "namespace": "memoryos",
                    "kind": kind,
                    "id": identifier,
                    "parent_id": None,
                    "attributes": {},
                }
                for kind, identifier in refs
            ]
        },
        "visibility": {
            "tenant_id": tenant_id,
            "allowed_principal_ids": list(principals),
            "allowed_service_ids": [],
            "private": bool(principals),
        },
        "origin_refs": [],
    }


def _write_claim(
    source,
    projector,
    *,
    claim_id: str,
    value: str,
    state: str,
    memory_type: str,
    scope: dict,
    profile: TransitionProfile = TransitionProfile.AUTHORITATIVE_STATE,
    owner_user_id: str = "u1",
):  # noqa: ANN001, ANN202
    uri = f"memoryos://user/{owner_user_id}/memories/canonical/slots/{claim_id}-slot/claims/{claim_id}"
    revision = MemoryRevision(
        revision=1,
        state=state,
        value_fields={"canonical_value": value},
        evidence_refs=(EvidenceRef("e1", None, "hash"),),
        proposal_id=f"p-{claim_id}",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
    )
    claim = MemoryClaim(claim_id, uri, f"{claim_id}-slot", value, profile, (revision,))
    obj = claim.to_context_object(tenant_id="t1", owner_user_id=owner_user_id, memory_type=memory_type, scope=scope)
    source.write_object(obj, content=json.dumps({"value": value, "state": state}))
    projector.project(uri)
    return uri


def test_retrieval_distinguishes_current_options_history_visibility_and_scope(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    projector = CanonicalMemoryProjector(source, index, tmp_path)
    workspace_scope = _scope(("workspace", "memoryos"))
    active = _write_claim(
        source,
        projector,
        claim_id="sqlite",
        value="sqlite",
        state="ACTIVE",
        memory_type="project_decision",
        scope=workspace_scope,
    )
    proposed = _write_claim(
        source,
        projector,
        claim_id="postgres",
        value="postgresql",
        state="PROPOSED",
        memory_type="project_decision",
        scope=workspace_scope,
    )
    superseded = _write_claim(
        source,
        projector,
        claim_id="mysql",
        value="mysql",
        state="SUPERSEDED",
        memory_type="project_decision",
        scope=workspace_scope,
    )
    retriever = CanonicalMemoryRetriever(source, index, relations)

    def query(
        *,
        tenant_id: str = "t1",
        scopes: tuple[str, ...] = ("memoryos:workspace:memoryos",),
        states: tuple[str, ...] = (),
        intent: CanonicalQueryIntent | None = None,
    ) -> CanonicalMemoryQuery:
        return CanonicalMemoryQuery(
            text="",
            tenant_id=tenant_id,
            principal_id="u1",
            applicability_scope_keys=scopes,
            states=states,
            intent=intent,
            limit=10,
        )

    current = retriever.search(query(intent=CanonicalQueryIntent.CURRENT))
    assert [item["uri"] for item in current] == [active]
    options = retriever.search(query(intent=CanonicalQueryIntent.OPTIONS))
    assert {item["uri"] for item in options} == {proposed}
    history = retriever.search(query(states=("SUPERSEDED",)))
    assert [item["uri"] for item in history] == [superseded]
    assert history[0]["memory_category"] == "history"
    assert retriever.search(query(tenant_id="other", intent=CanonicalQueryIntent.OPTIONS)) == []
    assert (
        retriever.search(
            query(
                scopes=("memoryos:workspace:other",),
                intent=CanonicalQueryIntent.OPTIONS,
            )
        )
        == []
    )


def test_canonical_scope_filter_precedes_limit(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    projector = CanonicalMemoryProjector(source, index, tmp_path)
    for number in range(30):
        _write_claim(
            source,
            projector,
            claim_id=f"other-{number}",
            value=f"shared target {number}",
            state="ACTIVE",
            memory_type="project_decision",
            scope=_scope(("workspace", "other")),
        )
    target = _write_claim(
        source,
        projector,
        claim_id="target-workspace",
        value="shared target",
        state="ACTIVE",
        memory_type="project_decision",
        scope=_scope(("workspace", "memoryos")),
    )
    results = CanonicalMemoryRetriever(source, index).search(
        CanonicalMemoryQuery(
            text="shared target",
            tenant_id="t1",
            principal_id="u1",
            applicability_scope_keys=("memoryos:workspace:memoryos",),
            limit=1,
        )
    )
    assert [item["uri"] for item in results] == [target]


def test_reachy_preference_applies_to_person_and_environment_not_device_only(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    projector = CanonicalMemoryProjector(source, index, tmp_path)
    preference = _write_claim(
        source,
        projector,
        claim_id="quiet-hours",
        value="no music after 22:00",
        state="ACTIVE",
        memory_type="preference",
        scope=_scope(
            ("principal", "user_1"),
            ("environment", "home_01"),
            principals=("user_1",),
        ),
        owner_user_id="user_1",
    )
    retriever = CanonicalMemoryRetriever(source, index)
    visible = retriever.search(
        CanonicalMemoryQuery(
            text="music",
            tenant_id="t1",
            principal_id="user_1",
            applicability_scope_keys=(
                "memoryos:principal:user_1",
                "memoryos:environment:home_01",
                "memoryos:asset:reachy_01",
                "memoryos:location:home_01:kitchen",
            ),
        )
    )
    assert [item["uri"] for item in visible] == [preference]
    device_only = retriever.search(
        CanonicalMemoryQuery(
            text="music",
            tenant_id="t1",
            principal_id="user_1",
            applicability_scope_keys=("memoryos:asset:reachy_01",),
        )
    )
    assert device_only == []


def test_canonical_retrieval_supports_exact_vector_and_relation_expansion(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    embedding = HashingEmbeddingProvider()
    relations = InMemoryRelationStore()
    projector = CanonicalMemoryProjector(
        source,
        index,
        tmp_path,
        vector_store=vectors,
        embedding_provider=embedding,
    )
    scope = _scope(("workspace", "memoryos"))
    sqlite = _write_claim(
        source,
        projector,
        claim_id="sqlite-vector",
        value="sqlite durable local database",
        state="ACTIVE",
        memory_type="project_decision",
        scope=scope,
    )
    postgres = _write_claim(
        source,
        projector,
        claim_id="postgres-related",
        value="postgresql future option",
        state="PROPOSED",
        memory_type="project_decision",
        scope=scope,
    )
    cockroach = _write_claim(
        source,
        projector,
        claim_id="cockroach-related",
        value="cockroachdb distributed option",
        state="PROPOSED",
        memory_type="project_decision",
        scope=scope,
    )
    relations.add_relation(
        ContextRelation(
            source_uri=postgres,
            relation_type="alternative",
            target_uri=cockroach,
            metadata={"tenant_id": "t1", "owner_user_id": "u1"},
        )
    )
    hybrid = HybridSearch(index, vector_store=vectors, embedding_provider=embedding, source_store=source)
    retriever = CanonicalMemoryRetriever(source, index, relations, hybrid_search=hybrid)
    def query(
        text: str,
        *,
        claim_uris: tuple[str, ...] = (),
        intent: CanonicalQueryIntent = CanonicalQueryIntent.OPTIONS,
    ) -> CanonicalMemoryQuery:
        return CanonicalMemoryQuery(
            text=text,
            tenant_id="t1",
            principal_id="u1",
            applicability_scope_keys=("memoryos:workspace:memoryos",),
            intent=intent,
            claim_uris=claim_uris,
            limit=10,
        )

    exact = retriever.search(
        query("no lexical match", claim_uris=(sqlite,), intent=CanonicalQueryIntent.CURRENT)
    )
    assert exact[0]["uri"] == sqlite
    vector = retriever.search(query("durable local database", intent=CanonicalQueryIntent.CURRENT))
    assert any(item["uri"] == sqlite for item in vector)
    expanded = CanonicalMemoryRetriever(source, index, relations).search(query("postgresql"))
    assert {item["uri"] for item in expanded} >= {postgres, cockroach}
    assert any(item["retrieval_source"] == "canonical_relation_expansion" for item in expanded)
