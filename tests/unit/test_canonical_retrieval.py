from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.session_model import SessionArchive
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
    MemorySlot,
    ScopeRef,
    SessionArchiveEpisodeAdapter,
    TransitionProfile,
)
from memoryos.memory.canonical.current_head import publish_current_head_sets
from memoryos.memory.canonical.prefetch import ExistingMemoryPrefetcher
from memoryos.memory.canonical.projection_state import ProjectionRecordStore
from memoryos.memory.canonical.retrieval import CanonicalInvariantViolation
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.providers.embedding import HashingEmbeddingProvider
from memoryos.runtime.readiness import RuntimeNotReadyError, RuntimeReadinessState


def _scope(*refs, tenant_id: str = "t1", principals=()):  # noqa: ANN001, ANN202
    scope_refs = [
        {
            "namespace": "memoryos",
            "kind": kind,
            "id": identifier,
            "parent_id": None,
            "attributes": {},
            "confidence": 1.0,
            "source": "explicit",
            "inferred": False,
        }
        for kind, identifier in refs
    ]
    return {
        "canonical_subject": scope_refs[0],
        "applicability": {"all_of": scope_refs},
        "visibility": {
            "tenant_id": tenant_id,
            "allowed_principal_ids": list(principals),
            "allowed_service_ids": [],
            "private": bool(principals),
        },
        "authority": {
            "principal_ids": list(principals or ("u1",)),
            "service_ids": [],
            "inferred": False,
        },
        "origin_refs": [],
    }


def _write_committed_canonical_fixture(
    source: FileSystemSourceStore,
    entries: list[tuple[ContextObject, str]],
    *,
    key: str,
    action: OperationAction = OperationAction.ADD,
) -> None:
    """Persist canonical fixtures behind an integrity-valid transaction marker."""

    transaction_id = f"tx-{key}"
    idempotency_key = f"idem-{key}"
    commit_group_id = f"fixture-group-{key}"
    operations: list[ContextOperation] = []
    for index, (obj, content) in enumerate(entries):
        owner_user_id = obj.owner_user_id
        assert isinstance(owner_user_id, str)
        obj.metadata = {
            **dict(obj.metadata or {}),
            "canonical_transaction_id": transaction_id,
            "canonical_idempotency_key": idempotency_key,
        }
        operations.append(
            ContextOperation(
                operation_id=f"op-{key}-{index}",
                user_id=owner_user_id,
                context_type=obj.context_type,
                action=action,
                target_uri=obj.uri,
                status=OperationStatus.COMMITTED,
                payload={
                    "canonical_memory": True,
                    "transaction_id": transaction_id,
                    "idempotency_key": idempotency_key,
                    "commit_group_id": commit_group_id,
                    "tenant_id": obj.tenant_id,
                    "expected_revision": 0
                    if action == OperationAction.ADD
                    else max(0, int(obj.metadata.get("revision", 1)) - 1),
                    "context_object": obj.to_dict(),
                    "content": content,
                },
            )
        )
    assert operations
    assert len({operation.user_id for operation in operations}) == 1
    fixture_relations = InMemoryRelationStore()
    committer = OperationCommitter(
        source,
        InMemoryIndexStore(),
        str(source.root),
        relation_store=fixture_relations,
        tenant_id=source.tenant_id,
    )
    before_images = committer._capture_canonical_state(operations)
    before_by_uri = {str(item["uri"]): item.get("object") for item in before_images}
    relation_manifests = {
        operation.operation_id: committer._build_canonical_relation_manifest(
            operation,
            before_by_uri.get(str(operation.target_uri or "")),
        )
        for operation in operations
    }
    diff = ContextDiff(
        user_id=operations[0].user_id,
        operations=operations,
        diff_id=f"diff-{transaction_id}",
    )
    planning_digest = committer._ensure_canonical_planning_digest(operations)
    for operation in operations:
        operation.payload["planning_digest"] = planning_digest
    committer._write_outbox_event(
        transaction_id,
        idempotency_key,
        operations,
        status="prepared",
        before_images=before_images,
        relation_manifests=relation_manifests,
    )
    for obj, content in entries:
        source.write_object(obj, content=content)
    committer._write_outbox_event(
        transaction_id,
        idempotency_key,
        operations,
        status="source_committed",
        before_images=before_images,
        relation_manifests=relation_manifests,
    )
    marker = committer._transaction_marker(idempotency_key)
    committer._write_transaction_marker(
        marker,
        diff,
        operations,
        relation_manifests=relation_manifests,
    )
    committer._validate_transaction_marker(marker, operations)
    publish_current_head_sets(
        committer.artifact_root,
        marker,
        json.loads(marker.read_text(encoding="utf-8")),
    )


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
    object_relations: tuple[ContextRelation, ...] = (),
    metadata_extra: dict | None = None,
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
    subject = ScopeRef.from_dict(scope["canonical_subject"])
    claim = MemoryClaim(
        claim_id,
        uri,
        f"{claim_id}-slot",
        value,
        profile,
        (revision,),
        canonical_subject_key=subject.key,
    )
    obj = claim.to_context_object(tenant_id="t1", owner_user_id=owner_user_id, memory_type=memory_type, scope=scope)
    obj.relations.extend(object_relations)
    obj.metadata = {**obj.metadata, **dict(metadata_extra or {})}
    applicability = dict(scope.get("applicability", {}) or {})
    scope_keys = tuple(
        f"{item.get('namespace', 'memoryos')}:{item.get('kind')}:{item.get('id')}"
        for item in applicability.get("all_of", []) or []
    )
    slot_uri = uri.rsplit("/claims/", 1)[0]
    slot = MemorySlot(
        slot_id=f"{claim_id}-slot",
        uri=slot_uri,
        memory_type=memory_type,
        identity_fields={"test_identity": claim_id},
        scope_keys=scope_keys,
        claim_ids=(claim_id,),
        active_claim_id=claim_id if state == "ACTIVE" else None,
        revision=1,
        canonical_subject_key=subject.key,
        canonical_subject=subject,
    )
    slot_obj = slot.to_context_object(
        tenant_id="t1",
        owner_user_id=owner_user_id,
        scope=scope,
    )
    _write_committed_canonical_fixture(
        source,
        [
            (slot_obj, json.dumps({"slot_id": slot.slot_id})),
            (obj, json.dumps({"value": value, "state": state})),
        ],
        key=f"claim-{claim_id}-r1",
    )
    projector.project(uri)
    return uri


def _write_two_revision_claim(
    source,
    projector,
    *,
    old_value: str,
    new_value: str,
    project_current: bool = True,
):  # noqa: ANN001, ANN202
    claim_id = "revisioned"
    slot_id = "revisioned-slot"
    slot_uri = "memoryos://user/u1/memories/canonical/slots/revisioned-slot"
    claim_uri = f"{slot_uri}/claims/{claim_id}"
    scope = _scope(("workspace", "memoryos"))
    subject = ScopeRef.from_dict(scope["canonical_subject"])
    revision_one = MemoryRevision(
        revision=1,
        state="ACTIVE",
        value_fields={"canonical_value": old_value},
        evidence_refs=(EvidenceRef("e1", None, "hash-1"),),
        proposal_id="p1",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
    )
    claim_one = MemoryClaim(
        claim_id,
        claim_uri,
        slot_id,
        old_value,
        TransitionProfile.AUTHORITATIVE_STATE,
        (revision_one,),
        canonical_subject_key=subject.key,
    )
    slot_one = MemorySlot(
        slot_id=slot_id,
        uri=slot_uri,
        memory_type="project_decision",
        identity_fields={"decision_topic": "revision test"},
        scope_keys=("memoryos:workspace:memoryos",),
        claim_ids=(claim_id,),
        active_claim_id=claim_id,
        revision=1,
        canonical_subject_key=subject.key,
        canonical_subject=subject,
    )
    _write_committed_canonical_fixture(
        source,
        [
            (slot_one.to_context_object(tenant_id="t1", owner_user_id="u1", scope=scope), ""),
            (
                claim_one.to_context_object(
                    tenant_id="t1",
                    owner_user_id="u1",
                    memory_type="project_decision",
                    scope=scope,
                ),
                old_value,
            ),
        ],
        key="revisioned-r1",
    )
    projector.project(claim_uri, 1)

    revision_two = MemoryRevision(
        revision=2,
        state="ACTIVE",
        value_fields={"canonical_value": new_value},
        evidence_refs=(EvidenceRef("e2", None, "hash-2"),),
        proposal_id="p2",
        relation="CORRECTS",
        epistemic_status="EXPLICIT",
        previous_revision=1,
    )
    claim_two = MemoryClaim(
        claim_id,
        claim_uri,
        slot_id,
        new_value,
        TransitionProfile.AUTHORITATIVE_STATE,
        (replace(revision_one, valid_to=revision_two.valid_from), revision_two),
        canonical_subject_key=subject.key,
    )
    slot_two = replace(slot_one, revision=2)
    _write_committed_canonical_fixture(
        source,
        [
            (slot_two.to_context_object(tenant_id="t1", owner_user_id="u1", scope=scope), ""),
            (
                claim_two.to_context_object(
                    tenant_id="t1",
                    owner_user_id="u1",
                    memory_type="project_decision",
                    scope=scope,
                ),
                new_value,
            ),
        ],
        key="revisioned-r2",
        action=OperationAction.UPDATE,
    )
    if project_current:
        projector.project(claim_uri, 2)
    return claim_uri, slot_uri


def test_retrieval_distinguishes_current_options_history_visibility_and_scope(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
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
    history = retriever.search(query(states=("SUPERSEDED",), intent=CanonicalQueryIntent.HISTORY))
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
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
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
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
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
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
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
    postgres_uri = "memoryos://user/u1/memories/canonical/slots/postgres-related-slot/claims/postgres-related"
    cockroach_uri = "memoryos://user/u1/memories/canonical/slots/cockroach-related-slot/claims/cockroach-related"
    related = ContextRelation(
        source_uri=postgres_uri,
        relation_type="alternative",
        target_uri=cockroach_uri,
        metadata={
            "tenant_id": "t1",
            "owner_user_id": "u1",
            "canonical_idempotency_key": "idem-claim-postgres-related-r1",
            "canonical_transaction_id": "tx-claim-postgres-related-r1",
        },
    )
    postgres = _write_claim(
        source,
        projector,
        claim_id="postgres-related",
        value="postgresql future option",
        state="PROPOSED",
        memory_type="project_decision",
        scope=scope,
        object_relations=(related,),
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
    assert postgres == postgres_uri and cockroach == cockroach_uri
    relations.add_relation(related)
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

    exact = retriever.search(query("no lexical match", claim_uris=(sqlite,), intent=CanonicalQueryIntent.CURRENT))
    assert exact[0]["uri"] == sqlite
    vector = retriever.search(query("durable local database", intent=CanonicalQueryIntent.CURRENT))
    assert any(item["uri"] == sqlite for item in vector)
    expanded = CanonicalMemoryRetriever(source, index, relations).search(query("postgresql"))
    assert {item["uri"] for item in expanded} >= {postgres, cockroach}
    assert any(item["retrieval_source"] == "canonical_relation_expansion" for item in expanded)


def test_canonical_fallback_uses_token_boundaries_not_latin_substrings(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    projector = CanonicalMemoryProjector(source, index, tmp_path)
    claim_uri = _write_claim(
        source,
        projector,
        claim_id="redistribution",
        value="redistribution strategy",
        state="ACTIVE",
        memory_type="project_decision",
        scope=_scope(("workspace", "memoryos")),
    )
    retriever = CanonicalMemoryRetriever(source, index)
    query = CanonicalMemoryQuery(
        text="redis",
        tenant_id="t1",
        principal_id="u1",
        applicability_scope_keys=("memoryos:workspace:memoryos",),
        intent=CanonicalQueryIntent.CURRENT,
    )

    assert retriever.search(query) == []
    exact_token = retriever.search(replace(query, text="redistribution"))
    assert exact_token[0]["uri"] == claim_uri


def test_stale_index_vector_and_projection_hits_are_filtered_by_canonical_revision(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    vectors = InMemoryVectorStore()
    embedding = HashingEmbeddingProvider()
    projector = CanonicalMemoryProjector(
        source,
        index,
        tmp_path,
        vector_store=vectors,
        embedding_provider=embedding,
    )
    claim_uri, _slot_uri = _write_two_revision_claim(
        source,
        projector,
        old_value="legacyuniquetoken",
        new_value="currentuniquetoken",
        project_current=False,
    )
    hybrid = HybridSearch(index, vector_store=vectors, embedding_provider=embedding, source_store=source)
    retriever = CanonicalMemoryRetriever(source, index, hybrid_search=hybrid)
    query = CanonicalMemoryQuery(
        text="legacyuniquetoken",
        tenant_id="t1",
        principal_id="u1",
        applicability_scope_keys=("memoryos:workspace:memoryos",),
        intent=CanonicalQueryIntent.CURRENT,
    )

    assert retriever.search(query) == []
    exact = retriever.search(replace(query, text="", claim_uris=(claim_uri,)))
    assert len(exact) == 1
    assert exact[0]["revision"] == 2
    assert exact[0]["text"] == "currentuniquetoken"
    assert exact[0]["layer_texts"] == {}

    projector.project(claim_uri, 2)
    current = retriever.search(replace(query, text="current-only-token"))
    assert [item["uri"] for item in current] == [claim_uri]
    assert current[0]["projection_revision"] == 2


def test_vector_head_corruption_fails_closed_before_empty_search_trace(tmp_path) -> None:  # noqa: ANN001
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    committed = client.remember(
        user_id="u1",
        content="Always run tests before release",
        title="Release validation",
        memory_type="project_rule",
        project_id="p1",
        constraint_polarity="REQUIRE",
        identity_fields={"rule_topic": "release_validation"},
    )
    claim_uri = str(committed["uri"])
    assert claim_uri in vectors.vector_uris()

    tampered = False
    for path in (Path(tmp_path) / "system" / "current-heads").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        heads = dict(payload.get("heads", {}) or {})
        if claim_uri not in heads:
            continue
        heads[claim_uri] = {**dict(heads[claim_uri]), "head_digest": "0" * 64}
        payload["heads"] = heads
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tampered = True
        break
    assert tampered

    trace_root = Path(tmp_path) / "recall-traces"
    traces_before = tuple(trace_root.glob("*.json")) if trace_root.exists() else ()
    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        client.search_context(
            "Always run tests before release",
            user_id="u1",
            context_type=ContextType.RESOURCE,
        )

    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    traces_after = tuple(trace_root.glob("*.json")) if trace_root.exists() else ()
    assert traces_after == traces_before


def test_canonical_hybrid_recall_reuses_the_query_committed_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: ANN001
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    committed = client.remember(
        user_id="u1",
        content="Always run tests before release",
        title="Release validation",
        memory_type="project_rule",
        project_id="p1",
        constraint_polarity="REQUIRE",
        identity_fields={"rule_topic": "snapshot_release_validation"},
    )
    claim_uri = str(committed["uri"])
    assert client.hybrid_search is not None
    original_search = client.hybrid_search.search
    original_read = client.source_store.read_object
    in_canonical_hybrid = False
    canonical_calls = 0

    def guarded_search(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal in_canonical_hybrid, canonical_calls
        allowed = set(dict(kwargs.get("filters") or {}).get("allowed_uris", ()) or ())
        canonical_call = claim_uri in allowed
        if canonical_call:
            assert kwargs.get("source_snapshot", {}).get(claim_uri) is not None
            canonical_calls += 1
            in_canonical_hybrid = True
        try:
            return original_search(*args, **kwargs)
        finally:
            if canonical_call:
                in_canonical_hybrid = False

    def reject_live_canonical_reread(uri: str):  # noqa: ANN202
        if in_canonical_hybrid and uri == claim_uri:
            raise AssertionError("canonical hybrid recall re-read live Source after snapshot capture")
        return original_read(uri)

    monkeypatch.setattr(client.hybrid_search, "search", guarded_search)
    monkeypatch.setattr(client.source_store, "read_object", reject_live_canonical_reread)

    results = client.search_context(
        "Always run tests before release",
        user_id="u1",
        project_id="p1",
        context_type=ContextType.MEMORY,
    )

    assert canonical_calls == 1
    assert [item["uri"] for item in results] == [claim_uri]


def test_history_expands_old_revision_while_current_returns_only_slot_current(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    projector = CanonicalMemoryProjector(source, index, tmp_path)
    claim_uri, _slot_uri = _write_two_revision_claim(
        source,
        projector,
        old_value="historical sqlite choice",
        new_value="current postgres choice",
    )
    retriever = CanonicalMemoryRetriever(source, index)
    base = CanonicalMemoryQuery(
        text="",
        tenant_id="t1",
        principal_id="u1",
        applicability_scope_keys=("memoryos:workspace:memoryos",),
        claim_uris=(claim_uri,),
    )

    current = retriever.search(replace(base, intent=CanonicalQueryIntent.CURRENT))
    history = retriever.search(replace(base, intent=CanonicalQueryIntent.HISTORY))

    assert [(item["revision"], item["memory_state"]) for item in current] == [(2, "ACTIVE")]
    assert [(item["revision"], item["memory_state"]) for item in history] == [(1, "ACTIVE")]
    assert history[0]["revision_uri"].endswith("#revision-1")
    assert "historical sqlite choice" in history[0]["text"]
    assert history[0]["layer_revisions"] == {"L0": 1, "L1": 1, "L2": 1}

    historical_record = projector.record_store.load(claim_uri, 1)
    assert historical_record is not None
    source.write_content(historical_record.l2_uri, "tampered historical projection bait")

    after_tamper = retriever.search(replace(base, intent=CanonicalQueryIntent.HISTORY))

    assert len(after_tamper) == 1
    assert "historical sqlite choice" in after_tamper[0]["text"]
    assert "tampered historical projection bait" not in after_tamper[0]["text"]
    assert after_tamper[0]["layer"] == "canonical_source"
    assert after_tamper[0]["layer_revisions"] == {}


def test_late_historical_transaction_does_not_replace_effective_current_projection(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    projector = CanonicalMemoryProjector(source, index, tmp_path)
    scope = _scope(("workspace", "memoryos"))
    subject = ScopeRef.from_dict(scope["canonical_subject"])
    slot_uri = "memoryos://user/u1/memories/canonical/slots/late-history"
    claim_uri = f"{slot_uri}/claims/late-history"
    current_revision = MemoryRevision(
        revision=1,
        state="ACTIVE",
        value_fields={"canonical_value": "current effective rule"},
        evidence_refs=(EvidenceRef("current", None, "current-hash"),),
        proposal_id="current",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
        qualifiers={"display_fields": {"summary": "current effective display"}},
        valid_from="2026-01-02T00:00:00+00:00",
    )
    late_history = MemoryRevision(
        revision=2,
        state="ACTIVE",
        value_fields={"canonical_value": "older historical rule"},
        evidence_refs=(EvidenceRef("late", None, "late-hash"),),
        proposal_id="late",
        relation="SUPPLEMENTS",
        epistemic_status="EXPLICIT",
        qualifiers={
            "non_current_historical": True,
            "display_fields": {"summary": "older historical display"},
        },
        previous_revision=1,
        valid_from="2025-12-01T00:00:00+00:00",
        transaction_time="2026-07-11T00:00:00+00:00",
    )
    claim = MemoryClaim(
        "late-history",
        claim_uri,
        "late-history",
        "current effective rule",
        TransitionProfile.AUTHORITATIVE_STATE,
        (current_revision, late_history),
        canonical_subject_key=subject.key,
    )
    slot = MemorySlot(
        slot_id="late-history",
        uri=slot_uri,
        memory_type="project_rule",
        identity_fields={"rule_topic": "late history"},
        scope_keys=("memoryos:workspace:memoryos",),
        claim_ids=("late-history",),
        active_claim_id="late-history",
        revision=2,
        canonical_subject_key=subject.key,
        canonical_subject=subject,
    )
    revision_one_claim = MemoryClaim(
        "late-history",
        claim_uri,
        "late-history",
        "current effective rule",
        TransitionProfile.AUTHORITATIVE_STATE,
        (current_revision,),
        canonical_subject_key=subject.key,
    )
    revision_one_slot = MemorySlot(
        slot_id="late-history",
        uri=slot_uri,
        memory_type="project_rule",
        identity_fields={"rule_topic": "late history"},
        scope_keys=("memoryos:workspace:memoryos",),
        claim_ids=("late-history",),
        active_claim_id="late-history",
        revision=1,
        canonical_subject_key=subject.key,
        canonical_subject=subject,
    )
    _write_committed_canonical_fixture(
        source,
        [
            (revision_one_slot.to_context_object(tenant_id="t1", owner_user_id="u1", scope=scope), ""),
            (
                revision_one_claim.to_context_object(
                    tenant_id="t1",
                    owner_user_id="u1",
                    memory_type="project_rule",
                    scope=scope,
                ),
                "",
            ),
        ],
        key="late-history-r1",
    )
    _write_committed_canonical_fixture(
        source,
        [
            (slot.to_context_object(tenant_id="t1", owner_user_id="u1", scope=scope), ""),
            (
                claim.to_context_object(
                    tenant_id="t1",
                    owner_user_id="u1",
                    memory_type="project_rule",
                    scope=scope,
                ),
                "",
            ),
        ],
        key="late-history-r2",
        action=OperationAction.UPDATE,
    )
    projector.project(claim_uri, 2)
    retriever = CanonicalMemoryRetriever(source, index)
    query = CanonicalMemoryQuery(
        text="",
        tenant_id="t1",
        principal_id="u1",
        applicability_scope_keys=("memoryos:workspace:memoryos",),
        claim_uris=(claim_uri,),
    )

    current = retriever.search(replace(query, intent=CanonicalQueryIntent.CURRENT))
    history = retriever.search(replace(query, intent=CanonicalQueryIntent.HISTORY))

    assert current[0]["revision"] == 1
    assert current[0]["source_revision"] == 2
    assert current[0]["projection_record"]["current_claim_revision"] == 1
    assert "current effective rule" in current[0]["text"]
    assert "current effective display" in current[0]["text"]
    assert "older historical display" not in current[0]["text"]
    assert [(item["revision"], item["text"]) for item in history] == [(2, "older historical rule")]

    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="late-history-prefetch",
            archive_uri="memoryos://user/u1/sessions/history/late-history-prefetch",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "content": "current effective rule",
                    "metadata": {"memory_types": ["project_rule"]},
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        )
    )
    prefetched = ExistingMemoryPrefetcher(source, index).prefetch(episode, owner_user_id="u1")
    assert len(prefetched) == 1
    assert "current effective rule" in prefetched[0].l1
    assert "older historical rule" not in prefetched[0].l1


def test_multiple_active_claims_raise_invariant_violation_instead_of_returning_first(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    scope = _scope(("workspace", "memoryos"))
    subject = ScopeRef.from_dict(scope["canonical_subject"])
    slot_uri = "memoryos://user/u1/memories/canonical/slots/broken"
    claims = []
    entries = []
    for claim_id, value in (("c1", "sqlite"), ("c2", "postgres")):
        revision = MemoryRevision(
            revision=1,
            state="ACTIVE",
            value_fields={"canonical_value": value},
            evidence_refs=(EvidenceRef(f"e-{claim_id}", None, f"h-{claim_id}"),),
            proposal_id=f"p-{claim_id}",
            relation="ALTERNATIVE",
            epistemic_status="EXPLICIT",
        )
        claim = MemoryClaim(
            claim_id,
            f"{slot_uri}/claims/{claim_id}",
            "broken",
            value,
            TransitionProfile.AUTHORITATIVE_STATE,
            (revision,),
            canonical_subject_key=subject.key,
        )
        claims.append(claim)
        entries.append(
            (
                claim.to_context_object(
                    tenant_id="t1",
                    owner_user_id="u1",
                    memory_type="project_decision",
                    scope=scope,
                ),
                value,
            )
        )
    slot = MemorySlot(
        slot_id="broken",
        uri=slot_uri,
        memory_type="project_decision",
        identity_fields={"decision_topic": "database"},
        scope_keys=("memoryos:workspace:memoryos",),
        claim_ids=("c1", "c2"),
        active_claim_id="c1",
        revision=1,
        canonical_subject_key=subject.key,
        canonical_subject=subject,
    )
    entries.append((slot.to_context_object(tenant_id="t1", owner_user_id="u1", scope=scope), ""))
    _write_committed_canonical_fixture(source, entries, key="broken-multiple-active")

    with pytest.raises(CanonicalInvariantViolation, match="multiple ACTIVE claims"):
        CanonicalMemoryRetriever(source, index).search(
            CanonicalMemoryQuery(
                text="",
                tenant_id="t1",
                principal_id="u1",
                applicability_scope_keys=("memoryos:workspace:memoryos",),
            )
        )


def test_mixed_revision_layer_refs_are_rejected_by_retrieval_and_assembly(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    projector = CanonicalMemoryProjector(source, index, tmp_path)
    claim_uri, _slot_uri = _write_two_revision_claim(
        source,
        projector,
        old_value="old layer content",
        new_value="new canonical content",
    )
    records = ProjectionRecordStore(tmp_path)
    old = records.load(claim_uri, 1)
    current = records.load_current(claim_uri, source_revision=2)
    assert old is not None and current is not None
    records.save(replace(current, l0_uri=old.l0_uri))

    retriever = CanonicalMemoryRetriever(source, index, relations)
    query = CanonicalMemoryQuery(
        text="",
        tenant_id="t1",
        principal_id="u1",
        applicability_scope_keys=("memoryos:workspace:memoryos",),
        claim_uris=(claim_uri,),
    )
    result = retriever.search(query)
    assert len(result) == 1
    assert result[0]["layer_texts"] == {}
    assert result[0]["text"] == "new canonical content"

    assembler = ContextAssembler(ContextDB(source, index, relations))
    assembled = assembler.assemble(
        "canonical",
        user_id="u1",
        tenant_id="t1",
        project_id="memoryos",
        context_types=[ContextType.MEMORY],
        claim_uris=[claim_uri],
        query_intent="CURRENT",
        token_budget=200,
    )
    assert "new canonical content" in assembled["packed_context"]
    assert "old layer content" not in assembled["packed_context"]


def test_canonical_results_still_obey_connect_and_authority_filters(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    projector = CanonicalMemoryProjector(source, index, tmp_path)
    claim_uri = _write_claim(
        source,
        projector,
        claim_id="connect-filtered",
        value="workspace canonical rule",
        state="ACTIVE",
        memory_type="project_rule",
        scope=_scope(("workspace", "memoryos")),
        metadata_extra={
            "connect": {
                "connect_type": "agent",
                "adapter_id": "codex",
                "run_mode": "context_reduction",
                "world_domain": "software",
                "source_kind": "transcript",
            },
            "retrieval_views": ["project:memoryos:rules"],
            "authority": {"tenant_id": "t1", "allowed_principal_ids": ["u1"]},
            "asserted_by": "u1",
        },
    )
    assembler = ContextAssembler(ContextDB(source, index, relations))

    allowed = assembler.search(
        "canonical rule",
        user_id="u1",
        tenant_id="t1",
        project_id="memoryos",
        context_type=ContextType.MEMORY,
        connect_filters={"run_mode": "context_reduction"},
    )
    blocked_connect = assembler.search(
        "canonical rule",
        user_id="u1",
        tenant_id="t1",
        project_id="memoryos",
        context_type=ContextType.MEMORY,
        connect_filters={"run_mode": "action_capable"},
    )
    invalid_root = tmp_path / "invalid-authority"
    invalid_source = FileSystemSourceStore(invalid_root, tenant_id="t1")
    invalid_index = InMemoryIndexStore()
    invalid_relations = InMemoryRelationStore()
    invalid_projector = CanonicalMemoryProjector(invalid_source, invalid_index, invalid_root)
    _write_claim(
        invalid_source,
        invalid_projector,
        claim_id="invalid-authority",
        value="workspace canonical rule",
        state="ACTIVE",
        memory_type="project_rule",
        scope=_scope(("workspace", "memoryos")),
        metadata_extra={"asserted_by": "u2"},
    )
    blocked_authority = CanonicalMemoryRetriever(
        invalid_source,
        invalid_index,
        invalid_relations,
    ).search(
        CanonicalMemoryQuery(
            text="canonical rule",
            tenant_id="t1",
            principal_id="u1",
            applicability_scope_keys=("memoryos:workspace:memoryos",),
        )
    )

    assert [item["uri"] for item in allowed] == [claim_uri]
    assert blocked_connect == []
    assert blocked_authority == []


def test_prefetch_applicability_keeps_hierarchical_parent_path() -> None:
    prefetcher = ExistingMemoryPrefetcher(None, None)
    scope = _scope(("workspace", "workspace-a"))
    asset = ScopeRef("memoryos", "asset", "camera", parent_path=("workspace-a",)).to_dict()
    scope["canonical_subject"] = asset
    scope["applicability"] = {"all_of": [asset]}
    metadata = {
        "scope": scope,
        "asserted_by": "u1",
    }
    first_key = ScopeRef("memoryos", "asset", "camera", parent_path=("workspace-a",)).key
    second_key = ScopeRef("memoryos", "asset", "camera", parent_path=("workspace-b",)).key

    assert prefetcher._applicable(metadata, {first_key})
    assert not prefetcher._applicable(metadata, {second_key})
    assert not prefetcher._applicable(metadata, {"memoryos:asset:camera"})


def test_mixed_valid_and_malformed_all_of_fails_closed() -> None:
    scope = _scope(("workspace", "w1"))
    scope["applicability"] = {
        "all_of": [
            {"namespace": "memoryos", "kind": "workspace", "id": "w1"},
            {"namespace": "memoryos", "kind": "location"},
        ]
    }
    metadata = {
        "scope": scope,
        "asserted_by": "u1",
    }
    available = {"memoryos:workspace:w1"}

    assert not ExistingMemoryPrefetcher(None, None)._applicable(metadata, available)
    retriever = CanonicalMemoryRetriever(InMemoryIndexStore(), InMemoryIndexStore())  # type: ignore[arg-type]
    assert not retriever._applicable(metadata, tuple(available))


def test_canonical_visibility_and_authority_shapes_fail_closed() -> None:
    valid = _scope(("workspace", "w1"), tenant_id="default")
    query = CanonicalMemoryQuery(
        text="",
        tenant_id="default",
        principal_id="attacker",
        applicability_scope_keys=("memoryos:workspace:w1",),
    )
    retriever = CanonicalMemoryRetriever(InMemoryIndexStore(), InMemoryIndexStore())  # type: ignore[arg-type]
    malformed_visibility = {"scope": {**valid, "visibility": []}, "asserted_by": "u1"}
    malformed_authority = {"scope": {**valid, "authority": []}, "asserted_by": "u1"}

    assert not retriever._visible(malformed_visibility, query)
    assert not retriever._authority_permits(malformed_authority, query)
