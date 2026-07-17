from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.catalog import CatalogRecordKind, ServingTier
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.retrieval.orchestrator import RetrievalUnavailableError
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.contextdb.unified_migration import (
    DERIVED_SERVING_REBUILD_NAME,
    MigrationState,
    ReadRoute,
)
from memoryos.memory.canonical.visibility import CommittedStateIntegrityError
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.providers.embedding import HashingEmbeddingProvider
from memoryos.runtime.readiness import RuntimeReadinessState


def _archive(session_id: str, *, created_at: str) -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id=session_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        created_at=created_at,
        metadata={
            "tenant_id": "tenant-a",
            "timezone": "Asia/Singapore",
            "project_id": "memoryOS",
        },
        messages=[
            {
                "role": "user",
                "content": f"read desktop file for {session_id}",
                "occurred_at": created_at,
            }
        ],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": f"API_KEY=never-index-this rebuild marker {session_id}",
                "path": f"/Users/u1/Desktop/{session_id}.txt",
                "important": True,
                "occurred_at": created_at,
            }
        ],
    )


def _session_records(client: MemoryOSClient, session_id: str):
    store = client.index_store
    assert isinstance(store, SQLiteIndexStore)
    return store.scan_catalog_batch(
        filters={
            "tenant_id": "tenant-a",
            "session_ids": (session_id,),
            "include_inactive": True,
        },
        limit=1_000,
    )


def test_full_rebuild_restores_every_serving_plane_without_resurrecting_session_delete(
    tmp_path,
) -> None:
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        tenant_id="tenant-a",
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    recent = _archive(
        "recent-live",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    old = _archive("old-live", created_at="2000-01-01T00:00:00+00:00")
    deleted = _archive("deleted-session", created_at="2026-07-14T09:00:00+08:00")
    for archive in (recent, old, deleted):
        result = client.session_commit_service.sync_archive(
            archive,
            enqueue_commit_job=False,
        )
        assert result.session_projection_status == "projected"

    future_projector_record = replace(
        _session_records(client, deleted.session_id)[0],
        record_key="session:deleted-session:future-projector-kind",
        source_kind="future_projector_kind",
    )
    deleted_result = client.context_db.delete_session_context(deleted.session_id)
    assert deleted_result["processed"]
    assert not _session_records(client, deleted.session_id)
    catalog_store = client.index_store
    assert isinstance(catalog_store, SQLiteIndexStore)
    assert catalog_store.rebuildable_catalog_records((future_projector_record,)) == ()

    first_claim = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryOS",
        identity_fields={"decision_topic": "primary database"},
    )
    first_metadata = client.source_store.read_object(str(first_claim["uri"])).metadata
    slot_id = str(first_metadata["slot_id"])
    claim_id = str(first_metadata["claim_id"])

    claim_uri = str(first_claim["uri"])
    committed_relations = client.relation_store.relations_of(claim_uri)
    assert committed_relations
    removed_relation = committed_relations[0]
    client.relation_store.delete_relation(
        removed_relation.source_uri,
        removed_relation.relation_type,
        removed_relation.target_uri,
    )
    assert removed_relation not in client.relation_store.relations_of(claim_uri)

    vectors.upsert_vector(
        "orphan-tenant-a",
        [1.0, 0.0],
        metadata={"tenant_id": "tenant-a", "catalog_record_key": "missing"},
    )
    vectors.upsert_vector(
        "other-tenant",
        [0.0, 1.0],
        metadata={"tenant_id": "tenant-b", "catalog_record_key": "keep"},
    )
    store = client.index_store
    assert isinstance(store, SQLiteIndexStore)
    main_migration_before = store.get_migration_state(
        "unified-context-catalog-v1",
        tenant_id="tenant-a",
    )

    rebuilt = client.context_db.rebuild_index()

    assert rebuilt["state"] == "COMPLETED"
    assert rebuilt["consistent"] is True
    # Explicit remember() contributes its own immutable evidence archive.
    assert rebuilt["session_catalog"]["processed_archives"] >= 3
    assert rebuilt["session_catalog"]["tombstoned_records"] > 0
    assert _session_records(client, recent.session_id)
    assert not _session_records(client, deleted.session_id)
    assert all(
        record.serving_tier == ServingTier.ARCHIVED.value
        for record in _session_records(client, old.session_id)
    )
    assert vectors.get_vector_metadata("orphan-tenant-a") is None
    assert vectors.get_vector_metadata("other-tenant") is not None
    assert any(
        (
            relation.source_uri,
            relation.relation_type,
            relation.target_uri,
        )
        == (
            removed_relation.source_uri,
            removed_relation.relation_type,
            removed_relation.target_uri,
        )
        for relation in client.relation_store.relations_of(claim_uri)
    )

    current_rows = store.scan_catalog_batch(
        filters={
            "tenant_id": "tenant-a",
            "canonical_slot_ids": (slot_id,),
            "record_kind": CatalogRecordKind.CURRENT_SLOT.value,
            "include_inactive": True,
        },
        limit=10,
    )
    history_rows = store.scan_catalog_batch(
        filters={
            "tenant_id": "tenant-a",
            "canonical_slot_ids": (slot_id,),
            "record_kind": CatalogRecordKind.CLAIM_REVISION.value,
            "include_inactive": True,
        },
        limit=10,
    )
    assert len(current_rows) == 1
    assert current_rows[0].canonical_claim_id == claim_id
    assert len(history_rows) >= 1
    canonical_vector_kinds = {
        str((vectors.get_vector_metadata(uri) or {}).get("record_kind") or "")
        for uri in vectors.rows
    }
    assert CatalogRecordKind.CURRENT_SLOT.value in canonical_vector_kinds
    assert CatalogRecordKind.CLAIM_REVISION.value in canonical_vector_kinds

    derived = store.get_migration_state(
        DERIVED_SERVING_REBUILD_NAME,
        tenant_id="tenant-a",
    )
    assert derived is not None and derived["state"] == MigrationState.COMPLETED.value
    assert derived["details_json"]["last_completed_phase"] == "VERIFY"
    assert (
        store.get_migration_state("unified-context-catalog-v1", tenant_id="tenant-a")
        == main_migration_before
    )

    first_epoch = rebuilt["rebuild_epoch"]
    replay = client.context_db.rebuild_index()
    assert replay["state"] == "COMPLETED"
    assert replay["rebuild_epoch"] != first_epoch
    assert len(_session_records(client, recent.session_id)) == len(
        {record.record_key for record in _session_records(client, recent.session_id)}
    )
    assert not _session_records(client, deleted.session_id)


def test_atomic_clear_gate_is_fail_closed_and_startup_resumes_from_archive_checkpoint(
    tmp_path,
) -> None:
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        tenant_id="tenant-a",
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    first = _archive("resume-first", created_at="2026-07-14T09:00:00+08:00")
    second = _archive("resume-second", created_at="2026-07-14T10:00:00+08:00")
    for archive in (first, second):
        client.session_commit_service.sync_archive(archive, enqueue_commit_job=False)
    store = client.index_store
    assert isinstance(store, SQLiteIndexStore)

    started = store.begin_tenant_serving_rebuild(
        DERIVED_SERVING_REBUILD_NAME,
        tenant_id="tenant-a",
        batch_size=1,
        details={
            "rebuild_epoch": "simulated-crash-epoch",
            "phase": "VECTOR_CLEANUP",
            "session_checkpoint": "",
            "current_slot_checkpoint": "",
        },
    )
    assert started["state"] == MigrationState.BACKFILLING.value
    assert not _session_records(client, first.session_id)
    assert client.migration_gate is not None
    assert client.migration_gate.feature_gate.read_route is ReadRoute.LEGACY
    assert client.migration_gate.empty_result_requires_unavailable
    rolled_back = client.context_db.rollback_derived_serving_rebuild(
        "operator paused after simulated crash",
    )
    assert rolled_back["state"] == MigrationState.ROLLBACK.value
    assert rolled_back["details_json"]["rollback_from"] == MigrationState.BACKFILLING.value
    assert client.migration_gate.feature_gate.read_route is ReadRoute.LEGACY
    with pytest.raises(RetrievalUnavailableError):
        client.search_context(
            "resume-first",
            user_id="u1",
            project_id="memoryOS",
            tenant_id="tenant-a",
        )

    assert vectors.delete_by_filter({"tenant_id": "tenant-a"}) >= 0
    assert client.unified_context_migration is not None
    first_batch = client.unified_context_migration.rebuild_session_catalog_next_batch(
        "",
        batch_size=1,
    )
    assert first_batch.processed_archives == 1 and not first_batch.complete
    store.set_migration_state(
        DERIVED_SERVING_REBUILD_NAME,
        MigrationState.BACKFILLING.value,
        first_batch.checkpoint,
        {
            **started["details_json"],
            "phase": "SESSION_CATALOG",
            "session_checkpoint": first_batch.checkpoint,
            "session_archives": 1,
            "session_records": first_batch.projected_records,
            "session_vectors": first_batch.vectors_projected,
            "session_tombstoned_records": first_batch.tombstoned_records,
        },
        tenant_id="tenant-a",
        batch_size=1,
    )

    restarted = MemoryOSClient(
        str(tmp_path),
        tenant_id="tenant-a",
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )

    assert restarted.readiness.state is RuntimeReadinessState.READY
    assert _session_records(restarted, first.session_id)
    assert _session_records(restarted, second.session_id)
    resumed = restarted.index_store.get_migration_state(  # type: ignore[attr-defined]
        DERIVED_SERVING_REBUILD_NAME,
        tenant_id="tenant-a",
    )
    assert resumed is not None and resumed["state"] == MigrationState.COMPLETED.value
    assert resumed["details_json"]["session_catalog"]["processed_archives"] == 2


def test_full_rebuild_restores_ordinary_source_relations_and_preserves_other_tenant_canonical(
    tmp_path,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(
        str(tmp_path),
        tenant_id="tenant-a",
        vector_store=InMemoryVectorStore(),
        embedding_provider=HashingEmbeddingProvider(),
    )
    canonical = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryOS",
        identity_fields={"decision_topic": "ordinary relation rebuild target"},
    )
    canonical_uri = str(canonical["uri"])
    policy = ContextObject(
        uri="memoryos://user/u1/action_policies/rebuild/ordinary-source",
        context_type=ContextType.ACTION_POLICY,
        title="ordinary relation source",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    ordinary_target = ContextObject(
        uri="memoryos://user/u1/memories/rules/ordinary-target",
        context_type=ContextType.MEMORY,
        title="ordinary relation target",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    client.context_db.seed_object(policy, content="ordinary policy")
    client.context_db.seed_object(ordinary_target, content="ordinary target")
    inbound_canonical = ContextRelation(
        source_uri=policy.uri,
        relation_type="constrained_by",
        target_uri=canonical_uri,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    ordinary = ContextRelation(
        source_uri=policy.uri,
        relation_type="related_to",
        target_uri=ordinary_target.uri,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    client.context_db.add_relation(inbound_canonical)
    client.context_db.add_relation(ordinary)
    source_before = client.source_store.read_object(policy.uri).to_dict()

    old = ContextObject(
        uri="memoryos://user/u1/memories/rules/superseded-old",
        context_type=ContextType.MEMORY,
        title="old ordinary rule",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    new = ContextObject(
        uri="memoryos://user/u1/memories/rules/superseded-new",
        context_type=ContextType.MEMORY,
        title="new ordinary rule",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    client.context_db.seed_object(old, content="old")
    client.context_db.commit_operation(
        ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.SUPERSEDE,
            target_uri=old.uri,
            payload={
                "tenant_id": "tenant-a",
                "context_object": new.to_dict(),
                "content": "new",
                "reason": "ordinary rebuild proof",
            },
        )
    )

    canonical_rows = [
        relation
        for relation in client.relation_store.relations_of(canonical_uri, tenant_id="tenant-a")
        if relation.source_uri.startswith("memoryos://")
        and "/memories/canonical/" in relation.source_uri
    ]
    assert canonical_rows
    canonical_a = canonical_rows[0]
    canonical_b = ContextRelation(
        source_uri=canonical_a.source_uri,
        relation_type=canonical_a.relation_type,
        target_uri=canonical_a.target_uri,
        weight=canonical_a.weight,
        metadata={**dict(canonical_a.metadata), "tenant_id": "tenant-b"},
        created_at=canonical_a.created_at,
    )
    client.relation_store.add_relation(canonical_b)
    stale_inbound = ContextRelation(
        source_uri=policy.uri,
        relation_type="stale_inbound",
        target_uri=canonical_uri,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    stale_ordinary = ContextRelation(
        source_uri=policy.uri,
        relation_type="stale_ordinary",
        target_uri=ordinary_target.uri,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    client.relation_store.add_relation(stale_inbound)
    client.relation_store.add_relation(stale_ordinary)
    for wanted in (inbound_canonical, ordinary):
        client.relation_store.delete_relation(
            wanted.source_uri,
            wanted.relation_type,
            wanted.target_uri,
            tenant_id="tenant-a",
        )

    rebuilt = client.context_db.rebuild_index()
    assert rebuilt["ordinary_relations"]["objects"] >= 4
    assert rebuilt["ordinary_relations"]["written"] >= 4
    assert client.source_store.read_object(policy.uri).to_dict() == source_before
    tenant_a = client.relation_store.relations_of(policy.uri, tenant_id="tenant-a")
    tenant_a_keys = {
        (relation.source_uri, relation.relation_type, relation.target_uri)
        for relation in tenant_a
    }
    assert (inbound_canonical.source_uri, inbound_canonical.relation_type, inbound_canonical.target_uri) in tenant_a_keys
    assert (ordinary.source_uri, ordinary.relation_type, ordinary.target_uri) in tenant_a_keys
    assert (stale_inbound.source_uri, stale_inbound.relation_type, stale_inbound.target_uri) not in tenant_a_keys
    assert (stale_ordinary.source_uri, stale_ordinary.relation_type, stale_ordinary.target_uri) not in tenant_a_keys
    supersede_keys = {
        (relation.source_uri, relation.relation_type, relation.target_uri)
        for relation in client.relation_store.relations_of(old.uri, tenant_id="tenant-a")
    }
    assert (new.uri, "supersedes", old.uri) in supersede_keys
    assert (old.uri, "superseded_by", new.uri) in supersede_keys
    assert client.relation_store.relations_of(canonical_a.source_uri, tenant_id="tenant-b") == [canonical_b]

    # Source references remain evidence, but a retired target can never be
    # republished online or by a later serving rebuild.
    client.context_db.delete_context(ordinary_target.uri)
    assert any(
        (item.source_uri, item.relation_type, item.target_uri)
        == (ordinary.source_uri, ordinary.relation_type, ordinary.target_uri)
        for item in client.source_store.read_object(policy.uri).relations
    )
    assert not any(
        (item.source_uri, item.relation_type, item.target_uri)
        == (ordinary.source_uri, ordinary.relation_type, ordinary.target_uri)
        for item in client.relation_store.relations_of(policy.uri, tenant_id="tenant-a")
    )
    with pytest.raises(ValueError, match="not serving-eligible"):
        client.context_db.add_relation(ordinary)

    client.context_db.rebuild_index()
    assert not any(
        (item.source_uri, item.relation_type, item.target_uri)
        == (ordinary.source_uri, ordinary.relation_type, ordinary.target_uri)
        for item in client.relation_store.relations_of(policy.uri, tenant_id="tenant-a")
    )


def test_add_relation_fails_closed_on_tampered_canonical_target(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(
        str(tmp_path),
        tenant_id="tenant-a",
        vector_store=InMemoryVectorStore(),
        embedding_provider=HashingEmbeddingProvider(),
    )
    remembered = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryOS",
        identity_fields={"decision_topic": "tampered relation target"},
    )
    claim_uri = str(remembered["uri"])
    policy = ContextObject(
        uri="memoryos://user/u1/action_policies/tampered-canonical-target",
        context_type=ContextType.ACTION_POLICY,
        title="tampered canonical target source",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    client.context_db.seed_object(policy, content="policy")
    tampered = client.source_store.read_object(claim_uri)
    tampered.metadata = {**tampered.metadata, "canonical_value": "unproved overwrite"}
    client.source_store.write_object(tampered)
    edge = ContextRelation(
        source_uri=policy.uri,
        relation_type="constrained_by",
        target_uri=claim_uri,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )

    with pytest.raises(CommittedStateIntegrityError, match="live Source bundle disagree"):
        client.context_db.add_relation(edge)
    assert not client.source_store.read_object(policy.uri).relations
    assert not client.relation_store.relations_of(policy.uri, tenant_id="tenant-a")


def test_ordinary_relation_delete_barriers_prevent_online_and_rebuild_resurrection(
    tmp_path,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(
        str(tmp_path),
        tenant_id="tenant-a",
        vector_store=InMemoryVectorStore(),
        embedding_provider=HashingEmbeddingProvider(),
    )
    policy = ContextObject(
        uri="memoryos://user/u1/action_policies/delete-barrier/source",
        context_type=ContextType.ACTION_POLICY,
        title="delete barrier source",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    inbound = ContextObject(
        uri="memoryos://user/u1/memories/rules/delete-barrier-inbound",
        context_type=ContextType.MEMORY,
        title="delete barrier inbound",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    resource = ContextObject(
        uri="memoryos://user/u1/resources/delete-barrier-resource",
        context_type=ContextType.RESOURCE,
        title="delete barrier resource",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    session_source = ContextObject(
        uri="memoryos://user/u1/action_policies/delete-barrier/session-source",
        context_type=ContextType.ACTION_POLICY,
        title="session relation source",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    global_resource = ContextObject(
        uri="memoryos://resources/desktop/shared-delete-barrier-proof",
        context_type=ContextType.RESOURCE,
        title="global resource",
        owner_user_id=None,
        tenant_id="default",
    )
    for obj in (policy, inbound, resource, session_source, global_resource):
        client.context_db.seed_object(obj, content=obj.title)
    stale_current = client.index_store.enqueue_tombstone(  # type: ignore[attr-defined]
        tenant_id="tenant-a",
        record_key="slot:retired-projection:current",
        uri=resource.uri,
        reason="old_current_slot_replaced",
        source_revision=1,
        payload={"record_kind": "current_slot"},
    )
    client.index_store.mark_tombstone_applied(stale_current["tombstone_id"])  # type: ignore[attr-defined]
    assert (
        client.index_store.ordinary_relation_endpoint_state(
            resource.uri,
            tenant_id="tenant-a",
        )
        == "active"
    )
    archive = _archive("ordinary-relation-deleted-session", created_at="2026-07-14T09:00:00+08:00")
    client.session_commit_service.sync_archive(archive, enqueue_commit_job=False)

    resource_edge = ContextRelation(
        source_uri=policy.uri,
        relation_type="requires_resource",
        target_uri=resource.uri,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    inbound_edge = ContextRelation(
        source_uri=inbound.uri,
        relation_type="constrains",
        target_uri=policy.uri,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    session_edge = ContextRelation(
        source_uri=session_source.uri,
        relation_type="observed_in",
        target_uri=archive.archive_uri,
        metadata={
            "tenant_id": "tenant-a",
            "owner_user_id": "u1",
            "session_id": archive.session_id,
        },
    )
    global_edge = ContextRelation(
        source_uri=session_source.uri,
        relation_type="requires_resource",
        target_uri=global_resource.uri,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    for edge in (resource_edge, inbound_edge, session_edge, global_edge):
        client.context_db.add_relation(edge)

    client.context_db.delete_context(resource.uri, reason="resource_deleted")
    assert not [
        edge
        for edge in client.relation_store.relations_of(policy.uri, tenant_id="tenant-a")
        if edge.target_uri == resource.uri
    ]
    with pytest.raises(ValueError, match="not serving-eligible"):
        client.context_db.add_relation(resource_edge)

    client.context_db.delete_context(policy.uri, reason="source_deleted")
    assert not [
        edge
        for edge in client.relation_store.relations_of(inbound.uri, tenant_id="tenant-a")
        if edge.target_uri == policy.uri
    ]
    client.context_db.delete_session_context(archive.session_id, reason="session_deleted")
    assert not [
        edge
        for edge in client.relation_store.relations_of(session_source.uri, tenant_id="tenant-a")
        if edge.target_uri == archive.archive_uri
    ]

    client.context_db.rebuild_index()
    all_tenant_relations = [
        edge
        for edge in client.relation_store.all_relations()
        if str(edge.metadata.get("tenant_id") or "default") == "tenant-a"
    ]
    deleted_endpoints = {resource.uri, policy.uri, archive.archive_uri}
    assert not [
        edge
        for edge in all_tenant_relations
        if edge.source_uri in deleted_endpoints or edge.target_uri in deleted_endpoints
    ]
    assert any(
        edge.source_uri == global_edge.source_uri
        and edge.relation_type == global_edge.relation_type
        and edge.target_uri == global_edge.target_uri
        for edge in all_tenant_relations
    )

def test_cleaning_session_tombstone_blocks_rebuild_then_startup_replays_and_resumes(
    tmp_path,
) -> None:
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        tenant_id="tenant-a",
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    archive = _archive("cleaning-delete", created_at="2026-07-14T09:00:00+08:00")
    client.session_commit_service.sync_archive(archive, enqueue_commit_job=False)
    assert client.context_db.tombstone_service is not None
    tombstones = client.context_db.tombstone_service.enqueue_session(
        archive.session_id,
        tenant_id="tenant-a",
        reason="session_deleted",
    )
    assert tombstones
    store = client.index_store
    assert isinstance(store, SQLiteIndexStore)
    rows = store.get_tombstones(tombstones)
    barrier_id = next(
        str(row["tombstone_id"])
        for row in rows
        if row["payload_json"].get("record_kind") == "session_delete_barrier"
    )
    cleaning = store.begin_tombstone_cleanup(barrier_id)
    assert cleaning is not None and cleaning["status"] == "CLEANING"

    with pytest.raises(RuntimeError, match="in-progress tombstone"):
        client.context_db.rebuild_index()

    failed = store.get_migration_state(
        DERIVED_SERVING_REBUILD_NAME,
        tenant_id="tenant-a",
    )
    assert failed is not None and failed["state"] == MigrationState.FAILED.value
    assert failed["details_json"]["phase"] == "SESSION_CATALOG"
    assert client.readiness.state is RuntimeReadinessState.NOT_READY
    assert not _session_records(client, archive.session_id)

    restarted = MemoryOSClient(
        str(tmp_path),
        tenant_id="tenant-a",
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )

    assert restarted.readiness.state is RuntimeReadinessState.READY
    assert not _session_records(restarted, archive.session_id)
    completed = restarted.index_store.get_migration_state(  # type: ignore[attr-defined]
        DERIVED_SERVING_REBUILD_NAME,
        tenant_id="tenant-a",
    )
    assert completed is not None and completed["state"] == MigrationState.COMPLETED.value
    tombstone_rows = store.get_tombstones(tombstones)
    assert tombstone_rows and all(row["status"] == "APPLIED" for row in tombstone_rows)


def test_cross_process_gate_change_after_candidate_read_never_returns_partial_success(
    tmp_path,
    monkeypatch,
) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    archive = _archive(
        "cross-process-race",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    client.session_commit_service.sync_archive(archive, enqueue_commit_job=False)
    store = client.index_store
    assert isinstance(store, SQLiteIndexStore)
    assembler = ContextAssembler(client.context_db, hybrid_search=client.hybrid_search)
    original = assembler.unified_retrieval.generator.generate

    def clear_after_candidate_read(plan):  # noqa: ANN001, ANN202 - test fault injection.
        generated = original(plan)
        store.begin_tenant_serving_rebuild(
            DERIVED_SERVING_REBUILD_NAME,
            tenant_id="tenant-a",
            batch_size=1,
            details={"rebuild_epoch": "cross-process-race", "phase": "VECTOR_CLEANUP"},
        )
        return generated

    monkeypatch.setattr(
        assembler.unified_retrieval.generator,
        "generate",
        clear_after_candidate_read,
    )

    with pytest.raises(RetrievalUnavailableError, match="derived serving"):
        assembler.search(
            "cross-process-race",
            user_id="u1",
            project_id="memoryOS",
            tenant_id="tenant-a",
        )
