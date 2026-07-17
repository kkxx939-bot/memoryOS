from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.resource.resource_importer import ResourceImporter
from memoryos.contextdb.skill.skill_model import Skill
from memoryos.contextdb.skill.skill_registry import SkillRegistry
from memoryos.contextdb.store.index_consistency import (
    IndexConsistencyService,
    prepare_generic_index_rebuild,
)
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.contextdb.unified_migration import DERIVED_SERVING_REBUILD_NAME
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.workers.embedding_worker import EmbeddingWorker
from memoryos.workers.memory_proposal_worker import MemoryProposalWorker
from memoryos.workers.reindex_worker import ReindexWorker
from memoryos.workers.semantic_worker import SemanticWorker
from memoryos.workers.session_commit_worker import SessionCommitWorker
from tests.support.canonical_transactions import (
    _entity_aliases_proposal,
    _plan,
    _proposal,
    _setup,
)


def _add_operation(*, tenant_id: str, name: str) -> tuple[ContextObject, ContextOperation]:
    obj = ContextObject(
        uri=f"memoryos://user/u1/memories/rules/{name}",
        context_type=ContextType.MEMORY,
        title=name,
        owner_user_id="u1",
        tenant_id=tenant_id,
    )
    return obj, ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        payload={
            "tenant_id": tenant_id,
            "context_object": obj.to_dict(),
            "content": f"content for {name}",
            "reason": "projection fence integration proof",
        },
    )


def _catalog_snapshot(store: SQLiteIndexStore, *, tenant_id: str) -> tuple[dict, ...]:
    records = store.scan_catalog_batch(
        filters={"tenant_id": tenant_id, "include_inactive": True},
        limit=1_000,
    )
    assert all(isinstance(record, CatalogRecord) for record in records)
    return tuple(record.to_dict() for record in records)


def test_tenant_rebuild_fence_blocks_all_runtime_mutations_and_queue_leases(
    tmp_path: Path,
) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    first = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    competing = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    _first_identity, _first_transition, first_plan = _plan(
        first.source_store,
        episode,
        scope,
        _proposal(episode, "projection-fence-first", "SQLite", "confirmation", "confirmed"),
    )
    second_identity, _second_transition, second_plan = _plan(
        first.source_store,
        episode,
        scope,
        _entity_aliases_proposal(episode, "projection-fence-second", ["sqlite3"]),
    )
    first_operations = first_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    second_operations = second_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    first.committer.commit("u1", first_operations)
    first_job_id = f"outbox_{first_plan.transaction_id}"
    queued_before = competing.queue_store.get(first_job_id)
    assert queued_before is not None and queued_before.status == "pending"

    delete_obj, _delete_seed_operation = _add_operation(tenant_id="t1", name="blocked-delete")
    competing.context_db.seed_object(delete_obj, content="must remain live while rebuild is fenced")
    repair_source, _repair_source_operation = _add_operation(tenant_id="t1", name="repair-source")
    repair_target, _repair_target_operation = _add_operation(tenant_id="t1", name="repair-target")
    competing.context_db.seed_object(repair_source, content="repair source")
    competing.context_db.seed_object(repair_target, content="repair target")
    repair_relation = ContextRelation(
        source_uri=repair_source.uri,
        relation_type="related_to",
        target_uri=repair_target.uri,
        metadata={"tenant_id": "t1", "owner_user_id": "u1"},
    )
    competing.context_db.add_relation(repair_relation)
    competing.relation_store.delete_relation(
        repair_relation.source_uri,
        repair_relation.relation_type,
        repair_relation.target_uri,
        tenant_id="t1",
    )
    assert competing.relation_store.relations_of(repair_source.uri, tenant_id="t1") == []
    semantic_job = competing.queue_store.enqueue(
        QueueJob(
            job_id="semantic-blocked-by-rebuild",
            queue_name="semantic",
            action="refresh_layers",
            target_uri=repair_source.uri,
            payload={"tenant_id": "t1"},
        )
    )
    embedding_job = competing.queue_store.enqueue(
        QueueJob(
            job_id="embedding-blocked-by-rebuild",
            queue_name="embedding",
            action="embed",
            target_uri=repair_target.uri,
            payload={"tenant_id": "t1"},
        )
    )
    worker_vectors = InMemoryVectorStore()
    resource_uri = "memoryos://resources/repository/blocked-resource"
    skill = Skill(
        uri="memoryos://skills/blocked-skill",
        title="blocked skill",
        tool_name="blocked.tool",
    )
    skill_registry = SkillRegistry(competing.source_store, competing.index_store)

    index = first.index_store
    assert isinstance(index, SQLiteIndexStore)
    migration = first.unified_context_migration
    assert migration is not None
    normal_obj, normal_operation = _add_operation(tenant_id="t1", name="blocked-normal")
    seeded_obj, _seeded_operation = _add_operation(tenant_id="t1", name="blocked-seed")
    second_uris = tuple(
        str(payload["uri"])
        for operation in second_operations
        if isinstance((payload := operation.payload.get("context_object")), dict)
    )

    with migration.derived_rebuild_fence():
        started = index.begin_tenant_serving_rebuild(
            DERIVED_SERVING_REBUILD_NAME,
            tenant_id="t1",
            batch_size=16,
            details={
                "rebuild_epoch": "cross-runtime-fence-proof",
                "phase": "VECTOR_CLEANUP",
            },
        )
        assert started["state"] == "BACKFILLING"
        catalog_before = _catalog_snapshot(index, tenant_id="t1")
        relations_before = tuple(first.relation_store.all_relations())

        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.committer.commit("u1", [normal_operation])
        with pytest.raises(FileNotFoundError):
            competing.source_store.read_object(normal_obj.uri)

        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.committer.commit("u1", second_operations)
        for uri in second_uris:
            with pytest.raises(FileNotFoundError):
                competing.source_store.read_object(uri)

        delete_before = competing.source_store.read_object(delete_obj.uri).to_dict()
        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.context_db.seed_object(seeded_obj, content="must not cross rebuild")
        with pytest.raises(FileNotFoundError):
            competing.source_store.read_object(seeded_obj.uri)
        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.context_db.delete_context(delete_obj.uri, reason="blocked by rebuild")
        assert competing.source_store.read_object(delete_obj.uri).to_dict() == delete_before
        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.context_db.add_relation(repair_relation)
        assert competing.relation_store.relations_of(repair_source.uri, tenant_id="t1") == []

        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.committer.resume(
                "u1",
                normal_operation,
                "started",
            )
        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.committer.resume_canonical_batch("u1", [])
        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.committer.recover_pending_canonical("u1")
        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.committer.recover_pending_regular_memory(
                "u1",
                commit_group_id="not-reached-while-rebuild-fenced",
            )

        worker = competing.memory_projection_worker
        with pytest.raises(TimeoutError, match="Lock already held"):
            worker.dispatch_outbox()
        with pytest.raises(TimeoutError, match="Lock already held"):
            worker.process_pending(limit=10)
        with pytest.raises(TimeoutError, match="Lock already held"):
            worker._process_pending_during_startup(limit=10)
        with pytest.raises(TimeoutError, match="Lock already held"):
            worker.process_commit_group(
                "not-reached-while-rebuild-fenced",
                transaction_ids=(first_plan.transaction_id,),
            )
        assert competing.queue_store.get(first_job_id) == queued_before
        assert _catalog_snapshot(index, tenant_id="t1") == catalog_before
        assert tuple(first.relation_store.all_relations()) == relations_before

        persisted_archive = competing.session_archive_store.read_archive(
            "memoryos://user/u1/sessions/history/s1",
            tenant_id="t1",
        )
        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.session_commit_service.enqueue_failed_inline_commit(persisted_archive)
        with pytest.raises(TimeoutError, match="Lock already held"):
            competing.session_commit_service.recover_session_projection_frontier()
        with pytest.raises(TimeoutError, match="Lock already held"):
            SessionCommitWorker(competing.session_commit_service).process_pending()
        with pytest.raises(TimeoutError, match="Lock already held"):
            MemoryProposalWorker(competing.session_commit_service).process_pending()
        with pytest.raises(TimeoutError, match="Lock already held"):
            SemanticWorker(competing.source_store, competing.queue_store).process_pending()
        with pytest.raises(TimeoutError, match="Lock already held"):
            EmbeddingWorker(
                competing.source_store,
                competing.queue_store,
                worker_vectors,
            ).process_pending()
        assert competing.queue_store.get(semantic_job.job_id) == semantic_job
        assert competing.queue_store.get(embedding_job.job_id) == embedding_job
        assert worker_vectors.vector_uris() == []
        with pytest.raises(TimeoutError, match="Lock already held"):
            ResourceImporter(competing.source_store, competing.index_store).import_text(
                resource_uri,
                "blocked resource",
                "repository",
                "must not cross rebuild",
            )
        with pytest.raises(FileNotFoundError):
            competing.source_store.read_object(resource_uri)
        with pytest.raises(TimeoutError, match="Lock already held"):
            skill_registry.register(skill, content="must not cross rebuild")
        assert skill_registry.get(skill.uri) is None
        with pytest.raises(FileNotFoundError):
            competing.source_store.read_object(skill.uri)
        layer_before = competing.source_store.read_object(repair_target.uri).to_dict()
        with pytest.raises(TimeoutError, match="Lock already held"):
            LayerRefresher(competing.source_store).refresh(
                competing.source_store.read_object(repair_target.uri),
                "must not publish layers during rebuild",
            )
        assert competing.source_store.read_object(repair_target.uri).to_dict() == layer_before
        with pytest.raises(TimeoutError, match="Lock already held"):
            prepare_generic_index_rebuild(competing.source_store, competing.index_store)
        with pytest.raises(TimeoutError, match="Lock already held"):
            ReindexWorker(competing.source_store, competing.index_store).rebuild()
        consistency = IndexConsistencyService(
            competing.source_store,
            competing.index_store,
            competing.relation_store,
        )
        with pytest.raises(TimeoutError, match="Lock already held"):
            consistency.rebuild()
        with pytest.raises(TimeoutError, match="Lock already held"):
            consistency.rebuild_for_canonical_reprojection()
        with pytest.raises(TimeoutError, match="Lock already held"):
            consistency.rebuild_ordinary_relations_next_batch(tenant_id="t1")
        assert _catalog_snapshot(index, tenant_id="t1") == catalog_before
        assert tuple(first.relation_store.all_relations()) == relations_before
        blocked_runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
        assert blocked_runtime.readiness.state.value == "NOT_READY"
        assert any("Lock already held" in reason for reason in blocked_runtime.readiness.reasons)
        assert _catalog_snapshot(index, tenant_id="t1") == catalog_before
        assert tuple(first.relation_store.all_relations()) == relations_before

        other_tenant = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t2"))
        other_obj, other_operation = _add_operation(tenant_id="t2", name="tenant-two-unblocked")
        other_tenant.committer.commit("u1", [other_operation])
        assert other_tenant.source_store.read_object(other_obj.uri).tenant_id == "t2"

    competing.committer.commit("u1", [normal_operation])
    assert competing.source_store.read_object(normal_obj.uri).tenant_id == "t1"
    competing.context_db.seed_object(seeded_obj, content="released seed")
    assert competing.source_store.read_object(seeded_obj.uri).tenant_id == "t1"
    competing.context_db.add_relation(repair_relation)
    repaired = competing.relation_store.relations_of(repair_source.uri, tenant_id="t1")
    assert [(item.source_uri, item.relation_type, item.target_uri) for item in repaired] == [
        (repair_relation.source_uri, repair_relation.relation_type, repair_relation.target_uri)
    ]
    competing.committer.commit("u1", second_operations)
    projection = competing.memory_projection_worker.process_pending(limit=10)
    assert first_job_id in projection["processed"]
    assert f"outbox_{second_plan.transaction_id}" in projection["processed"]
    completed_job = competing.queue_store.get(first_job_id)
    assert completed_job is not None and completed_job.status == "done"
    slot, claims = CanonicalMemoryRepository(
        competing.source_store,
        competing.relation_store,
    ).load(second_identity)
    assert slot is not None
    assert len(claims) == 1 and claims[0].current.state == "ACTIVE"


class _RenewalProbeLock:
    def __init__(self) -> None:
        self.renewals = 0
        self.released = False

    def acquire(self, lock_key: str, ttl_seconds: int = 30):  # noqa: ANN201
        from memoryos.contextdb.store.source_store import LockToken

        return LockToken(lock_key=lock_key, token="probe", fence=1)

    def renew(self, token, ttl_seconds: int = 30):  # noqa: ANN001, ANN201
        self.renewals += 1
        if self.renewals > 1:
            raise TimeoutError("simulated projection fence lease loss")
        return token

    def assert_owned(self, token) -> None:  # noqa: ANN001
        if self.renewals > 1:
            raise TimeoutError("simulated projection fence lease loss")

    def release(self, token) -> None:  # noqa: ANN001
        self.released = True


def test_projection_fence_renewal_loss_fails_closed(tmp_path: Path) -> None:
    from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
    from memoryos.contextdb.unified_migration import RuntimeMigrationCoordinator

    lock = _RenewalProbeLock()
    gate = RuntimeMigrationCoordinator(
        SQLiteIndexStore(tmp_path / "catalog.sqlite3"),
        tenant_id="t1",
        lock_store=lock,
    )
    lease = gate.acquire_projection_fence()
    assert lease is not None and lock.renewals == 1
    with pytest.raises(RuntimeError, match="renewal failed|previously failed"):
        lease.checkpoint()
    with pytest.raises(RuntimeError, match="renewal failed|previously failed"):
        gate.release_projection_fence(lease)
    assert lock.released

    unfenced = RuntimeMigrationCoordinator(
        SQLiteIndexStore(tmp_path / "unfenced-catalog.sqlite3"),
        tenant_id="t1",
    )
    with pytest.raises(RuntimeError, match="durable migration fence"):
        unfenced.acquire_projection_fence()


def test_online_writer_acquires_stable_state_fence_before_rebuild_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    writer = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    index = owner.index_store
    assert isinstance(index, SQLiteIndexStore)
    migration = owner.unified_context_migration
    assert migration is not None

    baseline_obj, baseline_operation = _add_operation(tenant_id="t1", name="baseline-before-race")
    owner.committer.commit("u1", [baseline_operation])
    baseline_catalog = _catalog_snapshot(index, tenant_id="t1")
    assert writer.source_store.read_object(baseline_obj.uri).uri == baseline_obj.uri

    raced_obj, raced_operation = _add_operation(tenant_id="t1", name="writer-first-race")
    entered = Event()
    continue_commit = Event()
    failures: list[BaseException] = []
    original_unfenced = writer.committer._commit_unfenced

    def blocked_unfenced(user_id, operations):  # noqa: ANN001, ANN202
        entered.set()
        if not continue_commit.wait(timeout=5):
            raise TimeoutError("test writer was not released")
        return original_unfenced(user_id, operations)

    monkeypatch.setattr(writer.committer, "_commit_unfenced", blocked_unfenced)

    def run_writer() -> None:
        try:
            writer.committer.commit("u1", [raced_operation])
        except BaseException as exc:  # pragma: no cover - asserted below.
            failures.append(exc)

    thread = Thread(target=run_writer)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(TimeoutError, match="Lock already held"):
            with migration.derived_rebuild_fence():
                raise AssertionError("rebuild unexpectedly entered while writer held the fence")
        assert _catalog_snapshot(index, tenant_id="t1") == baseline_catalog
        with pytest.raises(FileNotFoundError):
            writer.source_store.read_object(raced_obj.uri)
    finally:
        continue_commit.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []
    assert writer.source_store.read_object(raced_obj.uri).uri == raced_obj.uri
