from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

import pytest

from memoryos.contextdb.session import SessionArchive, SessionArchiveStore, SessionCommitService
from memoryos.contextdb.store.local_stores import InMemoryQueueStore
from memoryos.contextdb.store.source_store import LeaseLostError
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.memory.canonical import CanonicalMemoryProjector, MemoryProjectionWorker
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.projection_state import ProjectionIntegrityError, ProjectionRecordStore
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    build_outbox,
    validate_outbox,
)
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.runtime.readiness import (
    RuntimeNotReadyError,
    RuntimeReadiness,
    RuntimeReadinessState,
)
from tests.support.canonical_transactions import (
    _artifact_root,
    _entity_aliases_proposal,
    _persisted_episode,
    _plan,
    _proposal,
    _reviewed_resolution_plan,
    _setup,
)


def _fixture(tmp_path: Path):  # noqa: ANN202
    source, index, queue, relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "projection-proof", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)
    transaction_id = str(operations[0].payload["transaction_id"])
    group_id = str(operations[0].payload["commit_group_id"])
    vectors = InMemoryVectorStore()
    projector = CanonicalMemoryProjector(
        source,
        index,
        _artifact_root(tmp_path),
        relation_store=relations,
        vector_store=vectors,
    )
    worker = MemoryProjectionWorker(projector, queue, worker_id="projection-proof-worker")
    return source, index, queue, vectors, worker, identity, transaction_id, group_id


def test_canonical_commit_group_cannot_complete_without_projection_worker(tmp_path: Path) -> None:
    service = SessionCommitService(
        SessionArchiveStore(tmp_path),
        InMemoryQueueStore(),
        projection_worker=None,
    )
    result = service._project_commit_group(
        "commit-group-with-canonical-effect",
        {
            "operations": [
                {
                    "payload": {
                        "canonical_memory": True,
                        "transaction_id": "transaction-requiring-projection",
                    }
                }
            ]
        },
    )

    assert result["status"] == "failed"
    assert result["failed"] == ["projection_worker_unavailable"]


def test_missing_projection_worker_terminalizes_projection_consumer(tmp_path: Path) -> None:
    service = SessionCommitService(
        SessionArchiveStore(tmp_path),
        InMemoryQueueStore(),
        projection_worker=None,
    )
    group_id = "missing-projection-worker"
    service.commit_group_store.create(
        group_id,
        task_id="missing-projection-worker-task",
        archive_uri="memoryos://user/u1/sessions/history/missing-projection-worker-task",
        user_id="u1",
        tenant_id="default",
    )
    memory_diff = {
        "operations": [
            {
                "payload": {
                    "canonical_memory": True,
                    "transaction_id": "transaction-requiring-projection",
                }
            }
        ]
    }

    result = service._run_consumer(
        group_id,
        "projection",
        lambda: service._project_commit_group(group_id, memory_diff),
    )

    assert result == {
        "status": "dead_letter",
        "error": "DerivedConsumerError",
        "retryable": False,
    }


class _UnexpectedProjectionFailure(Exception):
    pass


def test_unexpected_projection_failure_terminally_settles_queue_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, _index, queue, _vectors, worker, _identity, transaction_id, group_id = _fixture(tmp_path)

    def fail_projection(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise _UnexpectedProjectionFailure("unexpected projection failure")

    monkeypatch.setattr(worker, "_project_event", fail_projection)

    result = worker.process_commit_group(group_id, transaction_ids=(transaction_id,))

    job_id = f"outbox_{transaction_id}"
    assert result["failed"] == [f"{job_id}:_UnexpectedProjectionFailure", f"{job_id}:queue_dead_letter"]
    job = queue.get(job_id)
    assert job is not None
    assert job.status == "dead_letter"
    assert job.lease_token == ""


def test_commit_group_authoritative_race_releases_remaining_leased_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, index, _queue, relations, _committer, episode, scope = _setup(tmp_path)
    queue = SQLiteQueueStore(_artifact_root(tmp_path) / "queues" / "jobs.sqlite3")
    committer = OperationCommitter(
        source,
        index,
        str(tmp_path),
        relation_store=relations,
        queue_store=queue,
    )
    group_id = "projection-authoritative-race-group"
    plans = [
        _plan(
            source,
            episode,
            scope,
            _proposal(episode, "projection-race-a", "SQLite", "confirmation", "confirmed"),
            commit_group_id=group_id,
        )[2],
        _plan(
            source,
            episode,
            scope,
            _entity_aliases_proposal(episode, "projection-race-b", ["sqlite"]),
            commit_group_id=group_id,
        )[2],
    ]
    transactions: list[str] = []
    claim_uris: list[str] = []
    for plan in plans:
        operations = plan.to_context_operations(
            user_id="u1",
            tenant_id="t1",
            episode_id=episode.episode_id,
        )
        committer.commit("u1", operations)
        transactions.append(str(operations[0].payload["transaction_id"]))
        claim_uris.append(
            next(
                str(operation.target_uri)
                for operation in operations
                if dict(operation.payload.get("context_object", {}).get("metadata", {})).get("canonical_kind")
                == "claim"
            )
        )
    readiness = RuntimeReadiness()
    readiness.transition(RuntimeReadinessState.READY)
    vars(source)["readiness"] = readiness
    worker = MemoryProjectionWorker(
        CanonicalMemoryProjector(
            source,
            index,
            _artifact_root(tmp_path),
            relation_store=relations,
            vector_store=InMemoryVectorStore(),
        ),
        queue,
        worker_id="projection-authoritative-race-worker",
    )
    original_load = worker._load_projection_job_outbox
    tampered_job_ids: list[str] = []

    def tamper_after_batch_lease(job, **kwargs):  # noqa: ANN001, ANN202
        if not tampered_job_ids:
            path = Path(str(job.payload["outbox_path"]))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["outbox_digest"] = "0" * 64
            atomic_write_json(path, payload, artifact_root=_artifact_root(tmp_path))
            tampered_job_ids.append(job.job_id)
        return original_load(job, **kwargs)

    monkeypatch.setattr(worker, "_load_projection_job_outbox", tamper_after_batch_lease)

    result = worker.process_commit_group(
        group_id,
        transaction_ids=tuple(transactions),
    )

    corrupt_job_id = tampered_job_ids[0]
    all_job_ids = {f"outbox_{transaction_id}" for transaction_id in transactions}
    remaining_job_id = next(iter(all_job_ids - {corrupt_job_id}))
    assert result["processed"] == []
    assert result["quarantine"] == [corrupt_job_id]
    assert result["released"] == [remaining_job_id]
    assert result["completion_proofs"] == []
    corrupt = queue.get(corrupt_job_id)
    remaining = queue.get(remaining_job_id)
    assert corrupt is not None and corrupt.status == "quarantine"
    assert remaining is not None and remaining.status == "pending"
    assert remaining.retry_count == 0
    assert remaining.lease_token == remaining.lease_owner == ""
    assert readiness.state == RuntimeReadinessState.NOT_READY
    for claim_uri, transaction_id in zip(claim_uris, transactions, strict=True):
        assert index.get_index_metadata(claim_uri) is None
        assert not worker.proof_store.publication_path(transaction_id).exists()
        assert not worker.proof_store.completion_path(transaction_id).exists()
    with pytest.raises(RuntimeNotReadyError):
        worker.process_commit_group(group_id, transaction_ids=tuple(transactions))


def test_unexpected_derived_consumer_failure_releases_commit_group_lease(tmp_path: Path) -> None:
    service = SessionCommitService(SessionArchiveStore(tmp_path), InMemoryQueueStore())
    group_id = "unexpected-derived-consumer"
    service.commit_group_store.create(
        group_id,
        task_id="unexpected-derived-task",
        archive_uri="memoryos://user/u1/sessions/history/unexpected-derived-task",
        user_id="u1",
        tenant_id="default",
    )

    def fail_consumer() -> dict:
        raise _UnexpectedProjectionFailure("unexpected derived failure")

    result = service._run_consumer(group_id, "projection", fail_consumer)

    assert result == {
        "status": "dead_letter",
        "error": "_UnexpectedProjectionFailure",
        "retryable": False,
    }
    group = service.commit_group_store.load(group_id)
    assert group is not None
    consumer = group.consumers["projection"]
    assert consumer.status == "dead_letter"
    assert consumer.attempt_id == ""
    assert consumer.owner_pid == 0
    assert consumer.lease_expires_at == ""


def test_projection_completion_rejects_missing_job_and_done_without_record(tmp_path: Path) -> None:
    _source, _index, queue, _vectors, worker, _identity, transaction_id, group_id = _fixture(tmp_path)
    job_id = f"outbox_{transaction_id}"
    queue.jobs.pop(job_id)
    assert worker._verify_projection_completion(group_id, (transaction_id,)) == [f"{job_id}:missing_job"]

    worker.dispatch_outbox()
    leased = queue.lease(
        "memory_projection",
        lease_owner="manual-completer",
        job_ids=(job_id,),
    )[0]
    queue.ack(leased)
    failures = worker._verify_projection_completion(group_id, (transaction_id,))
    assert failures == [f"{job_id}:ProjectionIntegrityError"]


@pytest.mark.parametrize("damage", ["transaction", "outbox_path", "operation_ids"])
def test_projection_completion_rejects_queue_job_detached_from_outbox(
    tmp_path: Path,
    damage: str,
) -> None:
    _source, _index, queue, _vectors, worker, _identity, transaction_id, group_id = _fixture(tmp_path)
    assert worker.process_commit_group(group_id, transaction_ids=(transaction_id,))["failed"] == []
    job_id = f"outbox_{transaction_id}"
    job = queue.get(job_id)
    assert job is not None
    payload = dict(job.payload)
    if damage == "transaction":
        payload["transaction_id"] = "different-transaction"
    elif damage == "outbox_path":
        payload["outbox_path"] = str(tmp_path / "detached-outbox.json")
    else:
        payload["operation_ids"] = ["detached-operation"]
    queue.jobs[job_id] = replace(job, payload=payload)

    failures = worker._verify_projection_completion(group_id, (transaction_id,))

    assert failures == [f"{job_id}:ProjectionOutboxIntegrityError"]


@pytest.mark.parametrize("queue_state", ["leased", "dead_letter", "quarantine"])
def test_projection_completion_rejects_every_non_done_queue_state(
    tmp_path: Path,
    queue_state: str,
) -> None:
    _source, _index, queue, _vectors, worker, _identity, transaction_id, group_id = _fixture(tmp_path)
    job_id = f"outbox_{transaction_id}"
    leased = queue.lease(
        "memory_projection",
        lease_owner="other-worker",
        job_ids=(job_id,),
    )[0]
    if queue_state == "dead_letter":
        queue.fail(leased, "terminal")
    elif queue_state == "quarantine":
        queue.quarantine(leased, "integrity")

    failures = worker._verify_projection_completion(group_id, (transaction_id,))

    assert failures == [f"{job_id}:queue_{queue_state}"]


@pytest.mark.parametrize(
    "damage",
    [
        "record",
        "index",
        "vector",
        "source",
        "index_token",
        "index_content",
        "vector_token",
        "layer_content",
        "relation_artifact",
        "manifest",
        "scope_view",
        "scope_view_symlink",
        "taxonomy_view",
        "index_transaction",
        "index_tenant",
        "vector_state",
        "scope_transaction",
        "manifest_transaction",
    ],
)
def test_projection_completion_rejects_missing_or_mismatched_derived_proof(
    tmp_path: Path,
    damage: str,
) -> None:
    source, index, _queue, vectors, worker, identity, transaction_id, group_id = _fixture(tmp_path)
    result = worker.process_commit_group(group_id, transaction_ids=(transaction_id,))
    assert result["failed"] == []
    claim_uri = identity.claim_uri
    record = worker.projector.record_store.load_current(claim_uri, source_revision=1)
    assert record is not None

    if damage == "record":
        worker.projector.record_store.current_path(claim_uri).unlink()
    elif damage == "index":
        index.delete_index(claim_uri)
    elif damage == "vector":
        vectors.delete_vector(claim_uri)
    elif damage == "source":
        tampered = source.read_object(claim_uri)
        tampered.title = "uncommitted source tamper"
        source.write_object(tampered, content="tampered")
    elif damage == "index_token":
        index.rows[claim_uri][0].metadata["projection_publish_token"] = "tampered"
    elif damage == "index_content":
        index.rows[claim_uri] = (index.rows[claim_uri][0], "tampered projected index content")
    elif damage == "vector_token":
        vectors.rows[claim_uri][1]["publish_token"] = "tampered"
    elif damage == "layer_content":
        source.write_content(record.l0_uri, "tampered projection layer")
    elif damage == "relation_artifact":
        source.write_content(record.relations_uri, '{"relations": []}')
    elif damage == "manifest":
        source.write_content(record.manifest_uri, '{"projection_attempt_id": "tampered"}')
    elif damage == "index_transaction":
        index.rows[claim_uri][0].metadata["current_transaction_id"] = "tampered"
    elif damage == "index_tenant":
        index.rows[claim_uri][0].tenant_id = "other-tenant"
    elif damage == "vector_state":
        vectors.rows[claim_uri][1]["claim_state"] = "SUPERSEDED"
    elif damage == "scope_transaction":
        path = next((_artifact_root(tmp_path) / "views" / "scope").glob("**/current.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["current_transaction_id"] = "tampered"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif damage == "manifest_transaction":
        payload = json.loads(source.read_content(record.manifest_uri))
        payload["current_transaction_id"] = "tampered"
        source.write_content(record.manifest_uri, json.dumps(payload))
    elif damage == "scope_view":
        next((_artifact_root(tmp_path) / "views" / "scope").glob("**/current.json")).unlink()
    elif damage == "scope_view_symlink":
        path = next((_artifact_root(tmp_path) / "views" / "scope").glob("**/current.json"))
        target = tmp_path / "copied-projection-view.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    else:
        next((_artifact_root(tmp_path) / "views" / "taxonomy").glob("**/current.json")).unlink()

    failures = worker._verify_projection_completion(group_id, (transaction_id,))

    expected_error = "CommittedStateIntegrityError" if damage == "source" else "ProjectionIntegrityError"
    assert failures == [f"outbox_{transaction_id}:{expected_error}"]


def test_projection_record_store_has_one_current_attempt_identity(tmp_path: Path) -> None:
    _source, _index, _queue, _vectors, worker, identity, transaction_id, group_id = _fixture(tmp_path)
    assert worker.process_commit_group(group_id, transaction_ids=(transaction_id,))["failed"] == []
    records = ProjectionRecordStore(_artifact_root(tmp_path)).attempts(identity.claim_uri, 1)
    assert len(records) == 1
    assert records[0].current is True
    assert records[0].usable is True

    completion = worker.verify_commit_group_completion(group_id, (transaction_id,))
    assert completion["failures"] == []
    proof = completion["proofs"][0]
    claim_proof = proof["claims"][0]
    assert claim_proof["projection_attempt_id"] == records[0].projection_attempt_id
    assert claim_proof["record_digest"] == records[0].to_dict()["record_digest"]


def _started_projection_record(store: ProjectionRecordStore):  # noqa: ANN202
    return store.start(
        claim_uri="memoryos://user/u1/memories/canonical/slots/s/claims/c",
        slot_uri="memoryos://user/u1/memories/canonical/slots/s",
        source_revision=1,
        projection_revision=1,
        projection_attempt_id="a" * 32,
        input_effect_hash="effect-hash",
        l0_uri="memoryos://derived/l0",
        l1_uri="memoryos://derived/l1",
        l2_uri="memoryos://derived/l2",
        manifest_uri="memoryos://derived/manifest",
    )


def test_projection_record_load_rejects_broken_attempt_symlink(tmp_path: Path) -> None:
    store = ProjectionRecordStore(tmp_path)
    record = _started_projection_record(store)
    path = store.attempt_path_for(record)
    path.unlink()
    missing_target = tmp_path / "missing-projection-attempt.json"
    path.symlink_to(missing_target)

    with pytest.raises(ProjectionIntegrityError, match="symbolic link"):
        store.load(
            record.claim_uri,
            record.source_revision,
            projection_attempt_id=record.projection_attempt_id,
        )

    assert not missing_target.exists()
    assert not path.exists() and not path.is_symlink()


def test_projection_record_save_rejects_broken_attempt_symlink(tmp_path: Path) -> None:
    store = ProjectionRecordStore(tmp_path)
    record = _started_projection_record(store)
    path = store.attempt_path_for(record)
    path.unlink()
    missing_target = tmp_path / "missing-projection-save.json"
    path.symlink_to(missing_target)

    with pytest.raises(ProjectionIntegrityError, match="symbolic link"):
        store.save(record)

    assert not missing_target.exists()
    assert not path.exists() and not path.is_symlink()


def test_historical_projection_completion_survives_later_legal_claim_revision(
    tmp_path: Path,
) -> None:
    source, index, queue, relations, committer, episode, scope = _setup(tmp_path)
    initial = _entity_aliases_proposal(episode, "projection-history-v1", ["sqlite"])
    identity, _transition, first_plan = _plan(source, episode, scope, initial)
    first_operations = first_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", first_operations)
    first_transaction_id = str(first_operations[0].payload["transaction_id"])
    first_group_id = str(first_operations[0].payload["commit_group_id"])
    worker = MemoryProjectionWorker(
        CanonicalMemoryProjector(
            source,
            index,
            _artifact_root(tmp_path),
            relation_store=relations,
            vector_store=InMemoryVectorStore(),
        ),
        queue,
        worker_id="projection-history-worker",
    )
    first_completion = worker.process_commit_group(
        first_group_id,
        transaction_ids=(first_transaction_id,),
    )
    assert first_completion["failed"] == []

    current = next(
        claim
        for claim in CanonicalMemoryRepository(source, relations).load(identity)[1]
        if claim.current.state == "ACTIVE"
    )
    supplement_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="projection-history-v2",
            archive_uri="memoryos://user/u1/sessions/history/projection-history-v2",
            messages=[
                {
                    "id": "projection-history-v2-message",
                    "role": "user",
                    "content": "Confirm SQLite is also known as SQLite3.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    supplement = _entity_aliases_proposal(
        supplement_episode,
        "projection-history-v2",
        ["sqlite3"],
        target_claim=current,
    )
    second_plan = _reviewed_resolution_plan(
        source,
        committer,
        supplement_episode,
        supplement,
        command_suffix="projection-history-v2",
    )
    second_operations = list(second_plan.operations)
    committer.commit("u1", second_operations)
    second_claim_operation = next(
        operation
        for operation in second_operations
        if dict(operation.payload.get("context_object", {}).get("metadata", {})).get("canonical_kind") == "claim"
    )
    second_transaction_id = str(second_claim_operation.payload["transaction_id"])
    second_group_id = str(second_claim_operation.payload["commit_group_id"])
    second_completion = worker.process_commit_group(
        second_group_id,
        transaction_ids=(second_transaction_id,),
    )
    assert second_completion["failed"] == []

    historical = worker.verify_commit_group_completion(
        first_group_id,
        (first_transaction_id,),
    )

    assert historical["failures"] == []
    assert historical["proofs"] == first_completion["completion_proofs"]

    original_proof = historical["proofs"][0]
    original_claim = original_proof["claims"][0]
    legacy_claim_keys = (
        "claim_uri",
        "source_revision",
        "projection_revision",
        "projection_attempt_id",
        "input_effect_hash",
        "publish_token",
        "projected_content_digest",
        "projected_relation_digest",
        "record_digest",
    )
    legacy_core = {
        "schema_version": "projection_completion_proof_v1",
        "commit_group_id": first_group_id,
        "transaction_id": first_transaction_id,
        "job_id": f"outbox_{first_transaction_id}",
        "queue_status": "done",
        "outbox_digest": original_proof["outbox_digest"],
        "receipt_digest": original_proof["receipt_digest"],
        "claims": [{key: original_claim[key] for key in legacy_claim_keys}],
    }
    legacy_proof = {**legacy_core, "proof_digest": canonical_digest(legacy_core)}
    worker.proof_store.publication_path(first_transaction_id).unlink()
    worker.proof_store.completion_path(first_transaction_id).unlink()

    assert worker.migrate_legacy_completion_proof(
        first_group_id,
        first_transaction_id,
        legacy_proof,
    )
    migrated = worker.verify_commit_group_completion(
        first_group_id,
        (first_transaction_id,),
    )
    assert migrated["failures"] == []
    assert migrated["proofs"][0]["claims"][0]["migration_source_schema"] == ("projection_completion_proof_v1")

    historical_l0_uri = str(migrated["proofs"][0]["claims"][0]["layer_uris"]["L0"])
    source.write_content(historical_l0_uri, "tampered retired projection layer")
    damaged = worker.verify_commit_group_completion(
        first_group_id,
        (first_transaction_id,),
    )
    assert damaged["failures"] == [f"outbox_{first_transaction_id}:ProjectionIntegrityError"]


@pytest.mark.parametrize("crash_stage", ["before_queue_ack", "after_queue_ack"])
def test_projection_publication_receipt_closes_queue_ack_crash_window(
    tmp_path: Path,
    crash_stage: str,
) -> None:
    _source, _index, queue, _vectors, worker, _identity, transaction_id, group_id = _fixture(tmp_path)
    job_id = f"outbox_{transaction_id}"
    worker.dispatch_outbox()
    leased = queue.lease(
        "memory_projection",
        lease_owner="crashing-projection-worker",
        job_ids=(job_id,),
    )[0]
    outbox = worker._load_projection_job_outbox(leased)
    worker._project_event(outbox, leased.job_id, [])
    publication = worker._ensure_projection_publication(outbox, leased)
    publication_path = worker.proof_store.publication_path(transaction_id)
    publication_bytes = publication_path.read_bytes()
    assert publication["schema_version"] == "projection_publication_receipt_v1"
    assert not worker.proof_store.completion_path(transaction_id).exists()

    if crash_stage == "before_queue_ack":
        queue.retry(leased, "simulated_process_exit", max_retries=3, retryable=True)
        recovered = worker.process_commit_group(group_id, transaction_ids=(transaction_id,))
        assert recovered["failed"] == []
    else:
        queue.ack(leased)
        recovered = worker.verify_commit_group_completion(group_id, (transaction_id,))
        assert recovered["failures"] == []

    assert publication_path.read_bytes() == publication_bytes
    completion = worker.proof_store.load_completion(transaction_id)
    assert completion is not None
    assert completion["publication_digest"] == publication["publication_digest"]
    completed_job = queue.get(job_id)
    assert completed_job is not None and completed_job.status == "done"


@pytest.mark.parametrize("artifact", ["publication", "completion"])
def test_projection_completion_rejects_tampered_immutable_proof(
    tmp_path: Path,
    artifact: str,
) -> None:
    _source, _index, _queue, _vectors, worker, _identity, transaction_id, group_id = _fixture(tmp_path)
    assert worker.process_commit_group(group_id, transaction_ids=(transaction_id,))["failed"] == []
    path = (
        worker.proof_store.publication_path(transaction_id)
        if artifact == "publication"
        else worker.proof_store.completion_path(transaction_id)
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["receipt_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    failures = worker.verify_commit_group_completion(group_id, (transaction_id,))["failures"]

    assert failures == [f"outbox_{transaction_id}:ProjectionIntegrityError"]


def test_projection_completion_rejects_valid_outbox_for_unrelated_prepared_intent(
    tmp_path: Path,
) -> None:
    _source, _index, _queue, _vectors, worker, _identity, transaction_id, group_id = _fixture(tmp_path)
    assert worker.process_commit_group(group_id, transaction_ids=(transaction_id,))["failed"] == []
    outbox_path = _artifact_root(tmp_path) / "system" / "outbox" / f"{transaction_id}.json"
    original = json.loads(outbox_path.read_text(encoding="utf-8"))
    before_images = [dict(item) for item in original["before_images"]]
    before_images[0]["content"] = "different but internally valid prepared intent"
    operations = [ContextOperation.from_dict(dict(item)) for item in original["operations"]]
    replacement = build_outbox(
        transaction_id=str(original["transaction_id"]),
        idempotency_key=str(original["idempotency_key"]),
        tenant_id=str(original["tenant_id"]),
        user_id=str(original["user_id"]),
        operations=operations,
        status="committed",
        before_images=before_images,
        effect_manifests=[dict(item) for item in original["effect_manifests"]],
        claim_revisions=[dict(item) for item in original["claim_revisions"]],
        commit_group_id=str(original["commit_group_id"]),
        receipt_path=str(original["receipt_path"]),
        receipt_digest=str(original["receipt_digest"]),
    )
    atomic_write_json(
        outbox_path,
        replacement,
        artifact_root=_artifact_root(tmp_path),
    )

    failures = worker._verify_projection_completion(group_id, (transaction_id,))

    assert failures == [f"outbox_{transaction_id}:ProjectionIntegrityError"]


def test_outbox_rejects_claim_projection_set_detached_from_immutable_operations(
    tmp_path: Path,
) -> None:
    _source, _index, _queue, _vectors, _worker, _identity, transaction_id, _group_id = _fixture(tmp_path)
    outbox_path = _artifact_root(tmp_path) / "system" / "outbox" / f"{transaction_id}.json"
    original = json.loads(outbox_path.read_text(encoding="utf-8"))
    operations = [ContextOperation.from_dict(dict(item)) for item in original["operations"]]
    detached = build_outbox(
        transaction_id=str(original["transaction_id"]),
        idempotency_key=str(original["idempotency_key"]),
        tenant_id=str(original["tenant_id"]),
        user_id=str(original["user_id"]),
        operations=operations,
        status="committed",
        before_images=[dict(item) for item in original["before_images"]],
        effect_manifests=[dict(item) for item in original["effect_manifests"]],
        claim_revisions=[],
        commit_group_id=str(original["commit_group_id"]),
        receipt_path=str(original["receipt_path"]),
        receipt_digest=str(original["receipt_digest"]),
    )

    with pytest.raises(OutboxIntegrityError, match="claim revision"):
        validate_outbox(detached)


def test_rebuild_retires_projection_current_without_committed_claim_head(tmp_path: Path) -> None:
    _source, _index, _queue, _vectors, worker, identity, transaction_id, group_id = _fixture(tmp_path)
    assert worker.process_commit_group(group_id, transaction_ids=(transaction_id,))["failed"] == []
    current = worker.projector.record_store.load_current(identity.claim_uri, source_revision=1)
    assert current is not None
    dangling_uri = f"{identity.slot_uri}/claims/dangling-projection"
    dangling = replace(
        current,
        claim_uri=dangling_uri,
        projection_attempt_id="a" * 32,
        publish_token="b" * 32,
        current=False,
    )
    worker.projector.record_store.save(dangling)
    worker.projector.record_store.promote(dangling)

    with pytest.raises(ProjectionIntegrityError, match="closure mismatch"):
        worker.verify_current_projections()

    rebuilt = worker.projector.rebuild(clear_views=True)
    assert rebuilt["retired"] == 1
    assert worker.projector.record_store.load_current(dangling_uri) is None
    assert worker.verify_current_projections()["verified"] == 1


def test_two_projection_workers_cannot_process_the_same_live_lease(tmp_path: Path) -> None:
    _source, _index, _queue, _vectors, fixture_worker, _identity, transaction_id, _group_id = _fixture(tmp_path)
    queue = SQLiteQueueStore(tmp_path / "projection-queue.sqlite")
    entered = Event()
    release = Event()

    def block_first_projection(stage: str, _claim_uri: str, _revision: int) -> None:
        if stage == "after_read":
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("projection concurrency test did not release first worker")

    fixture_worker.projector.test_hook = block_first_projection
    first = MemoryProjectionWorker(fixture_worker.projector, queue, worker_id="projection-worker-a")
    second = MemoryProjectionWorker(fixture_worker.projector, queue, worker_id="projection-worker-b")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first.process_pending)
        assert entered.wait(timeout=10)
        second_result = pool.submit(second.process_pending).result(timeout=10)
        release.set()
        first_result = first_future.result(timeout=10)

    job_id = f"outbox_{transaction_id}"
    assert first_result["processed"] == [job_id]
    assert second_result["processed"] == []
    completed = queue.get(job_id)
    assert completed is not None and completed.status == "done"


def test_projection_lease_expiry_allows_takeover_but_old_worker_cannot_ack(tmp_path: Path) -> None:
    _source, _index, _queue, _vectors, fixture_worker, _identity, transaction_id, _group_id = _fixture(tmp_path)
    queue = SQLiteQueueStore(tmp_path / "projection-takeover.sqlite")
    worker = MemoryProjectionWorker(fixture_worker.projector, queue, worker_id="projection-takeover")
    worker.dispatch_outbox()
    job_id = f"outbox_{transaction_id}"
    clock = [datetime(2026, 7, 13, 0, 0, tzinfo=timezone.utc)]
    queue._now_dt = lambda: clock[0]  # type: ignore[method-assign]
    old_lease = queue.lease(
        "memory_projection",
        lease_owner="expired-projection-worker",
        lease_seconds=1,
        job_ids=(job_id,),
    )[0]
    clock[0] += timedelta(seconds=2)

    result = worker.process_pending()

    assert result["processed"] == [job_id]
    with pytest.raises(LeaseLostError, match="queue lease lost"):
        queue.ack(old_lease)
    completed = queue.get(job_id)
    assert completed is not None and completed.status == "done"
