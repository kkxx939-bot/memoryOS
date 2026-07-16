from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.session import SessionArchive
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.memory.canonical.projection import CanonicalMemoryProjector, MemoryProjectionWorker
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.memory.canonical.slot_projection import CurrentSlotProjection
from tests.unit.test_canonical_transaction_commit import (
    _artifact_root,
    _persisted_episode,
    _plan,
    _proposal,
    _replacement_proposal,
    _reviewed_resolution_plan,
    _setup,
)


class _FailOnceDurableTombstoneCatalog:
    """Leave a FAILED durable tombstone once, then replay through SQLite."""

    def __init__(self, delegate: SQLiteIndexStore) -> None:
        self.delegate = delegate
        self.fail_next_tombstone = False
        self.tombstone_attempts = 0

    def upsert_catalog(self, record: CatalogRecord) -> None:
        self.delegate.upsert_catalog(record)

    def apply_tombstone(self, **kwargs: Any) -> dict[str, Any]:
        self.tombstone_attempts += 1
        if self.fail_next_tombstone:
            self.fail_next_tombstone = False
            queued = self.delegate.enqueue_tombstone(**kwargs)
            self.delegate.mark_tombstone_failed(
                str(queued["tombstone_id"]),
                "injected current Slot projection deletion failure",
            )
            raise OSError("injected current Slot projection deletion failure")
        return self.delegate.apply_tombstone(**kwargs)


def test_durable_outbox_retries_active_switch_tombstone_before_ack(tmp_path: Path) -> None:
    source, claim_index, queue, relations, committer, episode, scope = _setup(tmp_path)
    initial = _proposal(episode, "slot-outbox-initial", "SQLite", "confirmation", "confirmed")
    identity, _transition, first_plan = _plan(source, episode, scope, initial)
    first_operations = first_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", first_operations)

    catalog = SQLiteIndexStore(_artifact_root(tmp_path) / "indexes" / "current-slot-catalog.sqlite3")
    fail_once_catalog = _FailOnceDurableTombstoneCatalog(catalog)
    claim_projector = CanonicalMemoryProjector(
        source,
        claim_index,
        _artifact_root(tmp_path),
        relation_store=relations,
    )
    slot_projector = CurrentSlotProjection(
        CanonicalMemoryRepository(source, relations),
        fail_once_catalog,
    )
    worker = MemoryProjectionWorker(
        claim_projector,
        queue,
        current_slot_projector=slot_projector,
        worker_id="current-slot-outbox-test",
    )

    first_result = worker.process_pending()
    assert first_result["failed"] == []
    assert len(first_result["processed"]) == 1
    first_slot, first_claims = CanonicalMemoryRepository(source, relations).load(identity)
    assert first_slot is not None and first_slot.active_claim_id is not None
    first_record = catalog.get_catalog(
        CurrentSlotProjection.record_key(first_slot.slot_id),
        tenant_id="t1",
    )
    assert first_record is not None
    assert first_record.canonical_claim_id == first_slot.active_claim_id
    previous_claim = next(claim for claim in first_claims if claim.claim_id == first_slot.active_claim_id)

    existing_outboxes = set((_artifact_root(tmp_path) / "system" / "outbox").glob("*.json"))
    replacement_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="slot-outbox-replacement",
            archive_uri="memoryos://user/u1/sessions/history/slot-outbox-replacement",
            messages=[
                {
                    "id": "slot-outbox-replacement-message",
                    "role": "user",
                    "content": "The primary storage backend is now changed from SQLite to PostgreSQL.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    replacement = _replacement_proposal(
        replacement_episode,
        "slot-outbox-replacement",
        "PostgreSQL",
        previous_claim,
    )
    replacement_plan = _reviewed_resolution_plan(
        source,
        committer,
        replacement_episode,
        replacement,
        command_suffix="current-slot-outbox",
    )
    committer.commit("u1", list(replacement_plan.operations))
    new_outboxes = set((_artifact_root(tmp_path) / "system" / "outbox").glob("*.json")) - existing_outboxes
    assert len(new_outboxes) == 1
    transaction_id = next(iter(new_outboxes)).stem
    job_id = f"outbox_{transaction_id}"

    fail_once_catalog.fail_next_tombstone = True
    failed = worker.process_pending()

    assert failed["processed"] == []
    assert failed["failed"] == [job_id]
    pending_job = queue.get(job_id)
    assert pending_job is not None
    assert pending_job.status == "pending"
    assert pending_job.retry_count == 1
    pending_tombstones = catalog.get_pending_tombstones()
    assert len(pending_tombstones) == 1
    assert pending_tombstones[0]["status"] == "FAILED"
    assert pending_tombstones[0]["reason"] == "canonical_active_claim_switched"
    assert pending_tombstones[0]["source_revision"] == first_slot.revision
    still_old = catalog.get_catalog(first_record.record_key, tenant_id="t1")
    assert still_old is not None and still_old.canonical_claim_id == previous_claim.claim_id

    replayed = worker.process_pending()

    assert replayed["failed"] == []
    assert replayed["processed"] == [job_id]
    completed_job = queue.get(job_id)
    assert completed_job is not None
    assert completed_job.status == "done"
    assert completed_job.retry_count == 1
    current_slot, current_claims = CanonicalMemoryRepository(source, relations).load(identity)
    assert current_slot is not None and current_slot.active_claim_id is not None
    current = catalog.get_catalog(first_record.record_key, tenant_id="t1")
    assert current is not None
    assert current.source_revision == first_slot.revision + 1
    assert current.canonical_claim_id == current_slot.active_claim_id
    assert current.canonical_claim_id != previous_claim.claim_id
    assert catalog.get_pending_tombstones() == []
    assert fail_once_catalog.tombstone_attempts == 2

    # Claim revision projections remain independently published for HISTORY/AUDIT.
    assert claim_projector.record_store.load_current(previous_claim.uri, source_revision=2) is not None
    active_claim = next(claim for claim in current_claims if claim.claim_id == current_slot.active_claim_id)
    assert claim_projector.record_store.load_current(active_claim.uri, source_revision=1) is not None

    with sqlite3.connect(catalog.path) as conn:
        durable = conn.execute(
            "SELECT reason, source_revision, status FROM context_tombstones ORDER BY tombstone_id"
        ).fetchall()
    assert durable == [("canonical_active_claim_switched", first_slot.revision, "APPLIED")]
