from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from foundation.identity import InternalJobNamespaceError, LocalUserContext
from infrastructure.store.model.catalog import CatalogRecordKind
from infrastructure.store.sqlite import SQLiteIndexStore
from openApi.sdk.client import MemoryOSClient
from tests.support.embedding import DeterministicEmbeddingProvider
from tests.support.persistence import InMemoryVectorStore


def _caller() -> LocalUserContext:
    return LocalUserContext(user_id="u1")


def _project_one(client: MemoryOSClient):  # noqa: ANN202
    leased = client.runtime.stores.queue.lease(
        "memory_projection",
        lease_owner="projection-test",
        limit=1,
    )[0]
    outcome = client.runtime.memory.projection_worker.process_job(leased)
    client.runtime.stores.queue.ack(leased)
    return leased, outcome


def _catalog(client: MemoryOSClient) -> SQLiteIndexStore:
    return cast(SQLiteIndexStore, client.runtime.stores.index)


def test_document_commit_projects_exact_generation_and_stale_job_cannot_republish(tmp_path) -> None:  # noqa: ANN001
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vectors,
        embedding_provider=DeterministicEmbeddingProvider(),
    )
    caller = _caller()
    marker = "projection-secret-marker"

    remembered = client.remember(
        f"Prefer SQLite WAL. {marker}",
        target_hint="preference:local database",
        caller=caller,
    )
    first_job, first_outcome = _project_one(client)

    assert first_outcome == "processed"
    assert marker not in repr(first_job.payload)
    records = _catalog(client).list_catalog(
        tenant_id="default",
        filters={
            "owner_user_id": "u1",
            "document_ids": (remembered["document_id"],),
            "include_inactive": True,
        },
        limit=100,
    )
    assert {record.record_kind for record in records} >= {
        CatalogRecordKind.MEMORY_DOCUMENT.value,
        CatalogRecordKind.MEMORY_BLOCK.value,
    }
    assert {record.projection_generation for record in records} == {1}
    assert vectors.rows
    assert marker not in repr([metadata for _embedding, metadata in vectors.rows.values()])

    edited = client.edit_memory_document(
        remembered["document_uri"],
        "Prefer PostgreSQL for production and SQLite WAL locally.",
        remembered["source_digest"],
        caller=caller,
    )
    _second_job, second_outcome = _project_one(client)
    assert second_outcome == "processed"
    current = _catalog(client).list_catalog(
        tenant_id="default",
        filters={
            "owner_user_id": "u1",
            "document_ids": (remembered["document_id"],),
            "include_inactive": True,
        },
        limit=100,
    )
    assert current and {record.projection_generation for record in current} == {2}
    assert {record.source_digest for record in current} == {edited["source_digest"]}

    assert client.runtime.memory.projection_worker.process_job(first_job) == "stale"
    after_stale = _catalog(client).list_catalog(
        tenant_id="default",
        filters={
            "owner_user_id": "u1",
            "document_ids": (remembered["document_id"],),
            "include_inactive": True,
        },
        limit=100,
    )
    assert {record.projection_generation for record in after_stale} == {2}

    forgotten = client.forget(
        remembered["document_uri"],
        mode="SOFT_FORGET",
        expected_digest=edited["source_digest"],
        caller=caller,
    )
    assert forgotten["recoverable"] is True
    _delete_job, delete_outcome = _project_one(client)
    assert delete_outcome == "processed"
    assert _catalog(client).list_catalog(
        tenant_id="default",
        filters={
            "owner_user_id": "u1",
            "document_ids": (remembered["document_id"],),
            "include_inactive": True,
        },
        limit=100,
    ) == []
    assert not vectors.rows
    barrier = client.runtime.memory.control_store.load_publication_barrier(
        "default",
        "u1",
        remembered["document_id"],
    )
    assert barrier is not None

    _catalog(client).clear(tenant_id="default")
    assert client.runtime.memory.projection_worker.process_job(first_job) == "stale"
    assert _catalog(client).list_catalog(
        tenant_id="default",
        filters={
            "owner_user_id": "u1",
            "document_ids": (remembered["document_id"],),
            "include_inactive": True,
        },
        limit=100,
    ) == []

    restored = client.restore_memory_revision(
        remembered["document_uri"],
        revision=2,
        expected_digest="",
        caller=caller,
    )
    _restore_job, restore_outcome = _project_one(client)
    assert restore_outcome == "processed"
    restored_control = client.runtime.memory.control_store.load_control(
        "default",
        "u1",
        remembered["document_id"],
    )
    assert restored_control is not None
    assert restored_control.restored_from_deletion_generation == barrier.deletion_generation
    assert restored_control.projection_generation > barrier.deletion_generation
    assert restored["source_digest"] == restored_control.raw_sha256

    _catalog(client).clear(tenant_id="default")
    rebuilt = client.runtime.memory.projection_worker.rebuild_owner("default", "u1")
    assert rebuilt["projected"] >= 1
    after_rebuild = _catalog(client).list_catalog(
        tenant_id="default",
        filters={
            "owner_user_id": "u1",
            "document_ids": (remembered["document_id"],),
            "include_inactive": True,
        },
        limit=100,
    )
    assert after_rebuild
    assert {record.source_digest for record in after_rebuild} == {restored["source_digest"]}


def test_projection_worker_rejects_job_outside_internal_namespace(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.remember(
        "Prefer deterministic projection jobs.",
        target_hint="preference:projection",
        caller=_caller(),
    )
    leased = client.runtime.stores.queue.lease(
        "memory_projection",
        lease_owner="namespace-test",
        limit=1,
    )[0]
    forged = replace(
        leased,
        payload={**leased.payload, "tenant_id": "foreign"},
    )

    with pytest.raises(InternalJobNamespaceError):
        client.runtime.memory.projection_worker.process_job(forged)
