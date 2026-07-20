from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from infrastructure.context.session_projector import SessionContextProjector
from infrastructure.store.sqlite.index_store import SQLiteIndexStore
from memory.commit.session_commit import SessionCommitService
from openApi.sdk.client import MemoryOSClient
from pre.session import SessionArchive
from tests.support.persistence.in_memory import InMemoryQueueStore
from tests.support.session_archive import build_session_archive_store


def _archive(session_id: str) -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id=session_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        messages=[{"role": "user", "content": f"archive summary marker {session_id}"}],
        metadata={"tenant_id": "default"},
    )


def test_archive_listing_is_tenant_bound_cursor_ordered_and_limited(tmp_path: Path) -> None:
    store = build_session_archive_store(tmp_path, tenant_id="default")
    for session_id in ("c", "a", "b"):
        store.write_sync_archive(_archive(session_id))

    first = store.list_archives(limit=2)
    second = store.list_archives(after_archive_uri=first[-1].archive_uri, limit=2)

    assert [archive.session_id for archive in first] == ["a", "b"]
    assert [archive.session_id for archive in second] == ["c"]
    with pytest.raises(PermissionError):
        store.list_archives(tenant_id="other")
    with pytest.raises(ValueError):
        store.list_archives(limit=1_001)


def test_async_summaries_reproject_and_rebuild_after_catalog_loss(tmp_path: Path) -> None:
    archive_store = build_session_archive_store(tmp_path, tenant_id="default")
    database = tmp_path / "context.sqlite3"
    index = SQLiteIndexStore(database)
    service = SessionCommitService(
        archive_store,
        InMemoryQueueStore(),
        session_projector=SessionContextProjector(index),
    )
    archive = _archive("rebuild")

    result = service.commit_session(archive, async_commit=True)

    assert result.done is True
    outputs = archive_store.read_async_outputs(archive)
    expected = SessionContextProjector(index).build_records(
        archive,
        async_outputs=outputs,
    )
    summaries = {
        record.record_kind: record
        for record in expected
        if record.record_kind in {"session_root", "session_l0", "session_l1"}
    }
    assert summaries["session_l0"].l0_text == outputs["abstract"]
    assert summaries["session_l1"].l1_text == outputs["overview"]
    assert summaries["session_root"].metadata["summary_source"] == "session_async_outputs"
    for record in summaries.values():
        actual = index.get_catalog(record.record_key, tenant_id="default")
        assert actual is not None
        assert (actual.l0_text, actual.l1_text) == (record.l0_text, record.l1_text)

    for suffix in ("", "-wal", "-shm"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)
    rebuilt_index = SQLiteIndexStore(database)
    rebuilt = SessionCommitService(
        archive_store,
        InMemoryQueueStore(),
        session_projector=SessionContextProjector(rebuilt_index),
    ).rebuild_session_archives()

    assert rebuilt["projected_archives"] == 1
    assert rebuilt["async_output_archives"] == 1
    for record in summaries.values():
        actual = rebuilt_index.get_catalog(record.record_key, tenant_id="default")
        assert actual is not None
        assert (actual.l0_text, actual.l1_text) == (record.l0_text, record.l1_text)


def test_runtime_startup_rebuilds_session_catalog_after_sqlite_loss(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    archive = _archive("runtime-rebuild")
    result = client.runtime.session.commit_service.commit_session(archive, async_commit=True)
    assert result.done is True
    outputs = client.runtime.session.archive_store.read_async_outputs(archive)
    projection_worker = client.runtime.session.commit_service.session_projector
    assert projection_worker is not None
    expected = projection_worker.build_records(
        archive,
        async_outputs=outputs,
    )
    summary_records = tuple(
        record
        for record in expected
        if record.record_kind in {"session_root", "session_l0", "session_l1"}
    )
    database = tmp_path / "indexes" / "context.sqlite3"
    del client
    for suffix in ("", "-wal", "-shm"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)

    restarted = MemoryOSClient(str(tmp_path))

    restarted.runtime.readiness.require_ready()
    assert restarted.runtime.readiness.details["session_archive_rebuild"]["projected_archives"] == 1
    rebuilt_index = cast(SQLiteIndexStore, restarted.runtime.stores.index)
    for record in summary_records:
        actual = rebuilt_index.get_catalog(record.record_key, tenant_id="default")
        assert actual is not None
        assert (actual.l0_text, actual.l1_text) == (record.l0_text, record.l1_text)
