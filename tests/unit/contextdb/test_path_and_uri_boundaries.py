from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.store.model.context.context_uri import ContextURI
from infrastructure.store.model.context.errors import InvalidContextURI
from infrastructure.store.trace import RecallTraceRepository
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore
from tests.support.transaction import build_test_operation_committer as OperationCommitter


@pytest.mark.parametrize(
    "value",
    [
        "memoryos://user/u1/memories//record",
        "memoryos://user/u1/memories/../record",
        "memoryos://user/u1/memories/%2e%2e/record",
        "memoryos://user/u1/memories/%2Fetc",
        "memoryos://user/u1/memories/%5Cetc",
        "memoryos://user/u1/memories/%zz",
        "memoryos://user/u1/memories/record?revision=1",
        "memoryos://user/u1/memories/record#revision-1",
        "memoryos://user:1234/u1/memories/record",
        "memoryos://[broken/u1/memories/record",
    ],
)
def test_context_uri_rejects_noncanonical_or_escaping_forms(value: str) -> None:
    with pytest.raises(InvalidContextURI):
        ContextURI.parse(value)


def test_context_uri_canonical_form_unifies_lock_and_disk_identity(tmp_path: Path) -> None:
    encoded = "memoryos://USER/u1/memories/%72ecord/"
    plain = "memoryos://user/u1/memories/record"
    assert str(ContextURI.parse(encoded)) == plain
    assert ContextURI.parse(encoded).to_source_path(tmp_path) == ContextURI.parse(plain).to_source_path(
        tmp_path
    )

    committer = OperationCommitter(
        FileSystemSourceStore(tmp_path),
        InMemoryIndexStore(),
        str(tmp_path),
    )
    assert committer._lock_key(encoded) == committer._lock_key(plain)


@pytest.mark.parametrize(
    "trace_id",
    [
        "../outside",
        "..\\outside",
        "/absolute/path",
        "%2e%2e%2foutside",
        "00000000-0000-0000-0000-000000000000/extra",
        "00000000-0000-0000-0000-00000000000A",
    ],
)
def test_recall_trace_validates_uuid_before_any_path_read(tmp_path: Path, trace_id: str) -> None:
    service = RecallTraceRepository(tmp_path / "traces")
    with pytest.raises(ValueError, match="canonical UUID"):
        service.read(trace_id)
    assert not (tmp_path / "outside.json").exists()


def test_trace_root_and_files_are_private(tmp_path: Path) -> None:
    service = RecallTraceRepository(tmp_path / "traces")
    assert service.trace_root.stat().st_mode & 0o777 == 0o700
