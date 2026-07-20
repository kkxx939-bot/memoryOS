"""召回轨迹读取和彻底删除测试。"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from infrastructure.store.trace import (
    RecallTraceEraseBackend,
    RecallTraceEraseIntegrityError,
    RecallTraceRepository,
)
from memory.commit.erase import DerivedEraseRequest


def _request(*, tenant_id: str = "default", owner_user_id: str = "user-a") -> DerivedEraseRequest:
    document_id = "memdoc_AAAAAAAAAAAAAAAA"
    return DerivedEraseRequest(
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        document_id=document_id,
        document_uri=f"memoryos://user/{owner_user_id}/memory/documents/{document_id}",
        relative_path="preferences.md",
        document_kind="preferences",
        erasure_epoch=f"erase_{'a' * 64}",
        source_digest="b" * 64,
        document_revision_floor=1,
        projection_generation_floor=1,
    )


def _save_trace(
    repository: RecallTraceRepository,
    *,
    owner_user_id: str,
    tenant_id: str = "default",
) -> str:
    trace_id = str(uuid.uuid4())
    repository.save(
        trace_id,
        {
            "trace_id": trace_id,
            "scope": {"user_id": owner_user_id, "tenant_id": tenant_id},
        },
    )
    return trace_id


def test_recall_trace_erasure_is_owner_scoped_idempotent_and_query_free(tmp_path: Path) -> None:
    repository = RecallTraceRepository(tmp_path / "recall-traces")
    user_a_trace = _save_trace(repository, owner_user_id="user-a")
    user_b_trace = _save_trace(repository, owner_user_id="user-b")

    assert b"private-query" not in (repository.trace_root / f"{user_a_trace}.json").read_bytes()
    backend = RecallTraceEraseBackend(tmp_path)

    assert backend.erase_document(_request()) is True
    assert not (repository.trace_root / f"{user_a_trace}.json").exists()
    assert (repository.trace_root / f"{user_b_trace}.json").exists()
    assert backend.erase_document(_request()) is True


def test_recall_trace_erasure_uses_the_exact_nondefault_tenant_root(tmp_path: Path) -> None:
    tenant_root = tmp_path / "tenants" / "tenant-a" / "recall-traces"
    repository = RecallTraceRepository(tenant_root)
    trace_id = _save_trace(
        repository,
        owner_user_id="user-a",
        tenant_id="tenant-a",
    )

    assert RecallTraceEraseBackend(tmp_path).erase_document(_request(tenant_id="tenant-a")) is True
    assert not (tenant_root / f"{trace_id}.json").exists()


def test_recall_trace_erasure_rejects_a_symbolic_link_trace_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "recall-traces").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RecallTraceEraseIntegrityError, match="unsafe"):
        RecallTraceEraseBackend(tmp_path).erase_document(_request())

    assert list(outside.iterdir()) == []
