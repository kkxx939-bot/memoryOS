from __future__ import annotations

from pathlib import Path

import pytest

from memoryos.application.context.retrieval_service import RetrievalService
from memoryos.application.context.trace_erase import (
    RecallTraceEraseBackend,
    RecallTraceEraseIntegrityError,
)
from memoryos.memory.documents.erase import DerivedEraseRequest


class _EmptyAssembler:
    reranker = None

    def search(self, _query: str, **_kwargs):  # noqa: ANN003, ANN201 - compact trace fixture.
        return []


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


def test_recall_trace_erasure_is_owner_scoped_idempotent_and_query_free(tmp_path: Path) -> None:
    service = RetrievalService(_EmptyAssembler(), tmp_path / "recall-traces")  # type: ignore[arg-type]
    _selected, user_a_trace = service.search("user-a-private-query", user_id="user-a")
    _selected, user_b_trace = service.search("user-b-query", user_id="user-b")

    assert b"user-a-private-query" not in (service.trace_root / f"{user_a_trace}.json").read_bytes()
    backend = RecallTraceEraseBackend(tmp_path)

    assert backend.erase_document(_request()) is True
    assert not (service.trace_root / f"{user_a_trace}.json").exists()
    assert (service.trace_root / f"{user_b_trace}.json").exists()
    assert backend.erase_document(_request()) is True


def test_recall_trace_erasure_uses_the_exact_nondefault_tenant_root(tmp_path: Path) -> None:
    tenant_root = tmp_path / "tenants" / "tenant-a" / "recall-traces"
    service = RetrievalService(_EmptyAssembler(), tenant_root)  # type: ignore[arg-type]
    _selected, trace_id = service.search("tenant-a-query", user_id="user-a", tenant_id="tenant-a")

    assert RecallTraceEraseBackend(tmp_path).erase_document(
        _request(tenant_id="tenant-a")
    ) is True
    assert not (tenant_root / f"{trace_id}.json").exists()


def test_recall_trace_erasure_rejects_a_symbolic_link_trace_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "recall-traces").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RecallTraceEraseIntegrityError, match="unsafe"):
        RecallTraceEraseBackend(tmp_path).erase_document(_request())

    assert list(outside.iterdir()) == []
