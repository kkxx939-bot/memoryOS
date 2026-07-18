from __future__ import annotations

import json

import pytest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.core.ids import require_safe_path_segment
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.redo_log import RedoControlFileError, RedoIntegrityError, RedoLog
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def _add(tenant_id: str = "default", *, operation_id: str = "ordinary-add") -> ContextOperation:
    obj = ContextObject(
        uri="memoryos://user/u1/behavior_cases/recovery",
        context_type=ContextType.BEHAVIOR_CASE,
        title="recovery",
        owner_user_id="u1",
        tenant_id=tenant_id,
    )
    return ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"tenant_id": tenant_id, "context_object": obj.to_dict(), "content": "durable"},
        operation_id=operation_id,
    )


def test_artifact_identifiers_reject_path_escape_before_io(tmp_path) -> None:  # noqa: ANN001
    for value in ("", ".", "..", "../escape", "..\\escape", "nul\x00escape"):
        with pytest.raises(ValueError):
            require_safe_path_segment(value, "artifact_id")
    operation = _add()
    operation.operation_id = "../escape"
    committer = OperationCommitter(
        FileSystemSourceStore(tmp_path), InMemoryIndexStore(), str(tmp_path)
    )
    with pytest.raises(ValueError, match="operation_id"):
        committer.commit("u1", [operation])
    assert not (tmp_path / "system").exists()

    safe = _add(operation_id="safe")
    safe.operation_id = "../redo"
    with pytest.raises(ValueError, match="operation_id"):
        RedoLog(tmp_path).begin(safe)
    diff = ContextDiff(user_id="u1", diff_id="safe-diff")
    diff.diff_id = "../diff"
    with pytest.raises(ValueError, match="diff_id"):
        DiffWriter(tmp_path).write(diff)
    with pytest.raises(ValueError, match="user_id"):
        AuditWriter(tmp_path).record("../user", "event", {})


def test_nondefault_tenant_artifacts_are_physically_isolated(tmp_path) -> None:  # noqa: ANN001
    source_a = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    source_b = FileSystemSourceStore(tmp_path, tenant_id="tenant-b")
    committer_a = OperationCommitter(source_a, InMemoryIndexStore(), str(tmp_path))
    committer_b = OperationCommitter(source_b, InMemoryIndexStore(), str(tmp_path))
    operation_a = _add("tenant-a", operation_id="same-operation")
    operation_b = _add("tenant-b", operation_id="same-operation")

    committer_a.commit("u1", [operation_a])
    committer_b.commit("u1", [operation_b])

    marker = "system/operations/same-operation.json"
    assert (tmp_path / "tenants" / "tenant-a" / marker).exists()
    assert (tmp_path / "tenants" / "tenant-b" / marker).exists()
    assert not (tmp_path / marker).exists()


def test_cross_tenant_redo_is_never_resumed(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    committer = OperationCommitter(source, InMemoryIndexStore(), str(tmp_path))
    operation = _add("tenant-b", operation_id="cross-tenant")
    committer.redo.begin(operation, phase="started")

    result = RecoveryService(committer.redo, committer).recover("u1")

    assert result.recovered_count == 0
    assert result.quarantine_count == 1
    assert not committer.redo.pending_entries()


def test_started_phase_adopts_exact_source_effect_after_phase_advance_crash(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    operation = _add()
    committer.redo.begin(operation, phase="started")
    committer._apply_source(operation)

    result = RecoveryService(committer.redo, committer).recover("u1")

    assert result.operation_ids == [operation.operation_id]
    assert source.read_content(str(operation.target_uri)) == "durable"
    assert index.search("durable", tenant_id="default", filters={"owner_user_id": "u1"})


def test_source_effect_tampering_is_quarantined_before_index(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    operation = _add()
    committer._apply_source(operation)
    committer.redo.begin(
        operation,
        phase="source_written",
        source_effect=committer._capture_regular_source_effect(operation),
    )
    source.write_content(str(operation.target_uri), "tampered")

    result = RecoveryService(committer.redo, committer).recover("u1")

    assert result.recovered_count == 0
    assert result.quarantine_count == 1
    assert not index.indexed_uris(tenant_id="default")
    assert list((tmp_path / "system" / "quarantine" / "redo").glob("*.json"))


def test_source_written_delete_finishes_index_removal(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    obj = ContextObject.from_dict(_add().payload["context_object"])
    source.write_object(obj, content="delete me")
    index.upsert_index(obj, content="delete me", tenant_id="default")
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.DELETE,
        target_uri=obj.uri,
        payload={"tenant_id": "default"},
        operation_id="delete-recovery",
    )
    committer._apply_source(operation)
    committer.redo.begin(
        operation,
        phase="source_written",
        source_effect=committer._capture_regular_source_effect(operation),
    )

    result = RecoveryService(committer.redo, committer).recover("u1")

    assert result.operation_ids == [operation.operation_id]
    assert source.read_object(obj.uri).lifecycle_state.value == "deleted"
    assert not index.indexed_uris(tenant_id="default")


def test_recovery_never_claims_another_users_entry(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    obj = ContextObject(
        uri="memoryos://user/u2/behavior_cases/private",
        context_type=ContextType.BEHAVIOR_CASE,
        title="private",
        owner_user_id="u2",
    )
    operation = ContextOperation(
        user_id="u2",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"context_object": obj.to_dict(), "content": "u2 only"},
        operation_id="u2-recovery",
    )
    committer._apply_source(operation)
    committer.redo.begin(
        operation,
        phase="source_written",
        source_effect=committer._capture_regular_source_effect(operation),
    )

    assert RecoveryService(committer.redo, committer).recover("u1").recovered_count == 0
    assert committer.redo.pending_entries()
    assert RecoveryService(committer.redo, committer).recover("u2").operation_ids == [operation.operation_id]


def test_document_owned_redo_fails_closed_and_is_quarantined(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    committer = OperationCommitter(source, InMemoryIndexStore(), str(tmp_path))
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.DELETE,
        target_uri="memoryos://user/u1/memory/documents/018f47c0-7c55-7b09-8f6e-123456789abc",
        payload={},
        operation_id="forbidden-document-redo",
    )
    committer.redo.begin(operation, phase="started")

    with pytest.raises(RedoIntegrityError):
        committer.resume("u1", operation, "started")
    result = RecoveryService(committer.redo, committer).recover("u1")
    assert result.quarantine_count == 1


def test_redo_payload_is_digest_protected(tmp_path) -> None:  # noqa: ANN001
    redo = RedoLog(tmp_path)
    path = redo.begin(_add(), phase="started")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["user_id"] = "forged"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RedoControlFileError):
        redo.pending_entries()
