from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

from memoryos.api.limits import MAX_RETRIEVAL_LIMIT, MAX_TOKEN_BUDGET, bounded_int
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.service import RetrievalService
from memoryos.contextdb.session.commit_group import CommitGroupIntegrityError, CommitGroupStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.core.file_lock import open_private_lock
from memoryos.core.integrity import canonical_digest
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.effect_marker import atomic_create_json, atomic_write_json
from memoryos.operations.commit.quarantine import quarantine_control_file
from memoryos.operations.commit.redo_log import RedoLog
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def _open_fresh_private_lock_concurrently(
    root: str,
    lock_path: str,
    barrier: Any,
    results: Any,
) -> None:
    """Spawn-safe worker used to race the first lock-file publication."""

    try:
        barrier.wait(timeout=10)
        descriptor = open_private_lock(lock_path, root=root)
        os.close(descriptor)
        results.put("")
    except BaseException as exc:  # pragma: no cover - asserted in the parent process.
        results.put(f"{type(exc).__name__}: {exc}")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["hotness", "semantic_hotness", "behavior_support_hotness"])
def test_context_hotness_rejects_nonfinite_values(field: str, value: float) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match="finite"):
        ContextObject(
            uri="memoryos://user/u1/resources/nonfinite",
            context_type=ContextType.RESOURCE,
            title="nonfinite",
            **kwargs,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_relation_and_vector_inputs_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ContextRelation(
            source_uri="memoryos://user/u1/resources/a",
            relation_type="related",
            target_uri="memoryos://user/u1/resources/b",
            weight=value,
        )
    vectors = InMemoryVectorStore()
    with pytest.raises(ValueError, match="finite"):
        vectors.upsert_vector("memoryos://user/u1/resources/a", [1.0, value])
    vectors.upsert_vector("memoryos://user/u1/resources/a", [1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        vectors.search_vector([value, 0.0], namespace="")


def test_api_limits_share_one_hard_bound() -> None:
    assert (
        bounded_int(
            MAX_RETRIEVAL_LIMIT,
            default=10,
            minimum=0,
            maximum=MAX_RETRIEVAL_LIMIT,
            label="limit",
        )
        == MAX_RETRIEVAL_LIMIT
    )
    assert (
        bounded_int(
            MAX_TOKEN_BUDGET,
            default=2000,
            minimum=0,
            maximum=MAX_TOKEN_BUDGET,
            label="token_budget",
        )
        == MAX_TOKEN_BUDGET
    )
    with pytest.raises(ValueError, match="between"):
        bounded_int(101, default=10, minimum=0, maximum=MAX_RETRIEVAL_LIMIT, label="limit")
    with pytest.raises(ValueError, match="between"):
        bounded_int(
            MAX_TOKEN_BUDGET + 1,
            default=2000,
            minimum=0,
            maximum=MAX_TOKEN_BUDGET,
            label="token_budget",
        )


def test_commit_group_consumer_retry_is_finite_and_corrupt_state_is_quarantined(tmp_path: Path) -> None:
    store = CommitGroupStore(tmp_path)
    status = store.create(
        "group-a",
        task_id="task-a",
        archive_uri="memoryos://user/u1/sessions/history/a",
        user_id="u1",
        tenant_id="default",
        archive_digest=canonical_digest({"archive": "a"}),
        manifest_digest=canonical_digest({"manifest": "a"}),
    )
    assert status.consumers["memory"].status == "pending"
    for attempt in range(1, store.MAX_ATTEMPTS + 1):
        attempt_id = f"attempt-{attempt}"
        assert store.claim_consumer("group-a", "memory", attempt_id=attempt_id)
        status = store.fail_consumer(
            "group-a",
            "memory",
            "OSError",
            retryable=True,
            attempt_id=attempt_id,
        )
    memory = status.consumers["memory"]
    assert memory.status == "dead_letter"
    assert memory.attempt_count == store.MAX_ATTEMPTS
    assert memory.last_error == "OSError"
    assert memory.next_retry_at == ""
    assert store.claim_consumer("group-a", "memory", attempt_id="stale") is False

    broken = store.path("broken-group")
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{broken", encoding="utf-8")
    with pytest.raises(CommitGroupIntegrityError, match="quarantined"):
        store.load("broken-group")
    assert not broken.exists()
    quarantined = list((tmp_path / "system" / "quarantine" / "session_commit_group").glob("*.original"))
    assert len(quarantined) == 1


def test_quarantine_moves_a_control_symlink_without_following_its_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "system" / "target.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"preserve":true}', encoding="utf-8")
    link = tmp_path / "system" / "redo" / "forged.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)

    record = quarantine_control_file(
        tmp_path,
        link,
        kind="redo",
        error=ValueError("symbolic link is not a control record"),
    )

    assert target.read_text(encoding="utf-8") == '{"preserve":true}'
    assert not link.exists() and not link.is_symlink()
    quarantined = tmp_path / record.quarantined_relative_path
    assert quarantined.is_symlink()


def _regular_operation(operation_id: str) -> ContextOperation:
    return ContextOperation(
        operation_id=operation_id,
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.ADD,
        target_uri=f"memoryos://user/u1/resources/{operation_id}",
        payload={"tenant_id": "default"},
    )


def test_diff_writer_rejects_broken_control_symlink(tmp_path: Path) -> None:
    diff = ContextDiff(
        user_id="u1",
        operations=[_regular_operation("diff-link-operation")],
        diff_id="diff-link",
    )
    path = tmp_path / "system" / "diffs" / "diff-link.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "missing-diff-target.json"
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        DiffWriter(tmp_path).write(diff)

    assert path.is_symlink()
    assert not target.exists()


def test_immutable_control_writer_rejects_a_parent_directory_symlink(
    tmp_path: Path,
) -> None:
    redirected = tmp_path / "redirected-artifacts"
    redirected.mkdir()
    parent = tmp_path / "system" / "immutable-artifacts"
    parent.parent.mkdir(parents=True)
    parent.symlink_to(redirected, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        atomic_create_json(
            parent / "proof.json",
            {"proof": "must-stay-in-boundary"},
            artifact_root=tmp_path,
        )

    assert not (redirected / "proof.json").exists()


def test_immutable_control_writer_allows_platform_alias_before_artifact_root(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical-volume"
    artifact_root = physical / "tenant-root"
    artifact_root.mkdir(parents=True)
    platform_alias = tmp_path / "platform-alias"
    platform_alias.symlink_to(physical, target_is_directory=True)
    path = platform_alias / "tenant-root" / "system" / "proofs" / "proof.json"

    assert (
        atomic_create_json(
            path,
            {"proof": "inside-trusted-root"},
            artifact_root=platform_alias / "tenant-root",
        )
        is True
    )

    assert (artifact_root / "system" / "proofs" / "proof.json").exists()


def test_private_lock_allows_normalized_platform_alias_before_artifact_root(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical-lock-volume"
    artifact_root = physical / "tenant-root"
    artifact_root.mkdir(parents=True)
    platform_alias = tmp_path / "lock-platform-alias"
    platform_alias.symlink_to(physical, target_is_directory=True)
    resolved_lock = artifact_root.resolve() / "system" / "locks" / "proof.lock"

    descriptor = open_private_lock(
        resolved_lock,
        root=platform_alias / "tenant-root",
    )
    os.close(descriptor)

    assert resolved_lock.is_file()


def test_private_lock_first_publication_is_multiprocess_safe(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    process_count = 8
    barrier = context.Barrier(process_count)
    results = context.Queue()
    artifact_root = tmp_path / "tenant-root"
    lock_path = artifact_root / "system" / "locks" / "fresh.lock"
    processes = [
        context.Process(
            target=_open_fresh_private_lock_concurrently,
            args=(str(artifact_root), str(lock_path), barrier, results),
        )
        for _ in range(process_count)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    failures = [results.get(timeout=5) for _ in range(process_count)]
    assert failures == [""] * process_count
    assert lock_path.is_file()


@pytest.mark.parametrize("mutable", [False, True])
def test_control_writer_rejects_repeated_reserved_name_after_parent_symlink(
    tmp_path: Path,
    mutable: bool,
) -> None:
    artifact_root = tmp_path / "tenant-root"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    redirected = artifact_root / "views" / "scope" / "redirect"
    redirected.parent.mkdir(parents=True)
    redirected.symlink_to(foreign, target_is_directory=True)
    path = redirected / "nested" / "views" / "proof.json"

    with pytest.raises(ValueError, match="symbolic link"):
        if mutable:
            atomic_write_json(path, {"proof": "mutable"}, artifact_root=artifact_root)
        else:
            atomic_create_json(path, {"proof": "immutable"}, artifact_root=artifact_root)

    assert not (foreign / "nested" / "views" / "proof.json").exists()


def test_redo_writer_rejects_broken_control_symlink(tmp_path: Path) -> None:
    operation = _regular_operation("redo-link-operation")
    path = tmp_path / "system" / "redo" / f"{operation.operation_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "missing-redo-target.json"
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        RedoLog(tmp_path).begin(operation)

    assert path.is_symlink()
    assert not target.exists()


def test_audit_writer_rejects_broken_control_symlink(tmp_path: Path) -> None:
    path = tmp_path / "system" / "audit" / "u1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "missing-audit-target.jsonl"
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        AuditWriter(tmp_path).record("u1", "test", {"operation_id": "audit-link"})

    assert path.is_symlink()
    assert not target.exists()


@pytest.mark.parametrize("group_id", ["../escape", "nested/group", "", ".", ".."])
def test_commit_group_rejects_unsafe_control_and_lock_paths(
    tmp_path: Path,
    group_id: str,
) -> None:
    store = CommitGroupStore(tmp_path)

    with pytest.raises(ValueError, match="safe.*path segment"):
        store.load(group_id)
    with pytest.raises(ValueError, match="safe.*path segment"):
        with store.group_lock(group_id):
            pass

    assert not (tmp_path / "system" / "escape.json").exists()
    assert not (tmp_path / "system" / "commit_groups" / "nested").exists()


def test_commit_group_lock_rejects_broken_symbolic_link(tmp_path: Path) -> None:
    store = CommitGroupStore(tmp_path)
    lock_path = store.root / ".locks" / "linked-group.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    missing_target = tmp_path / "missing-lock-target"
    lock_path.symlink_to(missing_target)

    with pytest.raises(RuntimeError, match="symbolic link"):
        with store.group_lock("linked-group"):
            pass

    assert lock_path.is_symlink()
    assert not missing_target.exists()


def test_recall_trace_redacts_secret_query_and_uses_private_file(tmp_path: Path) -> None:
    class EmptyAssembler:
        reranker = None

        def search(self, _query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    service = RetrievalService(EmptyAssembler(), tmp_path / "traces")  # type: ignore[arg-type]
    _results, trace_id = service.search(
        "OPENAI_API_KEY=sk-live-secret",
        user_id="u1",
        tenant_id="default",
    )
    trace = service.read_trace(trace_id)
    assert "query" not in trace
    assert trace["query_digest"] == hashlib.sha256(
        b"OPENAI_API_KEY=sk-live-secret"
    ).hexdigest()
    assert trace["query_utf8_bytes"] == len(b"OPENAI_API_KEY=sk-live-secret")
    path = service.trace_root / f"{trace_id}.json"
    assert path.stat().st_mode & 0o777 == 0o600


def test_recall_trace_sanitizes_all_fields_and_fails_closed(tmp_path: Path) -> None:
    class SecretAssembler:
        reranker = None

        def search(self, _query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "uri": "memoryos://resources/report",
                    "metadata": {"authorization": "Bearer never-write-this-token"},
                    "layer": "L0",
                }
            ]

    service = RetrievalService(SecretAssembler(), tmp_path / "traces")  # type: ignore[arg-type]
    _results, trace_id = service.search(
        "report",
        user_id="u1",
        tenant_id="default",
        connect_filters={"cookie": "session=never-write-this-cookie"},
    )
    encoded = json.dumps(service.read_trace(trace_id), ensure_ascii=False)
    assert "never-write-this-token" not in encoded
    assert "never-write-this-cookie" not in encoded

    class BrokenSanitizer:
        def sanitize_trace(self, _value: Any) -> Any:
            raise ValueError("sanitization unavailable")

    service.sanitizer = BrokenSanitizer()  # type: ignore[assignment]
    before = set(service.trace_root.glob("*.json"))
    with pytest.raises(ValueError, match="sanitization unavailable"):
        service.search("safe query", user_id="u1", tenant_id="default")
    assert set(service.trace_root.glob("*.json")) == before
