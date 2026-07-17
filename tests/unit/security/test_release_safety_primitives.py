from __future__ import annotations

import json
import multiprocessing
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from memoryos.api.limits import MAX_RETRIEVAL_LIMIT, MAX_TOKEN_BUDGET, bounded_int
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.service import RetrievalService
from memoryos.contextdb.session.commit_group import CommitGroupIntegrityError, CommitGroupStore
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
)
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.core.file_lock import open_private_lock
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.retrieval import CanonicalQueryIntent, OfflineCanonicalMemoryRetriever
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    MemoryClaim,
    MemoryRevision,
    TransitionProfile,
    revision_payload_with_effective_validity,
)
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.effect_marker import atomic_create_json, atomic_write_json
from memoryos.operations.commit.operation_committer import OperationCommitter
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
            uri="memoryos://user/u1/memories/nonfinite",
            context_type=ContextType.MEMORY,
            title="nonfinite",
            **kwargs,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_relation_and_vector_inputs_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ContextRelation(
            source_uri="memoryos://user/u1/memories/a",
            relation_type="related",
            target_uri="memoryos://user/u1/memories/b",
            weight=value,
        )
    vectors = InMemoryVectorStore()
    with pytest.raises(ValueError, match="finite"):
        vectors.upsert_vector("memoryos://user/u1/memories/a", [1.0, value])
    vectors.upsert_vector("memoryos://user/u1/memories/a", [1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        vectors.search_vector([value, 0.0], namespace="")


def test_regular_commit_rejects_unproved_canonical_target_and_canonical_source(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    relations = InMemoryRelationStore()
    committer = OperationCommitter(
        source,
        InMemoryIndexStore(),
        str(tmp_path),
        relation_store=relations,
        tenant_id="t1",
    )
    ordinary_uri = "memoryos://user/u1/memories/ordinary"
    canonical_uri = "memoryos://user/u1/memories/canonical/slots/s1/claims/c1"
    ordinary = ContextObject(
        uri=ordinary_uri,
        context_type=ContextType.MEMORY,
        title="ordinary",
        owner_user_id="u1",
        tenant_id="t1",
        relations=[
            ContextRelation(
                source_uri=ordinary_uri,
                relation_type="related_to",
                target_uri=canonical_uri,
                metadata={"tenant_id": "t1", "owner_user_id": "u1"},
            )
        ],
    )
    operation = ContextOperation(
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=ordinary_uri,
        user_id="u1",
        payload={
            "tenant_id": "t1",
            "context_object": ordinary.to_dict(),
            "content": "ordinary",
        },
    )

    with pytest.raises(FileNotFoundError, match="canonical object is not committed"):
        committer.commit("u1", [operation])
    with pytest.raises(FileNotFoundError):
        source.read_object(ordinary_uri)
    assert not relations.relations_of(ordinary_uri, tenant_id="t1")

    forged_authority = ContextObject(
        uri="memoryos://user/u1/memories/ordinary-forged-authority",
        context_type=ContextType.MEMORY,
        title="ordinary authority carrying a forged canonical Source edge",
        owner_user_id="u1",
        tenant_id="t1",
        relations=[
            ContextRelation(
                source_uri=canonical_uri,
                relation_type="forged_outgoing",
                target_uri=ordinary_uri,
                metadata={"tenant_id": "t1", "owner_user_id": "u1"},
            )
        ],
    )
    forged = ContextOperation(
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=forged_authority.uri,
        user_id="u1",
        payload={
            "tenant_id": "t1",
            "context_object": forged_authority.to_dict(),
            "content": "forged",
        },
    )
    with pytest.raises(ValueError, match="canonical Source relation"):
        committer.commit("u1", [forged])
    with pytest.raises(FileNotFoundError):
        source.read_object(forged_authority.uri)


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


def _revision(revision: int, *, valid_from: str, valid_to: str | None = None) -> MemoryRevision:
    return MemoryRevision(
        revision=revision,
        state="ACTIVE",
        value_fields={"canonical_value": f"value-{revision}"},
        evidence_refs=(EvidenceRef(f"e{revision}", None, f"hash-{revision}"),),
        proposal_id=f"p{revision}",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
        previous_revision=revision - 1 if revision > 1 else None,
        valid_from=valid_from,
        valid_to=valid_to,
        transaction_time=valid_from,
    )


def test_revision_intervals_require_timezone_and_positive_duration() -> None:
    with pytest.raises(ValueError, match="timezone"):
        _revision(1, valid_from="2026-01-01T00:00:00")
    with pytest.raises(ValueError, match="later"):
        _revision(
            1,
            valid_from="2026-01-01T00:00:00+00:00",
            valid_to="2026-01-01T00:00:00+00:00",
        )
    with pytest.raises(ValueError, match="later"):
        _revision(
            1,
            valid_from="2026-01-02T00:00:00+00:00",
            valid_to="2026-01-01T00:00:00+00:00",
        )


def test_equal_effective_time_transition_is_normalized_to_legal_interval() -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    first = _revision(1, valid_from=timestamp)
    claim = MemoryClaim(
        "claim",
        "memoryos://user/u1/memories/canonical/slots/slot/claims/claim",
        "slot",
        "value",
        TransitionProfile.AUTHORITATIVE_STATE,
        (first,),
    )
    updated = claim.with_revision(_revision(2, valid_from=timestamp))
    first_start = datetime.fromisoformat(updated.revisions[0].valid_from)
    second_start = datetime.fromisoformat(updated.revisions[1].valid_from)
    assert updated.revisions[0] == first
    assert updated.revisions[0].valid_to is None
    assert first_start < second_start


def test_derived_revision_validity_is_half_open_and_fails_closed_on_corruption() -> None:
    first = _revision(1, valid_from="2026-01-01T00:00:00+00:00")
    second = _revision(2, valid_from="2026-01-02T00:00:00+00:00")
    rows = (first.to_dict(), second.to_dict())

    effective = revision_payload_with_effective_validity(rows, 1)

    assert first.valid_to is None
    assert effective["valid_to"] == second.valid_from
    with pytest.raises(CanonicalMemoryInvariantError, match="duplicated"):
        revision_payload_with_effective_validity((first.to_dict(), first.to_dict()), 1)
    corrupt = first.to_dict()
    corrupt["valid_to"] = first.valid_from
    with pytest.raises(CanonicalMemoryInvariantError, match="later"):
        revision_payload_with_effective_validity((corrupt,), 1)


def test_current_effective_time_and_negated_intent_are_fail_closed(tmp_path: Path) -> None:
    retriever = OfflineCanonicalMemoryRetriever(
        FileSystemSourceStore(tmp_path),
        InMemoryIndexStore(),
        offline_admin=True,
    )
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    expired_start = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    expired_end = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    assert retriever._revision_is_effective({"valid_from": future, "valid_to": None}) is False
    assert retriever._revision_is_effective({"valid_from": expired_start, "valid_to": expired_end}) is False
    for text in ("没有冲突", "不看历史", "do not show alternatives", "without history"):
        assert retriever.classify_intent(text) == CanonicalQueryIntent.CURRENT


def test_commit_group_retry_is_finite_and_corrupt_state_is_quarantined(tmp_path: Path) -> None:
    store = CommitGroupStore(tmp_path)
    status = store.create(
        "group-a",
        task_id="task-a",
        archive_uri="memoryos://user/u1/sessions/history/a",
        user_id="u1",
        tenant_id="default",
    )
    assert status.canonical_status == "pending"
    for attempt in range(1, store.MAX_ATTEMPTS + 1):
        attempt_id = f"attempt-{attempt}"
        assert store.claim_canonical("group-a", attempt_id=attempt_id)
        status = store.fail_canonical(
            "group-a",
            "OSError",
            retryable=True,
            attempt_id=attempt_id,
        )
    assert status.canonical_status == "dead_letter"
    assert status.canonical_attempt_count == store.MAX_ATTEMPTS
    assert status.canonical_last_error == "OSError"
    assert status.canonical_next_retry_at == ""
    assert store.claim_canonical("group-a", attempt_id="stale") is False
    assert store.pending() == []

    broken = store.path("broken-group")
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{broken", encoding="utf-8")
    with pytest.raises(CommitGroupIntegrityError, match="quarantined"):
        store.load("broken-group")
    assert not broken.exists()
    quarantined = list((tmp_path / "system" / "quarantine" / "commit_group").glob("*.original"))
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
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=f"memoryos://user/u1/memories/{operation_id}",
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
    assert "sk-live-secret" not in trace["query"]
    assert "<redacted>" in trace["query"]
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
