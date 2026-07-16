from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, BrokenBarrierError

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.orchestrator import RetrievalUnavailableError
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import CanonicalMemoryRepository
from memoryos.memory.canonical.history import validate_canonical_receipt_history
from memoryos.operations.commit import operation_committer as committer_module
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.runtime.readiness import RuntimeReadinessState
from tests.unit.test_canonical_transaction_commit import (
    _artifact_root,
    _entity_aliases_proposal,
    _persisted_episode,
    _plan,
    _proposal,
    _setup,
)


def test_two_concurrent_writers_create_exactly_one_first_slot_revision(tmp_path: Path) -> None:
    _source, _index, _queue, _relations, _committer, sqlite_episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    postgres_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="writer-postgres",
            archive_uri="memoryos://user/u1/sessions/history/writer-postgres",
            messages=[
                {
                    "id": "m2",
                    "role": "user",
                    "content": "I confirm the primary storage backend is PostgreSQL.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
            task_id="writer-postgres-task",
        ),
    )
    proposals = (
        _proposal(sqlite_episode, "writer-sqlite", "SQLite", "confirmation", "confirmed"),
        _proposal(postgres_episode, "writer-postgres", "PostgreSQL", "confirmation", "confirmed"),
    )
    planned = []
    identities = []
    for episode, proposal in zip((sqlite_episode, postgres_episode), proposals, strict=True):
        identity, _transition, plan = _plan(runtime.source_store, episode, scope, proposal)
        identities.append(identity)
        planned.append(
            plan.to_context_operations(
                user_id="u1",
                tenant_id="t1",
                episode_id=episode.episode_id,
            )
        )
    assert identities[0].slot_uri == identities[1].slot_uri
    barrier = Barrier(2)

    def commit(operations):  # noqa: ANN001, ANN202
        barrier.wait(timeout=10)
        return runtime.committer.commit("u1", operations)

    outcomes: list[object] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(commit, operations) for operations in planned]
        for future in futures:
            try:
                outcomes.append(future.result(timeout=20))
            except Exception as exc:  # noqa: BLE001 - exact one-winner invariant below.
                outcomes.append(exc)

    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    slot, claims = CanonicalMemoryRepository(
        runtime.source_store,
        runtime.relation_store,
    ).load(identities[0])
    assert slot is not None and slot.revision == 1
    assert len(claims) == 1 and claims[0].current.state == "ACTIVE"
    receipts = list((_artifact_root(tmp_path) / "system" / "transactions").glob("*.json"))
    assert len(receipts) == 1


def test_same_idempotency_key_cannot_publish_two_different_slot_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The receipt identity must serialize independently of the Slot lock."""

    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    proposals = (
        _proposal(episode, "idempotency-slot-a", "SQLite", "confirmation", "confirmed"),
        _entity_aliases_proposal(episode, "idempotency-slot-b", ["sqlite3"]),
    )
    identities = []
    plans = []
    for proposal in proposals:
        identity, _transition, plan = _plan(runtime.source_store, episode, scope, proposal)
        identities.append(identity)
        plans.append(plan)
    assert identities[0].slot_uri != identities[1].slot_uri
    assert plans[0].transaction_id != plans[1].transaction_id

    shared_key = plans[0].idempotency_key
    plans[1] = replace(plans[1], idempotency_key=shared_key)
    batches = [
        plan.to_context_operations(
            user_id="u1",
            tenant_id="t1",
            episode_id=episode.episode_id,
        )
        for plan in plans
    ]

    # Synchronize immediately after both writers have passed their historical
    # exists() check.  With replace-based publication both writers can then
    # overwrite the same supposedly immutable receipt.
    receipt_barrier = Barrier(2)
    original_atomic_create = committer_module.atomic_create_json

    def synchronized_atomic_create(path, payload, *, artifact_root):  # noqa: ANN001, ANN202
        if path.parent.name == "transactions":
            try:
                receipt_barrier.wait(timeout=0.5)
            except BrokenBarrierError:
                # Once idempotency locking is correct only the winner reaches
                # publication; do not turn that expected serialization into a
                # test deadlock.
                pass
        return original_atomic_create(path, payload, artifact_root=artifact_root)

    monkeypatch.setattr(committer_module, "atomic_create_json", synchronized_atomic_create)
    start = Barrier(2)

    def commit(batch):  # noqa: ANN001, ANN202
        start.wait(timeout=10)
        return runtime.committer.commit("u1", batch)

    outcomes: list[object] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(commit, batch) for batch in batches]
        for future in futures:
            try:
                outcomes.append(future.result(timeout=20))
            except Exception as exc:  # noqa: BLE001 - one winner is asserted below.
                outcomes.append(exc)

    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    receipt_root = _artifact_root(tmp_path) / "system" / "transactions"
    assert len(list(receipt_root.glob("*.json"))) == 1
    committed = [
        CanonicalMemoryRepository(runtime.source_store, runtime.relation_store).load(identity)[0]
        for identity in identities
    ]
    assert sum(slot is not None for slot in committed) == 1
    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.READY


def test_same_operation_id_cannot_publish_two_different_regular_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared receipt identity is fenced even when target URIs differ."""

    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    shared_operation_id = "op_shared_immutable_identity"

    def operation(suffix: str) -> ContextOperation:
        uri = f"memoryos://user/u1/memories/candidates/{suffix}"
        obj = ContextObject(
            uri=uri,
            context_type=ContextType.MEMORY,
            title=suffix,
            tenant_id="t1",
            owner_user_id="u1",
        )
        return ContextOperation(
            context_type=ContextType.MEMORY,
            action=OperationAction.ADD,
            target_uri=uri,
            user_id="u1",
            operation_id=shared_operation_id,
            payload={"context_object": obj.to_dict(), "content": suffix},
        )

    operations = (operation("candidate-a"), operation("candidate-b"))
    publication_barrier = Barrier(2)
    original_atomic_create = committer_module.atomic_create_json

    def synchronized_atomic_create(path, payload, *, artifact_root):  # noqa: ANN001, ANN202
        if path.parent.name == "operations":
            try:
                publication_barrier.wait(timeout=0.5)
            except BrokenBarrierError:
                pass
        return original_atomic_create(path, payload, artifact_root=artifact_root)

    monkeypatch.setattr(committer_module, "atomic_create_json", synchronized_atomic_create)
    start = Barrier(2)

    def commit(item: ContextOperation):  # noqa: ANN202
        start.wait(timeout=10)
        return runtime.committer.commit("u1", [item])

    outcomes: list[object] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(commit, operation) for operation in operations]
        for future in futures:
            try:
                outcomes.append(future.result(timeout=20))
            except Exception as exc:  # noqa: BLE001 - one winner is asserted below.
                outcomes.append(exc)

    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    visible = []
    for operation_item in operations:
        try:
            visible.append(runtime.source_store.read_object(str(operation_item.target_uri)).uri)
        except FileNotFoundError:
            pass
    assert len(visible) == 1
    marker_root = _artifact_root(tmp_path) / "system" / "operations"
    assert len(list(marker_root.glob("*.json"))) == 1
    assert not list((_artifact_root(tmp_path) / "system" / "redo").glob("*.json"))
    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.READY


def test_recovery_and_idempotent_writer_race_converge_to_one_receipt(tmp_path: Path) -> None:
    _source, _index, _queue, _relations, _committer, episode, scope = _setup(tmp_path)
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    proposal = _proposal(episode, "recovery-writer-race", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(runtime.source_store, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )

    def crash_after_redo(stage: str, _transaction_id: str) -> None:
        if stage == "after_redo_begin":
            runtime.committer.test_hook = None
            raise SystemExit("crash after redo begin")

    runtime.committer.test_hook = crash_after_redo
    try:
        runtime.committer.commit("u1", operations)
    except SystemExit:
        pass
    else:  # pragma: no cover
        raise AssertionError("fault injection did not stop the writer")
    barrier = Barrier(2)

    def recover():  # noqa: ANN202
        barrier.wait(timeout=10)
        return runtime.recovery_worker.process_pending("u1")

    def replay():  # noqa: ANN202
        barrier.wait(timeout=10)
        return runtime.committer.commit("u1", operations)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(recover), pool.submit(replay))
        lock_conflicts = []
        for future in futures:
            try:
                future.result(timeout=20)
            except TimeoutError as exc:
                lock_conflicts.append(str(exc))
        assert len(lock_conflicts) <= 1
        assert all(message.startswith("Lock already held:") for message in lock_conflicts)

    runtime.recovery_worker.process_pending("u1")
    runtime.committer.commit("u1", operations)
    slot, claims = CanonicalMemoryRepository(
        runtime.source_store,
        runtime.relation_store,
    ).load(identity)
    assert slot is not None and slot.revision == 1 and len(claims) == 1
    artifact_root = _artifact_root(tmp_path)
    assert len(list((artifact_root / "system" / "transactions").glob("*.json"))) == 1
    assert not list((artifact_root / "system" / "redo").glob("*.json"))
    assert validate_canonical_receipt_history(artifact_root, tenant_id="t1")["receipts"] == 1


def test_rebuild_and_queries_share_only_committed_snapshots(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    barrier = Barrier(2)
    observed: list[list[str]] = []

    def rebuild() -> None:
        barrier.wait(timeout=10)
        for _ in range(10):
            client.context_db.rebuild_index()

    def query() -> None:
        barrier.wait(timeout=10)
        for _ in range(30):
            observed.append(
                [
                    item["uri"]
                    for item in client.search_context(
                        "PostgreSQL",
                        user_id="u1",
                        project_id="memoryos",
                        context_type="memory",
                    )
                ]
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        rebuild_future = pool.submit(rebuild)
        query_future = pool.submit(query)
        rebuild_future.result(timeout=30)
        query_future.result(timeout=30)

    assert observed
    assert all(rows == [committed["uri"]] for rows in observed)
    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path)))
    assert restarted.readiness.state == RuntimeReadinessState.READY


def test_bounded_final_candidate_validation_fails_closed_across_concurrent_head_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    identity = {"decision_topic": "primary storage backend"}
    old = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields=identity,
    )
    pending_result = client.remember(
        user_id="u1",
        content="MySQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields=identity,
    )
    assert old["status"] == "COMMITTED" and pending_result["status"] == "PENDING"
    pending = client.list_pending(user_id="u1", lifecycle_states=["PENDING"])[0]
    orchestrator = client._retrieval_orchestrator()
    original_resolve = orchestrator.resolver.resolve
    advanced = False

    def advance_then_validate(candidates, *, plan):  # noqa: ANN001, ANN202
        nonlocal advanced
        assert 0 < len(candidates) <= plan.candidate_limit
        if not advanced:
            advanced = True
            client.review_pending(
                user_id="u1",
                pending_uri=pending["uri"],
                decision="CONFIRM_AND_APPLY",
                expected_lifecycle_revision=pending["lifecycle_revision"],
                expected_proposal_fingerprint=pending["proposal_fingerprint"],
                command_id="advance-after-query-snapshot",
                reason="test committed head advance",
            )
        return original_resolve(candidates, plan=plan)

    monkeypatch.setattr(orchestrator.resolver, "resolve", advance_then_validate)
    monkeypatch.setattr(client, "_retrieval_orchestrator", lambda: orchestrator)
    with pytest.raises(
        RetrievalUnavailableError,
        match="Canonical Current candidate failed bounded authoritative validation",
    ) as unavailable:
        client.search_context(
            "PostgreSQL",
            user_id="u1",
            project_id="memoryos",
            context_type="memory",
        )
    assert advanced is True
    assert unavailable.value.degraded_modes == ("stale_canonical_current_projection",)

    after = client.search_context(
        "MySQL",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )
    assert len(after) == 1
    # Resolver egress is rebuilt from the receipt-proved current Claim rather
    # than the disposable Catalog candidate, so the authoritative value keeps
    # its Source spelling instead of a projection-side normalized token.
    assert after[0]["metadata"]["canonical_value"] == "MySQL"
