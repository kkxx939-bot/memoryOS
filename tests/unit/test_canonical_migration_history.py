from __future__ import annotations

import json
from pathlib import Path

import pytest

import memoryos.memory.canonical.migration as migration_module
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.planning_envelope import PlanningEnvelopeStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import CanonicalMemoryRepository
from memoryos.memory.canonical.current_head import publish_current_head_sets
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.history import (
    CanonicalHistoryIntegrityError,
    validate_canonical_receipt_history,
)
from memoryos.memory.canonical.migration import MemoryClosureMigration, MemoryClosureMigrationError
from memoryos.operations.commit.effect_marker import (
    atomic_write_json,
    build_marker,
    object_effect_from_store,
)
from memoryos.operations.commit.receipt import TRANSACTION_RECEIPT_SCHEMA_VERSION
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.runtime.readiness import RuntimeReadinessState
from tests.unit.test_canonical_transaction_commit import (
    _artifact_root,
    _persisted_episode,
    _plan,
    _proposal,
    _replacement_proposal,
    _reviewed_resolution_plan,
    _setup,
)


def _write_one_legacy_canonical_transaction(
    tmp_path: Path,
    *,
    invalid_active_claim: bool = False,
):  # noqa: ANN202
    source, _index, _queue, relations, _committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "legacy-canonical", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    if invalid_active_claim:
        slot_operation = next(
            operation
            for operation in operations
            if dict(operation.payload["context_object"]["metadata"]).get("canonical_kind") == "slot"
        )
        slot_operation.payload["context_object"]["metadata"]["active_claim_id"] = "forged-active-claim"
    for operation in operations:
        obj = ContextObject.from_dict(operation.payload["context_object"])
        source.write_object(obj, content=str(operation.payload.get("content", "")))
    diff = ContextDiff(
        user_id="u1",
        operations=operations,
        diff_id="legacy_canonical_diff",
        created_at=operations[0].created_at,
    )
    marker = build_marker(
        transaction_id=str(operations[0].payload["transaction_id"]),
        idempotency_key=str(operations[0].payload["idempotency_key"]),
        tenant_id="t1",
        user_id="u1",
        operation_ids=[operation.operation_id for operation in operations],
        object_effects=[
            object_effect_from_store(source, str(operation.target_uri), operation_type="update")
            for operation in operations
        ],
        relation_effects=[],
        diff=diff.to_dict(),
        operations=[operation.to_dict() for operation in operations],
    )
    path = _artifact_root(tmp_path) / "system" / "transactions" / f"{operations[0].payload['idempotency_key']}.json"
    atomic_write_json(path, marker, artifact_root=_artifact_root(tmp_path))
    return source, relations, identity, path, marker


def test_legacy_canonical_marker_migrates_to_receipt_head_and_external_diff(
    tmp_path: Path,
) -> None:
    source, relations, identity, path, marker = _write_one_legacy_canonical_transaction(tmp_path)

    migration = MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    ).run()

    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == TRANSACTION_RECEIPT_SCHEMA_VERSION
    assert receipt["migration_source_marker_digest"] == marker["marker_digest"]
    assert (_artifact_root(tmp_path) / "system" / "diffs" / "legacy_canonical_diff.json").exists()
    assert CanonicalMemoryRepository(source, relations).load(identity)[0] is not None
    assert migration["consistency_check"] == "passed"
    assert validate_canonical_receipt_history(_artifact_root(tmp_path), tenant_id="t1")["receipts"] == 1
    assert (
        MemoryClosureMigration(
            tmp_path,
            tenant_id="t1",
            source_store=source,
            relation_store=relations,
        ).run()
        == migration
    )


def test_startup_rejects_cryptographically_valid_legacy_domain_invariant_violation(
    tmp_path: Path,
) -> None:
    _write_one_legacy_canonical_transaction(tmp_path, invalid_active_claim=True)

    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert runtime.readiness.state == RuntimeReadinessState.NOT_READY
    assert "active_claim_id" in " ".join(runtime.readiness.reasons)


def test_unexpected_startup_recovery_error_returns_explicit_not_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class UnexpectedStartupFailure(Exception):
        pass

    def fail_migration(_self, *, allow_inflight: bool = False):  # noqa: ANN001, ANN202, ARG001
        raise UnexpectedStartupFailure("unclassified recovery failure")

    monkeypatch.setattr(MemoryClosureMigration, "run", fail_migration)

    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert runtime.readiness.state == RuntimeReadinessState.NOT_READY
    assert runtime.readiness.reasons == ("UnexpectedStartupFailure: unclassified recovery failure",)


def test_historical_receipt_requires_its_immutable_external_diff(tmp_path: Path) -> None:
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "diff-proof", "SQLite", "confirmation", "confirmed")
    _identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committed = committer.commit("u1", operations)
    artifact_root = _artifact_root(tmp_path)
    validate_canonical_receipt_history(artifact_root, tenant_id="t1")
    diff_path = artifact_root / "system" / "diffs" / f"{committed.diff_id}.json"
    original = json.loads(diff_path.read_text(encoding="utf-8"))
    tampered = {**original, "created_at": "tampered"}
    atomic_write_json(diff_path, tampered, artifact_root=artifact_root)

    with pytest.raises(CanonicalHistoryIntegrityError, match="diff artifact is corrupt"):
        validate_canonical_receipt_history(artifact_root, tenant_id="t1")

    atomic_write_json(diff_path, original, artifact_root=artifact_root)
    diff_path.unlink()
    with pytest.raises(CanonicalHistoryIntegrityError, match="missing or unreadable"):
        validate_canonical_receipt_history(artifact_root, tenant_id="t1")


def test_receipt_alias_path_is_rejected_before_history_is_assembled(tmp_path: Path) -> None:
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "receipt-path", "SQLite", "confirmation", "confirmed")
    _identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)
    artifact_root = _artifact_root(tmp_path)
    receipt_path = committer._transaction_marker(str(operations[0].payload["idempotency_key"]))
    alias_path = receipt_path.with_name("receipt-alias.json")
    atomic_write_json(
        alias_path,
        json.loads(receipt_path.read_text(encoding="utf-8")),
        artifact_root=artifact_root,
    )

    with pytest.raises(CanonicalHistoryIntegrityError, match="receipt path identity"):
        validate_canonical_receipt_history(artifact_root, tenant_id="t1")


def test_migration_completion_rejects_late_legacy_canonical_marker(tmp_path: Path) -> None:
    first = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert first.readiness.state == RuntimeReadinessState.READY
    marker_core = {
        "schema_version": "effect_marker_v1",
        "status": "committed",
        "tenant_id": "t1",
        "operations": [{"payload": {"canonical_memory": True}}],
    }
    marker = {**marker_core, "marker_digest": canonical_digest(marker_core)}
    path = _artifact_root(tmp_path) / "system" / "transactions" / "late-legacy.json"
    atomic_write_json(path, marker, artifact_root=_artifact_root(tmp_path))

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert any("appeared after migration" in reason for reason in restarted.readiness.reasons)
    assert (_artifact_root(tmp_path) / "system" / "migrations" / "memory-closure-v1.failed.json").exists()


def test_only_regular_legacy_marker_is_preserved(tmp_path: Path) -> None:
    source, _index, _queue, relations, _committer, _episode, _scope = _setup(tmp_path)
    marker_core = {
        "schema_version": "effect_marker_v1",
        "status": "committed",
        "tenant_id": "t1",
        "operations": [{"payload": {"ordinary_memory": True}}],
    }
    marker = {**marker_core, "marker_digest": canonical_digest(marker_core)}
    path = _artifact_root(tmp_path) / "system" / "operations" / "regular.json"
    atomic_write_json(path, marker, artifact_root=_artifact_root(tmp_path))

    MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    ).run()

    assert json.loads(path.read_text(encoding="utf-8")) == marker


def test_interrupted_legacy_migration_resumes_without_rewriting_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, relations, _identity, path, _marker = _write_one_legacy_canonical_transaction(tmp_path)
    original_publish = migration_module.publish_current_head_sets
    interrupted = False

    def publish_then_interrupt(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal interrupted
        published = original_publish(*args, **kwargs)
        if not interrupted:
            interrupted = True
            raise SystemExit("migration process interrupted after head publication")
        return published

    monkeypatch.setattr(migration_module, "publish_current_head_sets", publish_then_interrupt)
    with pytest.raises(SystemExit, match="migration process interrupted"):
        MemoryClosureMigration(
            tmp_path,
            tenant_id="t1",
            source_store=source,
            relation_store=relations,
        ).run()
    converted = path.read_bytes()
    assert json.loads(converted)["schema_version"] == TRANSACTION_RECEIPT_SCHEMA_VERSION

    monkeypatch.setattr(migration_module, "publish_current_head_sets", original_publish)
    completed = MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    ).run()

    assert completed["status"] == "completed"
    assert path.read_bytes() == converted
    assert (
        MemoryClosureMigration(
            tmp_path,
            tenant_id="t1",
            source_store=source,
            relation_store=relations,
        ).run()
        == completed
    )


def test_empty_memory_directory_migration_is_idempotent(tmp_path: Path) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert runtime.readiness.state == RuntimeReadinessState.READY
    first = dict(runtime.readiness.details["migration"])

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.READY
    assert restarted.readiness.details["migration"] == first
    assert first["migrated_markers"] == []
    assert first["published_heads"] == []


@pytest.mark.parametrize(
    "receipt_name",
    (
        "memory-closure-v1.json",
        "memory-projection-v5.json",
        "memory-planning-v2.json",
    ),
)
def test_startup_rejects_migration_receipt_symbolic_link(
    tmp_path: Path,
    receipt_name: str,
) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert runtime.readiness.state == RuntimeReadinessState.READY
    receipt_path = _artifact_root(tmp_path) / "system" / "migrations" / receipt_name
    preserved = tmp_path / f"preserved-{receipt_name}"
    preserved.write_bytes(receipt_path.read_bytes())
    receipt_path.unlink()
    receipt_path.symlink_to(preserved)

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "symbolic link" in " ".join(restarted.readiness.reasons)
    assert preserved.exists()


def test_startup_rejects_broken_migration_receipt_symbolic_link(tmp_path: Path) -> None:
    artifact_root = _artifact_root(tmp_path)
    receipt_path = artifact_root / "system" / "migrations" / "memory-planning-v2.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    missing_target = tmp_path / "missing-planning-migration-receipt.json"
    receipt_path.symlink_to(missing_target)

    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert runtime.readiness.state == RuntimeReadinessState.NOT_READY
    assert "symbolic link" in " ".join(runtime.readiness.reasons)
    assert receipt_path.is_symlink()
    assert not missing_target.exists()


def test_orphan_raw_canonical_source_without_head_forces_not_ready(tmp_path: Path) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    orphan = ContextObject(
        uri="memoryos://user/u1/memories/pending/orphan-without-proof",
        context_type=ContextType.MEMORY,
        title="unproved pending",
        owner_user_id="u1",
        tenant_id="t1",
        metadata={"untrusted": True},
    )
    runtime.source_store.write_object(orphan, content="uncommitted")

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "no immutable receipt/current head proof" in " ".join(restarted.readiness.reasons)


def test_legacy_projection_record_is_quarantined_and_formally_rebuilt(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="t1")
    result = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    claim_uri = result["uri"]
    record_store = client.memory_projection_worker.projector.record_store
    current = record_store.load_current(claim_uri)
    assert current is not None
    old_attempt_id = current.projection_attempt_id
    attempt_path = record_store.attempt_path_for(current)
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "canonical_projection_v2"
    atomic_write_json(
        attempt_path,
        payload,
        artifact_root=_artifact_root(tmp_path),
    )

    restarted = MemoryOSClient(str(tmp_path), tenant_id="t1")

    assert restarted.readiness.state == RuntimeReadinessState.READY
    rebuilt = restarted.memory_projection_worker.projector.record_store.load_current(claim_uri)
    assert rebuilt is not None
    assert rebuilt.projection_attempt_id != old_attempt_id
    assert list((_artifact_root(tmp_path) / "system" / "quarantine" / "legacy_projection_record").glob("*.original"))
    projection_receipt_path = _artifact_root(tmp_path) / "system" / "migrations" / "memory-projection-v5.json"
    projection_receipt = json.loads(projection_receipt_path.read_text(encoding="utf-8"))
    core = {key: value for key, value in projection_receipt.items() if key != "migration_digest"}
    assert projection_receipt["schema_version"] == "memory_projection_migration_v5"
    assert projection_receipt["target_record_schema"] == "canonical_projection_v5"
    assert projection_receipt["target_pointer_schema"] == "canonical_projection_current_v5"
    assert projection_receipt["migration_digest"] == canonical_digest(core)


def test_first_projection_v5_migration_receipt_records_quarantined_legacy_state(
    tmp_path: Path,
) -> None:
    source, _index, _queue, relations, _committer, _episode, _scope = _setup(tmp_path)
    artifact_root = _artifact_root(tmp_path)
    legacy_path = (
        artifact_root
        / "system"
        / "projection-state"
        / "aa"
        / ("a" * 64)
        / "revisions"
        / "rev-1"
        / ("attempt-" + "b" * 32 + ".json")
    )
    atomic_write_json(
        legacy_path,
        {
            "schema_version": "canonical_projection_v4",
            "claim_uri": "memoryos://user/u1/memories/canonical/slots/s/claims/c",
            "source_revision": 1,
            "projection_attempt_id": "b" * 32,
        },
        artifact_root=artifact_root,
    )

    MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    ).run()

    receipt_path = artifact_root / "system" / "migrations" / "memory-projection-v5.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    core = {key: value for key, value in receipt.items() if key != "migration_digest"}
    assert receipt["migration_digest"] == canonical_digest(core)
    assert receipt["rebuild_required"] is True
    assert len(receipt["quarantined_legacy_artifacts"]) == 1
    assert not legacy_path.exists()
    assert list((artifact_root / "system" / "quarantine" / "legacy_projection_record").glob("*.original"))


def test_unproved_planning_envelope_is_quarantined_with_sticky_migration_failure(
    tmp_path: Path,
) -> None:
    source, _index, _queue, relations, _committer, _episode, _scope = _setup(tmp_path)
    artifact_root = _artifact_root(tmp_path)
    store = PlanningEnvelopeStore(tmp_path, tenant_id="t1")
    legacy_path = store.path("legacy-planning-task")
    atomic_write_json(
        legacy_path,
        {
            "schema_version": "memory_planning_envelope_v1",
            "task_id": "legacy-planning-task",
            "tenant_id": "t1",
        },
        artifact_root=artifact_root,
    )
    migration = MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    )

    with pytest.raises(MemoryClosureMigrationError, match="planning envelope migration failed"):
        migration.run()

    assert not legacy_path.exists()
    assert list((artifact_root / "system" / "quarantine" / "legacy_planning_envelope").glob("*.original"))
    failure_path = artifact_root / "system" / "migrations" / "memory-planning-v2.failed.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    core = {key: value for key, value in failure.items() if key != "migration_digest"}
    assert failure["status"] == "failed"
    assert failure["migration_digest"] == canonical_digest(core)
    with pytest.raises(MemoryClosureMigrationError, match="previously failed"):
        migration.run()


def test_startup_migration_then_recovers_inflight_canonical_redo(tmp_path: Path) -> None:
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "migration-redo", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )

    def crash_after_source(stage: str, _transaction_id: str) -> None:
        if stage == "after_source_effect":
            raise SystemExit("migration crash after Source publication")

    committer.test_hook = crash_after_source
    with pytest.raises(SystemExit, match="migration crash"):
        committer.commit("u1", operations)
    assert list((_artifact_root(tmp_path) / "system" / "redo").glob("*.json"))

    recovered = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))

    assert recovered.readiness.state == RuntimeReadinessState.READY, recovered.readiness.reasons
    slot, claims = CanonicalMemoryRepository(
        recovered.source_store,
        recovered.relation_store,
    ).load(identity)
    assert slot is not None and len(claims) == 1
    assert not list((_artifact_root(tmp_path) / "system" / "redo").glob("*.json"))
    receipt_paths = list((_artifact_root(tmp_path) / "system" / "transactions").glob("*.json"))
    receipt_bytes = [path.read_bytes() for path in receipt_paths]

    second = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert second.readiness.state == RuntimeReadinessState.READY
    assert [path.read_bytes() for path in receipt_paths] == receipt_bytes


def test_migration_rebuilds_latest_head_from_multi_revision_legacy_history(tmp_path: Path) -> None:
    source, _index, _queue, relations, committer, episode, scope = _setup(tmp_path)
    first = _proposal(episode, "legacy-history-one", "SQLite", "confirmation", "confirmed")
    identity, _transition, first_plan = _plan(source, episode, scope, first)
    first_operations = first_plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", first_operations)
    active = CanonicalMemoryRepository(source, relations).load(identity)[1][0]
    second_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="legacy-history-two",
            archive_uri="memoryos://user/u1/sessions/history/legacy-history-two",
            messages=[
                {
                    "id": "m2",
                    "role": "user",
                    "content": "I formally change the primary storage backend from SQLite to PostgreSQL now.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
            task_id="legacy-history-two-task",
        ),
    )
    second = _replacement_proposal(second_episode, "legacy-history-two", "PostgreSQL", active)
    second_plan = _reviewed_resolution_plan(
        source,
        committer,
        second_episode,
        second,
        command_suffix="legacy-history-two",
    )
    second_operations = list(second_plan.operations)
    committer.commit("u1", second_operations)
    artifact_root = _artifact_root(tmp_path)
    transaction_receipt_paths = sorted((artifact_root / "system" / "transactions").glob("*.json"))
    assert len(transaction_receipt_paths) == 2
    receipt_paths = transaction_receipt_paths + sorted((artifact_root / "system" / "operations").glob("*.json"))
    canonical_diff_names: set[str] = set()
    for path in receipt_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        canonical_diff_names.add(f"{receipt['diff']['diff_id']}.json")
        marker = build_marker(
            transaction_id=str(receipt["transaction_id"]),
            idempotency_key=str(receipt["idempotency_key"]),
            tenant_id="t1",
            user_id="u1",
            operation_ids=list(receipt["operation_ids"]),
            object_effects=[
                {
                    "operation_type": "update",
                    "uri": snapshot["uri"],
                    "expected_exists": True,
                    "object_digest": snapshot["object_digest"],
                    "content_digest": snapshot["content_digest"],
                }
                for snapshot in receipt["effect_snapshots"]
            ],
            relation_effects=list(receipt["relation_effects"]),
            diff=dict(receipt["diff"]),
            operations=list(receipt["operations"]),
        )
        atomic_write_json(path, marker, artifact_root=artifact_root)
        # This fixture models a deployment that only ever published the
        # legacy marker.  A committed v4 outbox is proof that a current
        # head had already existed, so retaining it while deleting the
        # head would correctly be treated as corruption rather than a
        # legacy migration input.
        outbox_path = artifact_root / "system" / "outbox" / f"{receipt['transaction_id']}.json"
        if outbox_path.exists():
            outbox_path.unlink()
        for intent_directory in ("canonical-prepared-intents", "prepared-intents"):
            intent_path = artifact_root / "system" / intent_directory / f"{receipt['transaction_id']}.json"
            if intent_path.exists():
                intent_path.unlink()
    # Every memory receipt in this fixture was rewritten as a legacy marker,
    # so no current-schema lifecycle head may survive the simulated upgrade.
    for current_head_path in (artifact_root / "system" / "current-heads").glob("*.json"):
        current_head_path.unlink()
    for diff_name in canonical_diff_names:
        (artifact_root / "system" / "diffs" / diff_name).unlink()

    result = MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    ).run()

    assert result["status"] == "completed"
    slot, claims = CanonicalMemoryRepository(source, relations).load(identity)
    assert slot is not None and slot.revision == 2
    assert {claim.canonical_value: claim.current.state for claim in claims} == {
        "sqlite": "SUPERSEDED",
        "postgresql": "ACTIVE",
    }
    assert validate_canonical_receipt_history(artifact_root, tenant_id="t1")["transaction_receipts"] == 2


def test_migration_orders_multi_object_receipts_by_revision_dependency_dag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, _index, _queue, relations, _committer, _episode, _scope = _setup(tmp_path)
    migration = MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    )

    def receipt(transaction_id: str, effects: list[tuple[str, int, int]]) -> dict:
        return {
            "transaction_id": transaction_id,
            "created_at": "2026-07-13T00:00:00Z",
            "effect_snapshots": [
                {
                    "uri": uri,
                    "before_revision": before,
                    "after_revision": after,
                }
                for uri, before, after in effects
            ],
        }

    payloads: dict[Path, dict] = {}
    y_predecessors: list[Path] = []
    for revision in range(1, 6):
        path = tmp_path / f"y-{revision}.json"
        payloads[path] = receipt(f"y-{revision}", [("memoryos://y", revision - 1, revision)])
        y_predecessors.append(path)
    bridge = tmp_path / "bridge-max-six.json"
    dependent = tmp_path / "dependent-max-two.json"
    payloads[bridge] = receipt(
        "bridge",
        [
            ("memoryos://x", 0, 1),
            ("memoryos://y", 5, 6),
        ],
    )
    payloads[dependent] = receipt("dependent", [("memoryos://x", 1, 2)])
    monkeypatch.setattr(migration_module, "load_transaction_receipt", payloads.__getitem__)

    ordered = migration._ordered_receipt_paths([dependent, bridge, *reversed(y_predecessors)])

    assert ordered.index(bridge) > ordered.index(y_predecessors[-1])
    assert ordered.index(dependent) > ordered.index(bridge)


def test_migration_rejects_cycle_in_multi_object_revision_dependency_dag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, _index, _queue, relations, _committer, _episode, _scope = _setup(tmp_path)
    migration = MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payloads = {
        first: {
            "transaction_id": "first",
            "created_at": "2026-07-13T00:00:00Z",
            "effect_snapshots": [
                {"uri": "memoryos://x", "before_revision": 0, "after_revision": 1},
                {"uri": "memoryos://y", "before_revision": 1, "after_revision": 2},
            ],
        },
        second: {
            "transaction_id": "second",
            "created_at": "2026-07-13T00:00:00Z",
            "effect_snapshots": [
                {"uri": "memoryos://y", "before_revision": 0, "after_revision": 1},
                {"uri": "memoryos://x", "before_revision": 1, "after_revision": 2},
            ],
        },
    }
    monkeypatch.setattr(migration_module, "load_transaction_receipt", payloads.__getitem__)

    with pytest.raises(MemoryClosureMigrationError, match="dependency graph contains a cycle"):
        migration._ordered_receipt_paths([second, first])


def test_prepared_intent_migration_backfills_once_and_is_interrupt_safe(
    tmp_path: Path,
) -> None:
    source, _index, _queue, relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(
        episode,
        "prepared-intent-migration",
        "SQLite",
        "confirmation",
        "confirmed",
    )
    _identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)
    transaction_id = str(operations[0].payload["transaction_id"])
    artifact_root = _artifact_root(tmp_path)
    receipt_path = committer._transaction_marker(str(operations[0].payload["idempotency_key"]))
    legacy_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    legacy_receipt.pop("prepared_intent_schema_version")
    receipt_core = {key: value for key, value in legacy_receipt.items() if key != "receipt_digest"}
    legacy_receipt["receipt_digest"] = canonical_digest(receipt_core)
    atomic_write_json(receipt_path, legacy_receipt, artifact_root=artifact_root)
    outbox_path = committer._outbox_path(transaction_id)
    legacy_outbox = json.loads(outbox_path.read_text(encoding="utf-8"))
    legacy_outbox["receipt_digest"] = legacy_receipt["receipt_digest"]
    outbox_core = {key: value for key, value in legacy_outbox.items() if key != "outbox_digest"}
    legacy_outbox["outbox_digest"] = canonical_digest(outbox_core)
    atomic_write_json(outbox_path, legacy_outbox, artifact_root=artifact_root)
    for head_path in (artifact_root / "system" / "current-heads").glob("*.json"):
        head_path.unlink()
    publish_current_head_sets(artifact_root, receipt_path, legacy_receipt)
    intent_path = committer.planning_proofs.canonical_intent_path(transaction_id)
    intent_path.unlink()

    migration = MemoryClosureMigration(
        tmp_path,
        tenant_id="t1",
        source_store=source,
        relation_store=relations,
    )
    migration.run()
    first_intent = intent_path.read_bytes()
    migration_receipt = (
        artifact_root / "system" / "migrations" / "canonical-prepared-intents" / f"{transaction_id}.json"
    )
    first_receipt = migration_receipt.read_bytes()

    migration.run()

    assert intent_path.read_bytes() == first_intent
    assert migration_receipt.read_bytes() == first_receipt
    validate_canonical_receipt_history(
        artifact_root,
        tenant_id="t1",
    )
