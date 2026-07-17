from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.visibility import relation_is_committed
from memoryos.operations.commit.effect_marker import validate_marker
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.outbox_envelope import (
    build_outbox,
    planned_effect_manifest,
    validate_outbox,
)
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.runtime.readiness import RuntimeReadiness, RuntimeReadinessState
from memoryos.workers.recovery_worker import RecoveryWorker
from tests.support.canonical_transactions import (
    _artifact_root,
    _plan,
    _proposal,
    _setup,
)


def _arm_prepared_transaction(tmp_path: Path) -> tuple[Any, Any, OperationCommitter, list[ContextOperation], Path]:
    source, _index, _queue, relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "strict-recovery", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    before_images = committer._capture_canonical_state(operations)
    before_by_uri = {str(item["uri"]): item["object"] for item in before_images}
    relation_manifests = {
        operation.operation_id: committer._build_canonical_relation_manifest(
            operation,
            before_by_uri.get(str(operation.target_uri or "")),
        )
        for operation in operations
    }
    transaction_id = str(operations[0].payload["transaction_id"])
    outbox = committer._write_outbox_event(
        transaction_id,
        str(operations[0].payload["idempotency_key"]),
        operations,
        status="prepared",
        before_images=before_images,
        relation_manifests=relation_manifests,
    )
    for operation in operations:
        committer.redo.begin(
            operation,
            phase="started",
            relation_manifest=relation_manifests[operation.operation_id],
        )
    return source, relations, committer, operations, outbox


def _rewrite_with_valid_outer_digest(path: Path, mutate: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    core = {key: value for key, value in payload.items() if key != "outbox_digest"}
    payload["outbox_digest"] = canonical_digest(core)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_aborted_outbox_is_terminal_and_clears_residual_redo_without_effects(tmp_path: Path) -> None:
    source, relations, committer, operations, outbox = _arm_prepared_transaction(tmp_path)
    committer._write_outbox_event(
        str(operations[0].payload["transaction_id"]),
        str(operations[0].payload["idempotency_key"]),
        operations,
        status="aborted",
    )

    worker = RecoveryWorker(RecoveryService(committer.redo, committer))
    first = worker.process_all()
    assert first["recovered_count"] == 0
    assert committer.redo.pending_entries() == []
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "aborted"
    assert relations.relations == []
    for operation in operations:
        with pytest.raises(FileNotFoundError):
            source.read_object(str(operation.target_uri))
    marker = committer._transaction_marker(str(operations[0].payload["idempotency_key"]))
    assert not marker.exists()

    snapshot = sorted(path.relative_to(tmp_path) for path in tmp_path.glob("**/*") if path.is_file())
    second = worker.process_all()
    assert second["recovered_count"] == second["failed_count"] == second["quarantine_count"] == 0
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.glob("**/*") if path.is_file()) == snapshot


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("transaction_id", "forged-transaction"),
        lambda payload: payload.__setitem__("idempotency_key", "forged-idempotency"),
        lambda payload: payload.__setitem__("operation_ids", payload["operation_ids"][:-1]),
        lambda payload: payload.__setitem__("tenant_id", "tenant-b"),
        lambda payload: payload.__setitem__("user_id", "user-b"),
        lambda payload: payload.__setitem__("event_type", "ForgedEvent"),
        lambda payload: payload.__setitem__("schema_version", "future-schema"),
    ],
)
def test_outbox_identity_or_membership_mismatch_is_quarantined_once(
    tmp_path: Path,
    mutation: Any,
) -> None:
    source, _relations, committer, operations, outbox = _arm_prepared_transaction(tmp_path)
    _rewrite_with_valid_outer_digest(outbox, mutation)
    worker = RecoveryWorker(RecoveryService(committer.redo, committer))

    first = worker.process_all()
    assert first["recovered_count"] == 0
    assert first["failed_count"] >= 1
    assert first["quarantine_count"] >= 1
    assert not outbox.exists()
    assert committer.redo.pending_entries() == []
    for operation in operations:
        with pytest.raises(FileNotFoundError):
            source.read_object(str(operation.target_uri))
    records = sorted((_artifact_root(tmp_path) / "system" / "quarantine").glob("**/*.json"))
    assert records
    second = worker.process_all()
    assert second["failed_count"] == second["quarantine_count"] == 0
    assert sorted((_artifact_root(tmp_path) / "system" / "quarantine").glob("**/*.json")) == records


def test_broken_outbox_json_is_quarantined_and_preserved(tmp_path: Path) -> None:
    _source, _relations, committer, _operations, outbox = _arm_prepared_transaction(tmp_path)
    outbox.write_text("{broken", encoding="utf-8")
    worker = RecoveryWorker(RecoveryService(committer.redo, committer))

    result = worker.process_all()
    assert result["quarantine_count"] >= 1
    quarantined = list((_artifact_root(tmp_path) / "system" / "quarantine" / "outbox").glob("*.original"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{broken"
    assert worker.process_all()["quarantine_count"] == 0


def test_missing_derived_relation_does_not_invalidate_immutable_receipt(tmp_path: Path) -> None:
    source, _index, _queue, relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "relation-tamper", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)
    relation = relations.relations[0]
    relations.delete_relation(relation.source_uri, relation.relation_type, relation.target_uri)
    idempotency_key = str(operations[0].payload["idempotency_key"])
    marker = committer._transaction_marker(idempotency_key)
    outbox = committer._outbox_path(str(operations[0].payload["transaction_id"]))

    result = RecoveryWorker(RecoveryService(committer.redo, committer)).process_all()
    assert result["recovered_count"] == 0
    assert result["quarantine_count"] == 0
    assert marker.exists()
    assert outbox.exists()
    validate_marker(marker, source, tenant_id="t1", user_id="u1")
    assert relation_is_committed(source, relation, relations) is True
    assert source.read_object(str(operations[0].target_uri))


def test_committed_outbox_mismatch_never_quarantines_valid_immutable_receipt(
    tmp_path: Path,
) -> None:
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "committed-outbox-mismatch", "SQLite", "confirmation", "confirmed")
    _identity, _, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)

    transaction_id = str(operations[0].payload["transaction_id"])
    idempotency_key = str(operations[0].payload["idempotency_key"])
    receipt = committer._transaction_marker(idempotency_key)
    outbox = committer._outbox_path(transaction_id)
    receipt_bytes = receipt.read_bytes()
    committed_outbox = json.loads(outbox.read_text(encoding="utf-8"))

    forged_operations = [ContextOperation.from_dict(item) for item in committed_outbox["operations"]]
    forged_operations[0].payload["content"] = (
        str(forged_operations[0].payload.get("content", "")) + "-different-self-consistent-outbox"
    )
    relation_manifests = {
        str(item["operation_id"]): dict(item.get("relation_manifest", {}) or {})
        for item in committed_outbox["effect_manifests"]
    }
    forged_effects = [
        planned_effect_manifest(operation, relation_manifests[operation.operation_id])
        for operation in forged_operations
    ]
    forged_claim_revisions = []
    for operation in forged_operations:
        obj = dict(operation.payload.get("context_object", {}) or {})
        metadata = dict(obj.get("metadata", {}) or {})
        if metadata.get("canonical_kind") == "claim":
            forged_claim_revisions.append(
                {
                    "uri": str(obj["uri"]),
                    "claim_id": str(metadata["claim_id"]),
                    "revision": int(metadata["revision"]),
                }
            )
    forged_outbox = build_outbox(
        transaction_id=transaction_id,
        idempotency_key=idempotency_key,
        tenant_id="t1",
        user_id="u1",
        operations=forged_operations,
        status="committed",
        before_images=list(committed_outbox["before_images"]),
        effect_manifests=forged_effects,
        claim_revisions=forged_claim_revisions,
        commit_group_id=str(committed_outbox["commit_group_id"]),
        receipt_path=str(committed_outbox["receipt_path"]),
        receipt_digest=str(committed_outbox["receipt_digest"]),
    )
    validate_outbox(forged_outbox)
    outbox.write_text(json.dumps(forged_outbox, ensure_ascii=False), encoding="utf-8")

    readiness = RuntimeReadiness()
    readiness.transition(RuntimeReadinessState.READY)
    vars(source)["readiness"] = readiness
    result = RecoveryWorker(RecoveryService(committer.redo, committer)).process_all()

    assert result["failed_count"] >= 1
    assert result["quarantine_count"] >= 1
    assert readiness.state == RuntimeReadinessState.NOT_READY
    assert receipt.exists()
    assert receipt.read_bytes() == receipt_bytes
    validate_marker(receipt, source, tenant_id="t1", user_id="u1")
    assert not outbox.exists()
    assert len(list((_artifact_root(tmp_path) / "system" / "quarantine" / "outbox").glob("*.original"))) == 1
    assert not list((_artifact_root(tmp_path) / "system" / "quarantine" / "marker").glob("*.original"))

    restarted = MemoryOSClient(str(tmp_path), tenant_id="t1")
    assert restarted.readiness.state.value == "NOT_READY"
    assert receipt.exists()
    assert receipt.read_bytes() == receipt_bytes
