"""Startup validation for immutable canonical receipt history."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from memoryos.contextdb.session.planning_envelope import (
    PlanningEnvelopeIntegrityError,
    PlanningEnvelopeStore,
)
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.path_safety import DurablePathIntegrityError, require_safe_artifact_path
from memoryos.memory.canonical.current_head import iter_current_head_uris, load_current_head
from memoryos.memory.canonical.event import canonical_digest
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    prepared_intent_digest,
    validate_outbox,
)
from memoryos.operations.commit.planning_proof import (
    CANONICAL_PREPARED_INTENT_SCHEMA_VERSION,
    ImmutablePlanningProofStore,
    PlanningProofIntegrityError,
)
from memoryos.operations.commit.receipt import (
    TRANSACTION_RECEIPT_SCHEMA_VERSION,
    ReceiptIntegrityError,
    load_transaction_receipt,
)


class CanonicalHistoryIntegrityError(RuntimeError):
    """Receipt history is corrupt, forked, incomplete, or detached from its head."""


def validate_canonical_receipt_history(
    artifact_root: Path,
    *,
    tenant_id: str,
) -> dict[str, Any]:
    """Validate every immutable memory receipt without consulting current Source."""

    receipts: list[tuple[Path, dict[str, Any]]] = []
    for directory_name in ("transactions", "operations"):
        directory = artifact_root / "system" / directory_name
        try:
            require_safe_artifact_path(
                artifact_root,
                directory,
                label="canonical receipt history directory",
            )
        except DurablePathIntegrityError as exc:
            raise CanonicalHistoryIntegrityError("canonical receipt history directory is invalid") from exc
        for path in sorted(directory.glob("*.json")) if directory.exists() else ():
            try:
                require_safe_artifact_path(
                    artifact_root,
                    path,
                    label="canonical receipt history artifact",
                )
            except DurablePathIntegrityError as exc:
                raise CanonicalHistoryIntegrityError(
                    f"receipt artifact path cannot be a symbolic link: {path.name}"
                ) from exc
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CanonicalHistoryIntegrityError(f"receipt artifact is unreadable: {path.name}") from exc
            if not isinstance(raw, dict) or raw.get("schema_version") != TRANSACTION_RECEIPT_SCHEMA_VERSION:
                continue
            try:
                receipt = load_transaction_receipt(path)
            except ReceiptIntegrityError as exc:
                raise CanonicalHistoryIntegrityError(f"historical receipt is corrupt: {path.name}") from exc
            if receipt.get("tenant_id") != tenant_id:
                raise CanonicalHistoryIntegrityError(f"historical receipt crosses tenant boundary: {path.name}")
            _validate_receipt_path_identity(path, receipt)
            receipts.append((path, receipt))

    by_transaction: dict[str, str] = {}
    by_idempotency: dict[str, str] = {}
    revisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, receipt in receipts:
        digest = str(receipt["receipt_digest"])
        transaction_id = str(receipt["transaction_id"])
        idempotency_key = str(receipt["idempotency_key"])
        if by_transaction.setdefault(transaction_id, digest) != digest:
            raise CanonicalHistoryIntegrityError(f"transaction id has conflicting immutable receipts: {transaction_id}")
        if by_idempotency.setdefault(idempotency_key, digest) != digest:
            raise CanonicalHistoryIntegrityError(
                f"idempotency key has conflicting immutable receipts: {idempotency_key}"
            )
        for snapshot in receipt.get("effect_snapshots", []) or []:
            if not isinstance(snapshot, dict):
                continue
            revisions[str(snapshot["uri"])].append(
                {
                    "before": int(snapshot.get("before_revision", 0) or 0),
                    "after": int(snapshot.get("after_revision", 0) or 0),
                    "receipt_digest": digest,
                    "created_at": str(receipt.get("created_at") or ""),
                    "path": str(path),
                    "snapshot": dict(snapshot),
                }
            )
        if path.parent.name == "transactions" and not receipt.get("migration_source_marker_digest"):
            _validate_receipt_outbox(artifact_root, receipt)
        if path.parent.name == "operations" and not receipt.get("migration_source_marker_digest"):
            _validate_pending_prepared_intent(artifact_root, receipt, tenant_id=tenant_id)
        _validate_receipt_diff(artifact_root, receipt)
        _validate_receipt_planning(artifact_root, receipt, tenant_id=tenant_id)

    for uri, rows in revisions.items():
        rows.sort(key=lambda item: (item["after"], item["before"], item["created_at"], item["path"]))
        seen_after: dict[int, str] = {}
        previous_after: int | None = None
        for index, row in enumerate(rows):
            existing = seen_after.setdefault(row["after"], row["receipt_digest"])
            if existing != row["receipt_digest"]:
                raise CanonicalHistoryIntegrityError(
                    f"canonical history has a same-revision fork: {uri}#{row['after']}"
                )
            if index == 0:
                if row["before"] != 0 or row["after"] not in {0, 1}:
                    raise CanonicalHistoryIntegrityError(
                        f"canonical history does not start at revision zero/one: {uri}"
                    )
            elif row["before"] != previous_after or row["after"] != row["before"] + 1:
                raise CanonicalHistoryIntegrityError(f"canonical history is non-contiguous: {uri}")
            previous_after = row["after"]

    latest_snapshots = {uri: dict(rows[-1]["snapshot"]) for uri, rows in revisions.items() if rows}
    required_heads: set[str] = set()
    slot_members: dict[str, set[str]] = {}
    for uri, snapshot in latest_snapshots.items():
        object_payload = dict(snapshot.get("object", {}) or {})
        metadata = dict(object_payload.get("metadata", {}) or {})
        kind = str(snapshot.get("canonical_kind") or metadata.get("canonical_kind") or "")
        if kind == "slot":
            members = {
                uri,
                *(f"{uri}/claims/{claim_id}" for claim_id in metadata.get("claim_ids", []) or []),
            }
            slot_members[uri] = members
            required_heads.update(members)
        elif kind == "pending_proposal":
            required_heads.add(uri)

    for uri in sorted(required_heads):
        try:
            load_current_head(artifact_root, uri)
        except FileNotFoundError as exc:
            raise CanonicalHistoryIntegrityError(f"required current head is missing: {uri}") from exc

    head_count = 0
    for uri in iter_current_head_uris(
        artifact_root,
        kinds=("slot", "claim", "pending_proposal"),
    ):
        head, _receipt, _snapshot = load_current_head(artifact_root, uri)
        rows = revisions.get(uri, [])
        if not rows:
            raise CanonicalHistoryIntegrityError(f"current head has no receipt history: {uri}")
        latest = rows[-1]
        if latest["after"] != int(head["current_revision"]) or latest["receipt_digest"] != head["receipt_digest"]:
            raise CanonicalHistoryIntegrityError(f"current head is not the unique latest receipt: {uri}")
        if str(head.get("canonical_kind") or "") in {"slot", "claim", "pending_proposal"} and uri not in required_heads:
            raise CanonicalHistoryIntegrityError(f"current head is not part of the latest canonical state: {uri}")
        head_count += 1
    current_slot_members = set(iter_current_head_uris(artifact_root, kinds=("slot", "claim")))
    for slot_uri, expected_members in slot_members.items():
        actual_members = {
            uri for uri in current_slot_members if uri == slot_uri or uri.startswith(f"{slot_uri}/claims/")
        }
        if actual_members != expected_members:
            raise CanonicalHistoryIntegrityError(f"current head-set does not match latest Slot membership: {slot_uri}")
    transaction_receipts = sum(path.parent.name == "transactions" for path, _receipt in receipts)
    operation_receipts = sum(path.parent.name == "operations" for path, _receipt in receipts)
    return {
        "receipts": len(receipts),
        "transaction_receipts": transaction_receipts,
        "operation_receipts": operation_receipts,
        "uris": len(revisions),
        "heads": head_count,
    }


def _validate_receipt_path_identity(path: Path, receipt: dict[str, Any]) -> None:
    directory_name = path.parent.name
    try:
        operations = receipt.get("operations")
        pending_only = bool(
            isinstance(operations, list)
            and len(operations) == 1
            and isinstance(operations[0], dict)
            and isinstance(operations[0].get("payload"), dict)
            and operations[0]["payload"].get("canonical_pending_proposal") is True
            and operations[0]["payload"].get("canonical_memory") is not True
        )
        expected_directory = "operations" if pending_only else "transactions"
        if directory_name != expected_directory:
            raise ValueError("receipt is stored in the wrong immutable namespace")
        if not pending_only:
            identity = require_safe_path_segment(
                receipt.get("idempotency_key"),
                "canonical receipt idempotency_key",
            )
        else:
            operation_ids = receipt.get("operation_ids")
            if not isinstance(operation_ids, list) or len(operation_ids) != 1:
                raise ValueError("pending receipt requires exactly one operation identity")
            identity = require_safe_path_segment(
                operation_ids[0],
                "pending receipt operation_id",
            )
    except (TypeError, ValueError) as exc:
        raise CanonicalHistoryIntegrityError("receipt path identity is invalid") from exc
    if path.is_symlink() or path.name != f"{identity}.json":
        raise CanonicalHistoryIntegrityError(f"receipt path identity does not match immutable receipt: {path.name}")


def _validate_receipt_outbox(artifact_root: Path, receipt: dict[str, Any]) -> None:
    prepared_schema = receipt.get("prepared_intent_schema_version")
    if prepared_schema not in {None, CANONICAL_PREPARED_INTENT_SCHEMA_VERSION}:
        raise CanonicalHistoryIntegrityError(
            f"canonical receipt declares an unsupported prepared-intent schema: {receipt['transaction_id']}"
        )
    try:
        ImmutablePlanningProofStore(
            artifact_root,
            tenant_id=str(receipt["tenant_id"]),
        ).load_canonical_intent(
            str(receipt["transaction_id"]),
            operations=[item for item in receipt.get("operations", []) if isinstance(item, dict)],
            prepared_intent_digest=str(receipt.get("prepared_intent_digest") or ""),
        )
    except PlanningProofIntegrityError as exc:
        raise CanonicalHistoryIntegrityError(
            f"canonical receipt has no durable matching prepared intent: {receipt['transaction_id']}"
        ) from exc
    path = artifact_root / "system" / "outbox" / f"{receipt['transaction_id']}.json"
    try:
        require_safe_artifact_path(
            artifact_root,
            path,
            label="canonical committed outbox",
        )
    except DurablePathIntegrityError as exc:
        raise CanonicalHistoryIntegrityError(
            f"canonical receipt has an invalid committed outbox: {receipt['transaction_id']}"
        ) from exc
    if not path.exists():
        raise CanonicalHistoryIntegrityError(f"canonical receipt has no committed outbox: {receipt['transaction_id']}")
    try:
        outbox = validate_outbox(
            json.loads(path.read_text(encoding="utf-8")),
            transaction_id=str(receipt["transaction_id"]),
            idempotency_key=str(receipt["idempotency_key"]),
            tenant_id=str(receipt["tenant_id"]),
            user_id=str(receipt["user_id"]),
            allowed_statuses={"committed"},
        )
    except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
        raise CanonicalHistoryIntegrityError(
            f"canonical receipt outbox is invalid: {receipt['transaction_id']}"
        ) from exc
    if prepared_intent_digest(outbox) != receipt.get("prepared_intent_digest") or outbox.get(
        "receipt_digest"
    ) != receipt.get("receipt_digest"):
        raise CanonicalHistoryIntegrityError(
            f"canonical receipt and committed outbox disagree: {receipt['transaction_id']}"
        )


def _validate_receipt_diff(artifact_root: Path, receipt: dict[str, Any]) -> None:
    try:
        diff_id = require_safe_path_segment(str(receipt["diff"]["diff_id"]), "diff_id")
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalHistoryIntegrityError("canonical receipt has an invalid diff identity") from exc
    path = artifact_root / "system" / "diffs" / f"{diff_id}.json"
    try:
        require_safe_artifact_path(
            artifact_root,
            path,
            label="canonical diff",
        )
    except DurablePathIntegrityError as exc:
        raise CanonicalHistoryIntegrityError(f"canonical receipt diff has an invalid artifact path: {diff_id}") from exc
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalHistoryIntegrityError(f"canonical receipt diff is missing or unreadable: {diff_id}") from exc
    if (
        not isinstance(payload, dict)
        or canonical_digest(payload) != receipt.get("diff_digest")
        or canonical_digest(payload) != canonical_digest(receipt.get("diff"))
    ):
        raise CanonicalHistoryIntegrityError(f"canonical receipt diff artifact is corrupt: {diff_id}")


def _validate_receipt_planning(
    artifact_root: Path,
    receipt: dict[str, Any],
    *,
    tenant_id: str,
) -> None:
    operations = [item for item in receipt.get("operations", []) if isinstance(item, dict)]
    task_ids = {str(dict(item.get("payload", {}) or {}).get("planning_task_id") or "") for item in operations} - {""}
    if len(task_ids) > 1:
        raise CanonicalHistoryIntegrityError("canonical receipt crosses planning task identities")
    task_id = next(iter(task_ids), "")
    if task_id:
        store = _planning_envelope_store_for_artifact_root(
            artifact_root,
            tenant_id=tenant_id,
        )
        path = store.path(task_id)
        anchor = store.anchor_path(task_id)
        try:
            require_safe_artifact_path(
                artifact_root,
                path,
                label="canonical planning envelope",
            )
            require_safe_artifact_path(
                artifact_root,
                anchor,
                label="canonical planning anchor",
            )
        except DurablePathIntegrityError as exc:
            raise CanonicalHistoryIntegrityError(f"canonical receipt planning path is invalid: {task_id}") from exc
        if path.exists() or path.is_symlink() or anchor.exists() or anchor.is_symlink():
            try:
                envelope = store.load_validated_payload(task_id)
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                PlanningEnvelopeIntegrityError,
            ) as exc:
                raise CanonicalHistoryIntegrityError(
                    f"canonical receipt planning envelope is invalid: {task_id}"
                ) from exc
            receipt_fingerprints = {
                str(value)
                for operation in operations
                for value in dict(operation.get("payload", {}) or {}).get("proposal_fingerprints", []) or []
            }
            envelope_fingerprints = {str(value) for value in envelope.get("proposal_fingerprints", []) or []}
            if (
                envelope.get("planning_digest") != receipt.get("planning_digest")
                or str(envelope.get("operation_group_identity") or "") != str(receipt.get("commit_group_id") or "")
                or not receipt_fingerprints.issubset(envelope_fingerprints)
            ):
                raise CanonicalHistoryIntegrityError(f"canonical receipt is detached from planning envelope: {task_id}")
            return

    marker_digest = str(receipt.get("migration_source_marker_digest") or "")
    if marker_digest:
        expected = canonical_digest(
            {"schema_version": "migrated_planning_proof_v1", "legacy_marker_digest": marker_digest}
        )
        if expected != receipt.get("planning_digest"):
            raise CanonicalHistoryIntegrityError("migrated canonical receipt planning proof is invalid")
        return
    try:
        ImmutablePlanningProofStore(artifact_root, tenant_id=tenant_id).load_direct(
            str(receipt.get("transaction_id") or ""),
            operations=operations,
            planning_digest=str(receipt.get("planning_digest") or ""),
        )
    except PlanningProofIntegrityError as exc:
        identity = task_id or str(receipt.get("transaction_id") or "")
        raise CanonicalHistoryIntegrityError(
            f"canonical receipt has no durable matching planning proof: {identity}"
        ) from exc


def _planning_envelope_store_for_artifact_root(
    artifact_root: Path,
    *,
    tenant_id: str,
) -> PlanningEnvelopeStore:
    """Bind the envelope path and payload validator to the same tenant.

    Receipt-history validation receives the already selected tenant artifact
    root.  ``PlanningEnvelopeStore`` receives the shared root instead, because
    it derives the tenant directory itself.  Keeping those two identities
    separate previously made non-default envelopes validate as ``default``.
    """

    artifact_root = Path(artifact_root)
    if tenant_id == "default":
        shared_root = artifact_root
    else:
        if artifact_root.name != tenant_id or artifact_root.parent.name != "tenants":
            raise CanonicalHistoryIntegrityError(
                "canonical receipt history tenant root does not match its tenant identity"
            )
        shared_root = artifact_root.parent.parent
    store = PlanningEnvelopeStore(shared_root, tenant_id=tenant_id)
    if store.artifact_root != artifact_root:
        raise CanonicalHistoryIntegrityError("canonical receipt planning store escaped its tenant artifact root")
    return store


def _validate_pending_prepared_intent(
    artifact_root: Path,
    receipt: dict[str, Any],
    *,
    tenant_id: str,
) -> None:
    operations = [item for item in receipt.get("operations", []) if isinstance(item, dict)]
    if (
        len(operations) != 1
        or dict(operations[0].get("payload", {}) or {}).get("canonical_pending_proposal") is not True
    ):
        return
    operation_id = str(operations[0].get("operation_id") or "")
    try:
        ImmutablePlanningProofStore(artifact_root, tenant_id=tenant_id).load_pending_intent(
            operation_id,
            operation=operations[0],
            prepared_intent_digest=str(receipt.get("prepared_intent_digest") or ""),
        )
    except PlanningProofIntegrityError as exc:
        raise CanonicalHistoryIntegrityError(
            f"pending receipt has no durable matching prepared intent: {operation_id}"
        ) from exc
