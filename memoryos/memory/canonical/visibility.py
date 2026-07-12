"""记忆系统里的可见性。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.source_store import RelationStore, SourceStore
from memoryos.core.ids import require_safe_path_segment
from memoryos.operations.commit.effect_marker import (
    EffectProofError,
    marker_proves_object,
    marker_proves_relation,
    validate_marker,
)
from memoryos.operations.commit.outbox_envelope import OutboxIntegrityError, validate_outbox
from memoryos.operations.commit.quarantine import quarantine_control_file


@dataclass(frozen=True)
class CommittedCanonicalRead:
    """负责 CommittedCanonicalRead 这部分逻辑。"""

    object: ContextObject
    content_override: str | None = None
    from_before_image: bool = False


def _tenant_artifact_root(source_store: SourceStore) -> Path | None:
    root = getattr(source_store, "root", None)
    tenant_id = getattr(source_store, "tenant_id", "default")
    if root is None or not isinstance(tenant_id, str) or not tenant_id.strip():
        return None
    if tenant_id in {".", ".."} or "/" in tenant_id or "\\" in tenant_id:
        return None
    root_path = Path(root)
    return root_path if tenant_id == "default" else root_path / "tenants" / tenant_id


def read_committed_canonical(
    source_store: SourceStore,
    uri: str,
    relation_store: RelationStore | None = None,
) -> CommittedCanonicalRead:
    """只读取已经完成事务提交的规范记忆。"""

    obj = source_store.read_object(uri)
    metadata = dict(obj.metadata or {})
    if metadata.get("canonical_kind") not in {"slot", "claim"}:
        return CommittedCanonicalRead(obj)
    try:
        idempotency_key = require_safe_path_segment(
            metadata.get("canonical_idempotency_key"),
            "canonical idempotency key",
        )
        transaction_id = require_safe_path_segment(
            metadata.get("canonical_transaction_id"),
            "canonical transaction id",
        )
    except ValueError:
        raise FileNotFoundError(f"canonical object has no committed transaction proof: {uri}") from None
    artifact_root = _tenant_artifact_root(source_store)
    if artifact_root is None:
        raise FileNotFoundError(f"canonical object has no committed transaction proof: {uri}")
    marker = artifact_root / "system" / "transactions" / f"{idempotency_key}.json"
    if marker.exists():
        try:
            proof = validate_marker(
                marker,
                source_store,
                relation_store,
                transaction_id=transaction_id,
                idempotency_key=idempotency_key,
                tenant_id=str(obj.tenant_id or "default"),
                user_id=str(obj.owner_user_id or ""),
                object_uris={uri},
            )
        except EffectProofError as exc:
            if marker.exists():
                quarantine_control_file(
                    artifact_root,
                    marker,
                    kind="transaction_marker",
                    error=exc,
                    identifiers={
                        "transaction_id": transaction_id,
                        "idempotency_key": idempotency_key,
                    },
                )
            raise FileNotFoundError(f"canonical object is not committed: {uri}") from None
        if not marker_proves_object(proof, uri):
            raise FileNotFoundError(f"canonical object is not committed: {uri}")
        return CommittedCanonicalRead(obj)
    outbox = artifact_root / "system" / "outbox" / f"{transaction_id}.json"
    try:
        event = validate_outbox(
            json.loads(outbox.read_text(encoding="utf-8")),
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            tenant_id=str(obj.tenant_id or "default"),
            user_id=str(obj.owner_user_id or ""),
            allowed_statuses={"prepared", "source_committed"},
        )
    except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
        if outbox.exists():
            quarantine_control_file(
                artifact_root,
                outbox,
                kind="outbox",
                error=exc,
                identifiers={"transaction_id": transaction_id},
            )
        raise FileNotFoundError(f"canonical object is not committed: {uri}") from None
    before = next(
        (item for item in event.get("before_images", []) or [] if str(item.get("uri", "")) == uri),
        None,
    )
    if not before or not before.get("exists") or not isinstance(before.get("object"), dict):
        raise FileNotFoundError(f"canonical object is not committed: {uri}")
    return CommittedCanonicalRead(
        ContextObject.from_dict(before["object"]),
        content_override=str(before.get("content", "")),
        from_before_image=True,
    )


def relation_is_committed(
    source_store: SourceStore,
    relation: ContextRelation,
    relation_store: RelationStore | None = None,
) -> bool:
    """检查关系两端是否都已提交并且当前可见。"""

    metadata = dict(relation.metadata or {})
    try:
        idempotency_key = require_safe_path_segment(
            metadata.get("canonical_idempotency_key"),
            "canonical relation idempotency key",
        )
        transaction_id = require_safe_path_segment(
            metadata.get("canonical_transaction_id"),
            "canonical relation transaction id",
        )
    except ValueError:
        return False
    artifact_root = _tenant_artifact_root(source_store)
    if artifact_root is None:
        return False
    marker = artifact_root / "system" / "transactions" / f"{idempotency_key}.json"
    if not marker.exists():
        return False
    try:
        proof = validate_marker(
            marker,
            source_store,
            relation_store,
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            tenant_id=str(metadata.get("tenant_id") or getattr(source_store, "tenant_id", "default")),
            user_id=str(metadata.get("owner_user_id") or ""),
        )
    except EffectProofError as exc:
        if marker.exists():
            quarantine_control_file(
                artifact_root,
                marker,
                kind="transaction_marker",
                error=exc,
                identifiers={
                    "transaction_id": transaction_id,
                    "idempotency_key": idempotency_key,
                },
            )
        return False
    return marker_proves_relation(proof, relation)
