"""记忆系统里的可见性。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.ids import require_safe_path_segment


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


def read_committed_canonical(source_store: SourceStore, uri: str) -> CommittedCanonicalRead:
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
    if (artifact_root / "system" / "transactions" / f"{idempotency_key}.json").exists():
        return CommittedCanonicalRead(obj)
    outbox = artifact_root / "system" / "outbox" / f"{transaction_id}.json"
    try:
        event = json.loads(outbox.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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


def relation_is_committed(source_store: SourceStore, relation: ContextRelation) -> bool:
    """检查关系两端是否都已提交并且当前可见。"""

    metadata = dict(relation.metadata or {})
    try:
        idempotency_key = require_safe_path_segment(
            metadata.get("canonical_idempotency_key"),
            "canonical relation idempotency key",
        )
    except ValueError:
        return False
    artifact_root = _tenant_artifact_root(source_store)
    if artifact_root is None:
        return False
    return (artifact_root / "system" / "transactions" / f"{idempotency_key}.json").exists()
