from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.source_store import SourceStore


@dataclass(frozen=True)
class CommittedCanonicalRead:
    object: ContextObject
    content_override: str | None = None
    from_before_image: bool = False


def read_committed_canonical(source_store: SourceStore, uri: str) -> CommittedCanonicalRead:
    obj = source_store.read_object(uri)
    metadata = dict(obj.metadata or {})
    idempotency_key = str(metadata.get("canonical_idempotency_key", ""))
    transaction_id = str(metadata.get("canonical_transaction_id", ""))
    root = getattr(source_store, "root", None)
    if not idempotency_key or not transaction_id or root is None:
        return CommittedCanonicalRead(obj)
    root_path = Path(root)
    if (root_path / "system" / "transactions" / f"{idempotency_key}.json").exists():
        return CommittedCanonicalRead(obj)
    outbox = root_path / "system" / "outbox" / f"{transaction_id}.json"
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
    metadata = dict(relation.metadata or {})
    idempotency_key = str(metadata.get("canonical_idempotency_key", ""))
    root = getattr(source_store, "root", None)
    if not idempotency_key or root is None:
        return True
    return (Path(root) / "system" / "transactions" / f"{idempotency_key}.json").exists()
