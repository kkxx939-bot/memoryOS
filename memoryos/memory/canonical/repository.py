"""规范记忆的仓库读取逻辑。"""

from __future__ import annotations

from memoryos.contextdb.store.source_store import SourceStore
from memoryos.memory.canonical.identity import ResolvedMemoryIdentity
from memoryos.memory.canonical.state import (
    MemoryClaim,
    MemoryRevision,
    MemorySlot,
    TransitionProfile,
)
from memoryos.memory.canonical.visibility import read_committed_canonical


class CanonicalMemoryRepository:
    """负责 CanonicalMemoryRepository 的持久化读写。"""

    def __init__(self, source_store: SourceStore) -> None:
        self.source_store = source_store

    def load(self, identity: ResolvedMemoryIdentity) -> tuple[MemorySlot | None, tuple[MemoryClaim, ...]]:
        try:
            obj = read_committed_canonical(self.source_store, identity.slot_uri).object
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None, ()
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "slot":
            return None, ()
        slot = MemorySlot(
            slot_id=str(metadata["slot_id"]),
            uri=obj.uri,
            memory_type=str(metadata["memory_type"]),
            identity_fields=dict(metadata.get("identity_fields", {}) or {}),
            scope_keys=tuple(str(item) for item in metadata.get("scope_keys", []) or []),
            claim_ids=tuple(str(item) for item in metadata.get("claim_ids", []) or []),
            active_claim_id=str(metadata["active_claim_id"]) if metadata.get("active_claim_id") else None,
            revision=int(metadata.get("revision", 0)),
        )
        claims = []
        for claim_id in slot.claim_ids:
            uri = f"{slot.uri}/claims/{claim_id}"
            try:
                claim_obj = read_committed_canonical(self.source_store, uri).object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            claim_metadata = dict(claim_obj.metadata or {})
            claims.append(
                MemoryClaim(
                    claim_id=claim_id,
                    uri=uri,
                    slot_id=slot.slot_id,
                    canonical_value=str(claim_metadata.get("canonical_value", "")),
                    profile=TransitionProfile(str(claim_metadata["transition_profile"])),
                    revisions=tuple(
                        MemoryRevision.from_dict(item) for item in claim_metadata.get("revisions", []) or []
                    ),
                )
            )
        return slot, tuple(claims)
