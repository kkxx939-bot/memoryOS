"""Validated Identity V2 canonical memory repository reads."""

from __future__ import annotations

from memoryos.contextdb.store.source_store import SourceStore
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2, ResolvedMemoryIdentity
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    MemoryClaim,
    MemoryRevision,
    MemorySlot,
    MissingClaimInvariantError,
    TransitionProfile,
)
from memoryos.memory.canonical.visibility import read_committed_canonical


class CanonicalMemoryRepository:
    """Load committed Identity V2 state and validate every Slot/Claim invariant."""

    def __init__(self, source_store: SourceStore) -> None:
        self.source_store = source_store

    def load(self, identity: ResolvedMemoryIdentity) -> tuple[MemorySlot | None, tuple[MemoryClaim, ...]]:
        if identity.identity_algorithm_version != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError("only Identity V2 can be loaded")
        try:
            slot, claims = self.load_uri(identity.slot_uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None, ()
        if slot.slot_id != identity.slot_id:
            raise CanonicalMemoryInvariantError("canonical Slot URI does not match resolved Identity V2")
        if slot.canonical_subject_key != identity.canonical_subject_key:
            raise CanonicalMemoryInvariantError("canonical Slot subject does not match resolved Identity V2")
        return slot, claims

    def load_uri(self, slot_uri: str) -> tuple[MemorySlot, tuple[MemoryClaim, ...]]:
        obj = read_committed_canonical(self.source_store, slot_uri).object
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "slot":
            raise CanonicalMemoryInvariantError(f"canonical Slot URI contains {metadata.get('canonical_kind')!r}")
        if metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError("canonical Slot is not Identity V2")
        scope_payload = dict(metadata.get("scope", {}) or {})
        subject_payload = scope_payload.get("canonical_subject")
        if not isinstance(subject_payload, dict):
            raise CanonicalMemoryInvariantError("Identity V2 Slot is missing canonical subject payload")
        subject = ScopeRef.from_dict(subject_payload)
        subject_key = str(metadata.get("canonical_subject") or "")
        if not subject_key or subject.key != subject_key:
            raise CanonicalMemoryInvariantError("Identity V2 Slot canonical subject is inconsistent")
        slot = MemorySlot(
            slot_id=str(metadata["slot_id"]),
            uri=obj.uri,
            memory_type=str(metadata["memory_type"]),
            identity_fields=dict(metadata.get("identity_fields", {}) or {}),
            scope_keys=tuple(str(item) for item in metadata.get("scope_keys", []) or []),
            claim_ids=tuple(str(item) for item in metadata.get("claim_ids", []) or []),
            active_claim_id=str(metadata["active_claim_id"]) if metadata.get("active_claim_id") else None,
            revision=int(metadata.get("revision", 0)),
            identity_algorithm_version=IDENTITY_ALGORITHM_V2,
            canonical_subject_key=subject_key,
            canonical_subject=subject,
        )
        claims_list = []
        for claim_id in slot.claim_ids:
            try:
                claims_list.append(self._load_claim(slot, claim_id))
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                raise MissingClaimInvariantError(slot.slot_id, (claim_id,)) from None
        claims = tuple(claims_list)
        slot.validate_claims(claims)
        return slot, claims

    def _load_claim(self, slot: MemorySlot, claim_id: str) -> MemoryClaim:
        uri = f"{slot.uri}/claims/{claim_id}"
        obj = read_committed_canonical(self.source_store, uri).object
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "claim":
            raise CanonicalMemoryInvariantError(f"slot claim URI is not a canonical claim: {uri}")
        if metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError(f"canonical Claim is not Identity V2: {uri}")
        persisted_revision = int(metadata.get("revision", 0))
        revisions = tuple(MemoryRevision.from_dict(item) for item in metadata.get("revisions", []) or [])
        claim = MemoryClaim(
            claim_id=str(metadata.get("claim_id") or claim_id),
            uri=obj.uri,
            slot_id=str(metadata.get("slot_id") or slot.slot_id),
            canonical_value=str(metadata.get("canonical_value", "")),
            profile=TransitionProfile(str(metadata["transition_profile"])),
            revisions=revisions,
            identity_algorithm_version=IDENTITY_ALGORITHM_V2,
            canonical_subject_key=str(metadata.get("canonical_subject") or ""),
        )
        if claim.slot_id != slot.slot_id or claim.claim_id != claim_id:
            raise CanonicalMemoryInvariantError(f"canonical Claim identity does not match Slot path: {uri}")
        if claim.canonical_subject_key != slot.canonical_subject_key:
            raise CanonicalMemoryInvariantError(f"canonical Claim subject does not match its Slot: {uri}")
        if persisted_revision != claim.latest_revision.revision:
            raise CanonicalMemoryInvariantError(
                f"claim {claim.claim_id} metadata revision {persisted_revision} "
                f"does not match history {claim.latest_revision.revision}"
            )
        persisted_current = metadata.get("current_revision")
        if persisted_current is not None and int(persisted_current) != claim.current.revision:
            raise CanonicalMemoryInvariantError(f"claim {claim.claim_id} current revision pointer is inconsistent")
        return claim
