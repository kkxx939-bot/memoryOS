"""Validated Identity V2 canonical memory repository reads."""

from __future__ import annotations

from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import RelationStore, SourceStore
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2, ResolvedMemoryIdentity
from memoryos.memory.canonical.proposal import PendingMemoryProposal
from memoryos.memory.canonical.scope import MemoryScope
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

    def __init__(
        self,
        source_store: SourceStore,
        relation_store: RelationStore | None = None,
    ) -> None:
        self.source_store = source_store
        self.relation_store = relation_store

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
        obj = read_committed_canonical(self.source_store, slot_uri, self.relation_store).object
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "slot":
            raise CanonicalMemoryInvariantError(f"canonical Slot URI contains {metadata.get('canonical_kind')!r}")
        if metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError("canonical Slot is not Identity V2")
        raw_scope = metadata.get("scope")
        if not isinstance(raw_scope, dict):
            raise CanonicalMemoryInvariantError("Identity V2 Slot is missing canonical subject payload")
        try:
            memory_scope = MemoryScope.from_dict(raw_scope)
        except (KeyError, TypeError, ValueError) as exc:
            raise CanonicalMemoryInvariantError("Identity V2 Slot scope is invalid") from exc
        self._validate_scope_authority(obj, metadata, memory_scope)
        subject = memory_scope.canonical_subject
        if subject is None:
            raise CanonicalMemoryInvariantError("Identity V2 Slot is missing canonical subject payload")
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

    def load_pending(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> PendingMemoryProposal:
        obj = self.source_store.read_object(uri)
        record = PendingMemoryProposal.from_context_object(obj)
        if tenant_id is not None and str(obj.tenant_id or "default") != str(tenant_id):
            raise PermissionError("pending proposal tenant does not match the requested tenant")
        if owner_user_id is not None and str(obj.owner_user_id or "") != str(owner_user_id):
            raise PermissionError("pending proposal owner does not match the requested owner")
        return record

    def list_pending(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        lifecycle_states: tuple[str, ...] = (),
    ) -> tuple[PendingMemoryProposal, ...]:
        requested = {str(item).casefold() for item in lifecycle_states}
        records: list[PendingMemoryProposal] = []
        for obj in self.source_store.list_objects():
            metadata = dict(obj.metadata or {})
            if metadata.get("canonical_kind") != "pending_proposal":
                continue
            if str(obj.tenant_id or "default") != str(tenant_id) or str(obj.owner_user_id or "") != str(
                owner_user_id
            ):
                continue
            record = PendingMemoryProposal.from_context_object(obj)
            if requested and record.lifecycle_state.value.casefold() not in requested:
                continue
            records.append(record)
        return tuple(sorted(records, key=lambda item: (item.created_at, item.uri)))

    def _load_claim(self, slot: MemorySlot, claim_id: str) -> MemoryClaim:
        uri = f"{slot.uri}/claims/{claim_id}"
        obj = read_committed_canonical(self.source_store, uri, self.relation_store).object
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "claim":
            raise CanonicalMemoryInvariantError(f"slot claim URI is not a canonical claim: {uri}")
        if metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError(f"canonical Claim is not Identity V2: {uri}")
        raw_scope = metadata.get("scope")
        if not isinstance(raw_scope, dict):
            raise CanonicalMemoryInvariantError(f"canonical Claim scope is invalid: {uri}")
        try:
            memory_scope = MemoryScope.from_dict(raw_scope)
        except (KeyError, TypeError, ValueError) as exc:
            raise CanonicalMemoryInvariantError(f"canonical Claim scope is invalid: {uri}") from exc
        self._validate_scope_authority(obj, metadata, memory_scope)
        if memory_scope.canonical_subject is None:
            raise CanonicalMemoryInvariantError(f"canonical Claim subject is missing: {uri}")
        if memory_scope.canonical_subject.key != str(metadata.get("canonical_subject") or ""):
            raise CanonicalMemoryInvariantError(f"canonical Claim subject payload is inconsistent: {uri}")
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

    def _validate_scope_authority(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        memory_scope: MemoryScope,
    ) -> None:
        tenant_id = str(getattr(obj, "tenant_id", None) or "default")
        if memory_scope.visibility.tenant_id != tenant_id or memory_scope.authority.inferred:
            raise CanonicalMemoryInvariantError("canonical scope visibility or authority is invalid")
        if not memory_scope.authority.principal_ids and not memory_scope.authority.service_ids:
            return
        asserted_by = str(metadata.get("asserted_by") or "")
        asserted_by_service = str(metadata.get("asserted_by_service") or "")
        if (
            asserted_by not in set(memory_scope.authority.principal_ids)
            and asserted_by_service not in set(memory_scope.authority.service_ids)
        ):
            raise CanonicalMemoryInvariantError("canonical scope assertion authority is invalid")
