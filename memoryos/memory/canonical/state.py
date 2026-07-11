"""Canonical memory state and structural invariants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.time import utc_now
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2
from memoryos.memory.canonical.scope import ScopeRef


class CanonicalMemoryInvariantError(ValueError):
    """Base class for corrupted canonical state."""


class RevisionSequenceError(CanonicalMemoryInvariantError):
    def __init__(self, claim_id: str, revisions: tuple[int, ...]) -> None:
        self.claim_id = claim_id
        self.revisions = revisions
        super().__init__(f"claim {claim_id} has a non-contiguous revision sequence: {revisions}")


class MissingClaimInvariantError(CanonicalMemoryInvariantError):
    def __init__(self, slot_id: str, missing_claim_ids: tuple[str, ...]) -> None:
        self.slot_id = slot_id
        self.missing_claim_ids = missing_claim_ids
        super().__init__(f"slot {slot_id} is missing claims: {','.join(missing_claim_ids)}")


class ActiveClaimInvariantError(CanonicalMemoryInvariantError):
    def __init__(
        self,
        slot_id: str,
        active_claim_ids: tuple[str, ...],
        declared_active_claim_id: str | None,
    ) -> None:
        self.slot_id = slot_id
        self.active_claim_ids = active_claim_ids
        self.declared_active_claim_id = declared_active_claim_id
        super().__init__(
            f"slot {slot_id} ACTIVE invariant violated: active={active_claim_ids}, declared={declared_active_claim_id}"
        )


class TransitionProfile(str, Enum):
    AUTHORITATIVE_STATE = "AUTHORITATIVE_STATE"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPERIENCE = "EXPERIENCE"


class ClaimState(str, Enum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    CONFLICTED = "CONFLICTED"
    RETRACTED = "RETRACTED"


CLAIM_STATES = frozenset(state.value for state in ClaimState)


def states_for(profile: TransitionProfile) -> frozenset[str]:  # noqa: ARG001
    return CLAIM_STATES


@dataclass(frozen=True)
class MemoryRevision:
    """An immutable transaction revision with separate effective time."""

    revision: int
    state: str
    value_fields: Mapping[str, Any]
    evidence_refs: tuple[EvidenceRef, ...]
    proposal_id: str
    relation: str
    epistemic_status: str
    field_evidence_refs: Mapping[str, tuple[EvidenceRef, ...]] = field(default_factory=dict)
    proposal_fingerprint: str = ""
    extractor_version: str = ""
    model_id: str | None = None
    prompt_version: str = ""
    policy_version: str = ""
    schema_version: str = ""
    qualifiers: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    previous_revision: int | None = None
    valid_from: str = ""
    valid_to: str | None = None
    transaction_time: str = ""

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError("memory revision must be positive")
        if self.state not in CLAIM_STATES:
            raise ValueError(f"invalid claim state: {self.state}")
        object.__setattr__(self, "value_fields", MappingProxyType(dict(self.value_fields)))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(
            self,
            "field_evidence_refs",
            MappingProxyType(
                {str(field_name): tuple(refs) for field_name, refs in sorted(dict(self.field_evidence_refs).items())}
            ),
        )
        object.__setattr__(self, "qualifiers", MappingProxyType(dict(self.qualifiers)))
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())
        if not self.transaction_time:
            object.__setattr__(self, "transaction_time", self.created_at)
        if not self.valid_from:
            object.__setattr__(self, "valid_from", self.created_at)

    @property
    def historical_only(self) -> bool:
        return bool(self.qualifiers.get("non_current_historical", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "state": self.state,
            "value_fields": dict(self.value_fields),
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "field_evidence_refs": {
                field_name: [ref.to_dict() for ref in refs] for field_name, refs in self.field_evidence_refs.items()
            },
            "proposal_id": self.proposal_id,
            "relation": self.relation,
            "epistemic_status": self.epistemic_status,
            "proposal_fingerprint": self.proposal_fingerprint,
            "extractor_version": self.extractor_version,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            "qualifiers": dict(self.qualifiers),
            "created_at": self.created_at,
            "transaction_time": self.transaction_time,
            "previous_revision": self.previous_revision,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryRevision:
        return cls(
            revision=int(payload["revision"]),
            state=str(payload["state"]),
            value_fields=dict(payload.get("value_fields", {}) or {}),
            evidence_refs=tuple(EvidenceRef(**dict(ref)) for ref in payload.get("evidence_refs", []) or []),
            proposal_id=str(payload.get("proposal_id", "")),
            relation=str(payload.get("relation", "UNRELATED")),
            epistemic_status=str(payload.get("epistemic_status", "INFERRED")),
            field_evidence_refs={
                str(field_name): tuple(EvidenceRef(**dict(ref)) for ref in refs)
                for field_name, refs in dict(payload.get("field_evidence_refs", {}) or {}).items()
            },
            proposal_fingerprint=str(payload.get("proposal_fingerprint", "")),
            extractor_version=str(payload.get("extractor_version", "")),
            model_id=str(payload["model_id"]) if payload.get("model_id") else None,
            prompt_version=str(payload.get("prompt_version", "")),
            policy_version=str(payload.get("policy_version", "")),
            schema_version=str(payload.get("schema_version", "")),
            qualifiers=dict(payload.get("qualifiers", {}) or {}),
            created_at=str(payload.get("created_at", "")),
            transaction_time=str(payload.get("transaction_time", "")),
            previous_revision=int(payload["previous_revision"])
            if payload.get("previous_revision") is not None
            else None,
            valid_from=str(payload.get("valid_from", "")),
            valid_to=str(payload["valid_to"]) if payload.get("valid_to") else None,
        )


@dataclass(frozen=True)
class MemoryClaim:
    claim_id: str
    uri: str
    slot_id: str
    canonical_value: str
    profile: TransitionProfile
    revisions: tuple[MemoryRevision, ...]
    identity_algorithm_version: str = IDENTITY_ALGORITHM_V2
    canonical_subject_key: str = ""

    def __post_init__(self) -> None:
        if self.identity_algorithm_version != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError("canonical Claim must use Identity V2")
        object.__setattr__(self, "revisions", tuple(self.revisions))
        if not self.revisions:
            raise ValueError("memory claim must have at least one revision")
        revision_numbers = tuple(revision.revision for revision in self.revisions)
        expected = tuple(range(1, len(self.revisions) + 1))
        if revision_numbers != expected:
            raise RevisionSequenceError(self.claim_id, revision_numbers)
        for index, revision in enumerate(self.revisions):
            if revision.state not in states_for(self.profile):
                raise ValueError(f"invalid {self.profile.value} state: {revision.state}")
            expected_previous = index if index else None
            if revision.previous_revision is not None and revision.previous_revision != expected_previous:
                raise RevisionSequenceError(self.claim_id, revision_numbers)

    @property
    def latest_revision(self) -> MemoryRevision:
        return self.revisions[-1]

    @property
    def current(self) -> MemoryRevision:
        return next(
            (revision for revision in reversed(self.revisions) if not revision.historical_only), self.latest_revision
        )

    def with_revision(self, revision: MemoryRevision) -> MemoryClaim:
        if revision.revision != self.latest_revision.revision + 1:
            raise ValueError("claim revision must increase by exactly one")
        if revision.state not in states_for(self.profile):
            raise ValueError(f"invalid {self.profile.value} state: {revision.state}")
        revisions = list(self.revisions)
        if not revision.historical_only:
            current_index = revisions.index(self.current)
            revisions[current_index] = replace(self.current, valid_to=revision.valid_from)
        revisions.append(revision)
        return MemoryClaim(
            self.claim_id,
            self.uri,
            self.slot_id,
            self.canonical_value,
            self.profile,
            tuple(revisions),
            self.identity_algorithm_version,
            self.canonical_subject_key,
        )

    def to_context_object(
        self, *, tenant_id: str, owner_user_id: str, memory_type: str, scope: dict[str, Any]
    ) -> ContextObject:
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.MEMORY,
            title=f"{memory_type}: {self.canonical_value}",
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={
                "canonical_kind": "claim",
                "memory_type": memory_type,
                "slot_id": self.slot_id,
                "claim_id": self.claim_id,
                "identity_algorithm_version": self.identity_algorithm_version,
                "canonical_subject": self.canonical_subject_key,
                "asserted_by": owner_user_id,
                "shared_authority": bool(
                    self.canonical_subject_key and ":principal:" not in self.canonical_subject_key
                ),
                "canonical_value": self.canonical_value,
                "transition_profile": self.profile.value,
                "state": self.current.state,
                "epistemic_status": self.current.epistemic_status,
                "semantic_relation": self.current.relation,
                "revision": self.latest_revision.revision,
                "current_revision": self.current.revision,
                "revisions": [revision.to_dict() for revision in self.revisions],
                "scope": scope,
                "projection_pending": True,
            },
            created_at=self.revisions[0].created_at,
            updated_at=self.latest_revision.transaction_time,
            schema_version="canonical_memory_v2",
        )


@dataclass(frozen=True)
class MemorySlot:
    slot_id: str
    uri: str
    memory_type: str
    identity_fields: Mapping[str, Any]
    scope_keys: tuple[str, ...]
    claim_ids: tuple[str, ...] = ()
    active_claim_id: str | None = None
    revision: int = 0
    identity_algorithm_version: str = IDENTITY_ALGORITHM_V2
    canonical_subject_key: str = ""
    canonical_subject: ScopeRef | None = None

    def __post_init__(self) -> None:
        if self.identity_algorithm_version != IDENTITY_ALGORITHM_V2:
            raise CanonicalMemoryInvariantError("canonical Slot must use Identity V2")
        object.__setattr__(self, "identity_fields", MappingProxyType(dict(self.identity_fields)))
        object.__setattr__(self, "scope_keys", tuple(sorted(dict.fromkeys(self.scope_keys))))
        claim_ids = tuple(self.claim_ids)
        if len(claim_ids) != len(set(claim_ids)):
            raise CanonicalMemoryInvariantError(f"slot {self.slot_id} contains duplicate claim IDs")
        object.__setattr__(self, "claim_ids", claim_ids)
        if self.revision < 0:
            raise ValueError("slot revision cannot be negative")
        if (
            self.canonical_subject is not None
            and self.canonical_subject_key
            and self.canonical_subject.key != self.canonical_subject_key
        ):
            raise ValueError("slot canonical subject payload does not match its identity key")

    def validate_claims(self, claims: tuple[MemoryClaim, ...]) -> None:
        by_id = {claim.claim_id: claim for claim in claims}
        missing = tuple(claim_id for claim_id in self.claim_ids if claim_id not in by_id)
        if missing:
            raise MissingClaimInvariantError(self.slot_id, missing)
        if any(claim.slot_id != self.slot_id for claim in claims):
            raise CanonicalMemoryInvariantError(f"slot {self.slot_id} contains a claim from another slot")
        active = tuple(claim.claim_id for claim in claims if claim.current.state == ClaimState.ACTIVE.value)
        if len(active) > 1 or (active and self.active_claim_id != active[0]) or (not active and self.active_claim_id):
            raise ActiveClaimInvariantError(self.slot_id, active, self.active_claim_id)

    def to_context_object(self, *, tenant_id: str, owner_user_id: str, scope: dict[str, Any]) -> ContextObject:
        scope_payload = dict(scope)
        if self.canonical_subject is not None:
            scope_payload["canonical_subject"] = self.canonical_subject.to_dict()
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.MEMORY,
            title=f"slot: {self.memory_type}",
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={
                "canonical_kind": "slot",
                "memory_type": self.memory_type,
                "slot_id": self.slot_id,
                "identity_algorithm_version": self.identity_algorithm_version,
                "canonical_subject": self.canonical_subject_key,
                "asserted_by": owner_user_id,
                "shared_authority": bool(
                    self.canonical_subject_key and ":principal:" not in self.canonical_subject_key
                ),
                "identity_fields": dict(self.identity_fields),
                "scope_keys": list(self.scope_keys),
                "claim_ids": list(self.claim_ids),
                "active_claim_id": self.active_claim_id,
                "revision": self.revision,
                "scope": scope_payload,
                "projection_pending": False,
            },
            schema_version="canonical_memory_v2",
        )


def profile_for(memory_type: str) -> TransitionProfile:
    if memory_type in {"profile", "preference", "project_rule", "project_decision"}:
        return TransitionProfile.AUTHORITATIVE_STATE
    if memory_type == "agent_experience":
        return TransitionProfile.EXPERIENCE
    return TransitionProfile.OBSERVATIONAL
