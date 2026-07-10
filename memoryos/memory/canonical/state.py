from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.time import utc_now
from memoryos.memory.canonical.evidence import EvidenceRef


class TransitionProfile(str, Enum):
    AUTHORITATIVE_STATE = "AUTHORITATIVE_STATE"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPERIENCE = "EXPERIENCE"


AUTHORITATIVE_STATES = frozenset({"PROPOSED", "PENDING", "ACTIVE", "SUPERSEDED", "RETRACTED", "CONFLICTED"})
OBSERVATIONAL_STATES = frozenset({"OBSERVED", "ACTIVE", "STALE", "ARCHIVED", "CONFLICTED"})
EXPERIENCE_STATES = frozenset({"OBSERVED", "VALIDATED", "ACTIVE", "STALE", "ARCHIVED"})


def states_for(profile: TransitionProfile) -> frozenset[str]:
    return {
        TransitionProfile.AUTHORITATIVE_STATE: AUTHORITATIVE_STATES,
        TransitionProfile.OBSERVATIONAL: OBSERVATIONAL_STATES,
        TransitionProfile.EXPERIENCE: EXPERIENCE_STATES,
    }[profile]


@dataclass(frozen=True)
class MemoryRevision:
    revision: int
    state: str
    value_fields: Mapping[str, Any]
    evidence_refs: tuple[EvidenceRef, ...]
    proposal_id: str
    relation: str
    epistemic_status: str
    proposal_fingerprint: str = ""
    extractor_version: str = ""
    model_id: str | None = None
    prompt_version: str = ""
    policy_version: str = ""
    schema_version: str = ""
    qualifiers: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "state": self.state,
            "value_fields": dict(self.value_fields),
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
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
            proposal_fingerprint=str(payload.get("proposal_fingerprint", "")),
            extractor_version=str(payload.get("extractor_version", "")),
            model_id=str(payload["model_id"]) if payload.get("model_id") else None,
            prompt_version=str(payload.get("prompt_version", "")),
            policy_version=str(payload.get("policy_version", "")),
            schema_version=str(payload.get("schema_version", "")),
            qualifiers=dict(payload.get("qualifiers", {}) or {}),
            created_at=str(payload.get("created_at", "")),
        )


@dataclass(frozen=True)
class MemoryClaim:
    claim_id: str
    uri: str
    slot_id: str
    canonical_value: str
    profile: TransitionProfile
    revisions: tuple[MemoryRevision, ...]

    @property
    def current(self) -> MemoryRevision:
        if not self.revisions:
            raise ValueError("memory claim must have at least one revision")
        return self.revisions[-1]

    def with_revision(self, revision: MemoryRevision) -> MemoryClaim:
        if revision.revision != self.current.revision + 1:
            raise ValueError("claim revision must increase by exactly one")
        if revision.state not in states_for(self.profile):
            raise ValueError(f"invalid {self.profile.value} state: {revision.state}")
        return MemoryClaim(
            self.claim_id,
            self.uri,
            self.slot_id,
            self.canonical_value,
            self.profile,
            (*self.revisions, revision),
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
                "canonical_value": self.canonical_value,
                "transition_profile": self.profile.value,
                "state": self.current.state,
                "epistemic_status": self.current.epistemic_status,
                "semantic_relation": self.current.relation,
                "revision": self.current.revision,
                "revisions": [revision.to_dict() for revision in self.revisions],
                "scope": scope,
                "projection_pending": True,
            },
            created_at=self.revisions[0].created_at,
            updated_at=self.current.created_at,
            schema_version="canonical_memory_v1",
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

    def to_context_object(self, *, tenant_id: str, owner_user_id: str, scope: dict[str, Any]) -> ContextObject:
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
                "identity_fields": dict(self.identity_fields),
                "scope_keys": list(self.scope_keys),
                "claim_ids": list(self.claim_ids),
                "active_claim_id": self.active_claim_id,
                "revision": self.revision,
                "scope": scope,
                "projection_pending": False,
            },
            schema_version="canonical_memory_v1",
        )


def profile_for(memory_type: str) -> TransitionProfile:
    if memory_type in {"profile", "preference", "project_rule", "project_decision"}:
        return TransitionProfile.AUTHORITATIVE_STATE
    if memory_type == "agent_experience":
        return TransitionProfile.EXPERIENCE
    return TransitionProfile.OBSERVATIONAL
