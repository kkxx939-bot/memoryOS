"""Immutable canonical memory transaction planning."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.event import canonicalize, immutable_snapshot
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.transition import MemoryStateTransition, MemoryTransitionPolicy
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class RevisionConflictError(RuntimeError):
    """Raised when a planned revision no longer matches canonical state."""

    def __init__(self, message: str, *, committed_diff=None) -> None:  # noqa: ANN001
        self.committed_diff = committed_diff
        super().__init__(message)


@dataclass(frozen=True)
class PlannedMemoryOperation:
    context_object: ContextObject
    expected_revision: int
    content: str

    def __post_init__(self) -> None:
        if self.expected_revision < 0:
            raise ValueError("expected revision cannot be negative")
        object.__setattr__(self, "context_object", deepcopy(self.context_object))
        object.__setattr__(self, "content", str(self.content))

    def context_object_copy(self) -> ContextObject:
        return deepcopy(self.context_object)


@dataclass(frozen=True)
class MemoryTransactionPlan:
    """A stable, replayable snapshot of one canonical Slot transaction."""

    transaction_id: str
    idempotency_key: str
    tenant_id: str
    owner_user_id: str
    slot_id: str
    expected_revisions: Mapping[str, int]
    operations: tuple[PlannedMemoryOperation, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    policy_version: str
    schema_version: str
    proposal_ids: tuple[str, ...]
    proposal_fingerprints: tuple[str, ...]
    proposal_proofs: tuple[Mapping[str, object], ...]
    identity_algorithm_version: str = IDENTITY_ALGORITHM_V2
    canonical_subject_key: str = ""
    commit_group_id: str = ""
    planning_task_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_revisions",
            MappingProxyType({str(uri): int(revision) for uri, revision in sorted(self.expected_revisions.items())}),
        )
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "proposal_ids", tuple(self.proposal_ids))
        object.__setattr__(self, "proposal_fingerprints", tuple(self.proposal_fingerprints))
        object.__setattr__(
            self,
            "proposal_proofs",
            tuple(immutable_snapshot(dict(item)) for item in self.proposal_proofs),
        )
        if not self.commit_group_id:
            object.__setattr__(self, "commit_group_id", f"commit_group_{self.idempotency_key}")
        if not self.planning_task_id:
            object.__setattr__(
                self, "planning_task_id", self.proposal_ids[0] if self.proposal_ids else self.transaction_id
            )

    def to_context_operations(self, *, user_id: str, tenant_id: str, episode_id: str) -> list[ContextOperation]:
        results = []
        for planned in self.operations:
            obj = planned.context_object_copy()
            obj.metadata = {
                **dict(obj.metadata or {}),
                "canonical_transaction_id": self.transaction_id,
                "canonical_idempotency_key": self.idempotency_key,
                "identity_algorithm_version": self.identity_algorithm_version,
                "canonical_subject": self.canonical_subject_key,
                "commit_group_id": self.commit_group_id,
            }
            operation_id = f"op_{stable_hash([self.idempotency_key, obj.uri], length=32)}"
            metadata = dict(obj.metadata or {})
            results.append(
                ContextOperation(
                    context_type=obj.context_type,
                    action=OperationAction.ADD if planned.expected_revision == 0 else OperationAction.UPDATE,
                    target_uri=obj.uri,
                    user_id=user_id,
                    operation_id=operation_id,
                    source_episode_id=episode_id,
                    created_at=self.created_at,
                    evidence=[ref.to_dict() for ref in self.evidence_refs],
                    payload={
                        "canonical_memory": True,
                        "transaction_id": self.transaction_id,
                        "idempotency_key": self.idempotency_key,
                        "commit_group_id": self.commit_group_id,
                        "planning_task_id": self.planning_task_id,
                        "identity_algorithm_version": self.identity_algorithm_version,
                        "canonical_subject": self.canonical_subject_key,
                        "expected_revision": planned.expected_revision,
                        "expected_revisions": dict(self.expected_revisions),
                        "slot_id": str(metadata.get("slot_id") or self.slot_id),
                        "claim_id": str(metadata.get("claim_id") or ""),
                        "memory_type": str(metadata.get("memory_type") or ""),
                        "policy_version": self.policy_version,
                        "schema_version": self.schema_version,
                        "proposal_ids": list(self.proposal_ids),
                        "proposal_fingerprints": list(self.proposal_fingerprints),
                        "proposal_proofs": [canonicalize(item) for item in self.proposal_proofs],
                        "tenant_id": tenant_id,
                        "context_object": obj.to_dict(),
                        "content": planned.content,
                    },
                )
            )
        return results


class MemoryTransactionPlanner:
    """Build plans only; all writes remain in OperationCommitter."""

    SCHEMA_VERSION = "canonical_memory_v2"

    def build(
        self,
        proposal: MemorySemanticProposal,
        memory_scope: MemoryScope,
        transition: MemoryStateTransition,
        *,
        tenant_id: str,
        owner_user_id: str,
        episode_id: str,
        commit_group_id: str = "",
        planning_task_id: str = "",
    ) -> MemoryTransactionPlan:
        scope_payload = memory_scope.to_dict()
        # Identity resolution canonicalizes aliases, case, and hierarchy.  The
        # persisted subject must be that exact object; retaining the
        # pre-resolution payload can make its key disagree with the Slot key.
        if transition.slot.canonical_subject is not None:
            scope_payload["canonical_subject"] = transition.slot.canonical_subject.to_dict()
        changed = set(transition.changed_claim_ids)
        planned: list[PlannedMemoryOperation] = []
        identity_version = transition.slot.identity_algorithm_version
        canonical_subject = transition.slot.canonical_subject_key
        transaction_time = self._transaction_time(proposal, transition)
        if not changed:
            idempotency_key = stable_hash(
                [
                    tenant_id,
                    episode_id,
                    commit_group_id,
                    proposal.fingerprint,
                    transition.slot.slot_id,
                    identity_version,
                    canonical_subject,
                    "no_change",
                ],
                length=40,
            )
            return MemoryTransactionPlan(
                transaction_id=f"memory_tx_{idempotency_key}",
                idempotency_key=idempotency_key,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                slot_id=transition.slot.slot_id,
                expected_revisions=dict(transition.expected_revisions),
                operations=(),
                evidence_refs=proposal.evidence_refs,
                policy_version=MemoryTransitionPolicy.VERSION,
                schema_version=self.SCHEMA_VERSION,
                proposal_ids=(proposal.proposal_id,),
                proposal_fingerprints=(proposal.fingerprint,),
                proposal_proofs=(proposal.to_dict(),),
                identity_algorithm_version=identity_version,
                canonical_subject_key=canonical_subject,
                commit_group_id=commit_group_id,
                planning_task_id=planning_task_id or proposal.proposal_id,
                created_at=transaction_time,
            )
        for claim in transition.claims:
            if claim.claim_id not in changed:
                continue
            obj = claim.to_context_object(
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                memory_type=proposal.memory_type,
                scope=scope_payload,
            )
            related_uris = {
                existing.uri
                for existing in transition.claims
                if existing.claim_id in proposal.all_related_memory_ids
                or existing.uri in proposal.all_related_memory_ids
            }
            obj.relations.extend(
                ContextRelation(
                    source_uri=obj.uri,
                    relation_type=transition.relation.value.lower(),
                    target_uri=target_uri,
                    metadata={"tenant_id": tenant_id, "owner_user_id": owner_user_id},
                    created_at=transaction_time,
                )
                for target_uri in sorted(related_uris)
                if target_uri != obj.uri
            )
            planned.append(
                PlannedMemoryOperation(
                    context_object=obj,
                    expected_revision=int(transition.expected_revisions.get(obj.uri, 0)),
                    content=json.dumps(
                        {
                            "slot_id": claim.slot_id,
                            "claim_id": claim.claim_id,
                            "canonical_value": claim.canonical_value,
                            "current": claim.current.to_dict(),
                            "latest_revision": claim.latest_revision.revision,
                            "revisions": [revision.to_dict() for revision in claim.revisions],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        slot_obj = transition.slot.to_context_object(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            scope=scope_payload,
        )
        slot_obj.updated_at = transaction_time
        planned.append(
            PlannedMemoryOperation(
                context_object=slot_obj,
                expected_revision=int(transition.expected_revisions.get(slot_obj.uri, 0)),
                content=json.dumps(
                    {
                        "slot_id": transition.slot.slot_id,
                        "identity_algorithm_version": identity_version,
                        "canonical_subject": canonical_subject,
                        "identity_fields": dict(transition.slot.identity_fields),
                        "claim_ids": list(transition.slot.claim_ids),
                        "active_claim_id": transition.slot.active_claim_id,
                        "revision": transition.slot.revision,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        )
        idempotency_key = stable_hash(
            [
                tenant_id,
                episode_id,
                commit_group_id,
                proposal.fingerprint,
                transition.slot.slot_id,
                transition.slot.revision,
                identity_version,
                canonical_subject,
                sorted(transition.changed_claim_ids),
                sorted(transition.expected_revisions.items()),
            ],
            length=40,
        )
        return MemoryTransactionPlan(
            transaction_id=f"memory_tx_{idempotency_key}",
            idempotency_key=idempotency_key,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            slot_id=transition.slot.slot_id,
            expected_revisions=dict(transition.expected_revisions),
            operations=tuple(planned),
            evidence_refs=proposal.evidence_refs,
            policy_version=MemoryTransitionPolicy.VERSION,
            schema_version=self.SCHEMA_VERSION,
            proposal_ids=(proposal.proposal_id,),
            proposal_fingerprints=(proposal.fingerprint,),
            proposal_proofs=(proposal.to_dict(),),
            identity_algorithm_version=identity_version,
            canonical_subject_key=canonical_subject,
            commit_group_id=commit_group_id,
            planning_task_id=planning_task_id or proposal.proposal_id,
            created_at=transaction_time,
        )

    @staticmethod
    def _transaction_time(
        proposal: MemorySemanticProposal,
        transition: MemoryStateTransition,
    ) -> str:
        changed = set(transition.changed_claim_ids)
        times = sorted(
            claim.latest_revision.transaction_time
            for claim in transition.claims
            if claim.claim_id in changed and claim.latest_revision.transaction_time
        )
        if times:
            return times[-1]
        explicit = proposal.metadata.get("planning_timestamp")
        if explicit:
            return str(explicit)
        evidence_times = sorted(
            str(ref.ingested_at or ref.occurred_at)
            for ref in proposal.evidence_refs
            if ref.ingested_at or ref.occurred_at
        )
        return evidence_times[-1] if evidence_times else ""
