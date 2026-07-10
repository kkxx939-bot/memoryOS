from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.transition import MemoryStateTransition, MemoryTransitionPolicy
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class RevisionConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedMemoryOperation:
    context_object: ContextObject
    expected_revision: int
    content: str


@dataclass(frozen=True)
class MemoryTransactionPlan:
    transaction_id: str
    idempotency_key: str
    expected_revisions: Mapping[str, int]
    operations: tuple[PlannedMemoryOperation, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    policy_version: str
    schema_version: str
    proposal_ids: tuple[str, ...]
    proposal_fingerprints: tuple[str, ...]

    def to_context_operations(self, *, user_id: str, tenant_id: str, episode_id: str) -> list[ContextOperation]:
        results = []
        for planned in self.operations:
            obj = planned.context_object
            obj.metadata = {
                **dict(obj.metadata or {}),
                "canonical_transaction_id": self.transaction_id,
                "canonical_idempotency_key": self.idempotency_key,
            }
            operation_id = f"op_{stable_hash([self.idempotency_key, obj.uri], length=32)}"
            results.append(
                ContextOperation(
                    context_type=obj.context_type,
                    action=OperationAction.ADD if planned.expected_revision == 0 else OperationAction.UPDATE,
                    target_uri=obj.uri,
                    user_id=user_id,
                    operation_id=operation_id,
                    source_episode_id=episode_id,
                    evidence=[ref.to_dict() for ref in self.evidence_refs],
                    payload={
                        "canonical_memory": True,
                        "transaction_id": self.transaction_id,
                        "idempotency_key": self.idempotency_key,
                        "expected_revision": planned.expected_revision,
                        "policy_version": self.policy_version,
                        "schema_version": self.schema_version,
                        "proposal_ids": list(self.proposal_ids),
                        "proposal_fingerprints": list(self.proposal_fingerprints),
                        "tenant_id": tenant_id,
                        "context_object": obj.to_dict(),
                        "content": planned.content,
                    },
                )
            )
        return results


class MemoryTransactionPlanner:
    SCHEMA_VERSION = "canonical_memory_v1"

    def build(
        self,
        proposal: MemorySemanticProposal,
        memory_scope: MemoryScope,
        transition: MemoryStateTransition,
        *,
        tenant_id: str,
        owner_user_id: str,
        episode_id: str,
    ) -> MemoryTransactionPlan:
        scope_payload = memory_scope.to_dict()
        changed = set(transition.changed_claim_ids)
        planned: list[PlannedMemoryOperation] = []
        if not changed:
            idempotency_key = stable_hash(
                [tenant_id, owner_user_id, episode_id, proposal.fingerprint, "no_change"],
                length=40,
            )
            return MemoryTransactionPlan(
                transaction_id=f"memory_tx_{idempotency_key}",
                idempotency_key=idempotency_key,
                expected_revisions=dict(transition.expected_revisions),
                operations=(),
                evidence_refs=proposal.evidence_refs,
                policy_version=MemoryTransitionPolicy.VERSION,
                schema_version=self.SCHEMA_VERSION,
                proposal_ids=(proposal.proposal_id,),
                proposal_fingerprints=(proposal.fingerprint,),
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
                if existing.claim_id in proposal.related_memory_ids or existing.uri in proposal.related_memory_ids
            }
            obj.relations.extend(
                ContextRelation(
                    source_uri=obj.uri,
                    relation_type=transition.relation.value.lower(),
                    target_uri=target_uri,
                    metadata={"tenant_id": tenant_id, "owner_user_id": owner_user_id},
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
                            "revisions": [revision.to_dict() for revision in claim.revisions],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        slot_obj = transition.slot.to_context_object(
            tenant_id=tenant_id, owner_user_id=owner_user_id, scope=scope_payload
        )
        planned.append(
            PlannedMemoryOperation(
                context_object=slot_obj,
                expected_revision=int(transition.expected_revisions.get(slot_obj.uri, 0)),
                content=json.dumps(
                    {
                        "slot_id": transition.slot.slot_id,
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
                owner_user_id,
                episode_id,
                proposal.fingerprint,
                transition.slot.slot_id,
                transition.slot.revision,
                sorted(transition.changed_claim_ids),
            ],
            length=40,
        )
        return MemoryTransactionPlan(
            transaction_id=f"memory_tx_{idempotency_key}",
            idempotency_key=idempotency_key,
            expected_revisions=dict(transition.expected_revisions),
            operations=tuple(planned),
            evidence_refs=proposal.evidence_refs,
            policy_version=MemoryTransitionPolicy.VERSION,
            schema_version=self.SCHEMA_VERSION,
            proposal_ids=(proposal.proposal_id,),
            proposal_fingerprints=(proposal.fingerprint,),
        )
