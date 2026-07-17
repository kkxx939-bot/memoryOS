"""Durable pending-memory review orchestration."""

from __future__ import annotations

from typing import Any

from memoryos.application.service import ApplicationService
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.transaction.path_lock import PathLock
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical import CanonicalMemoryRepository, MemorySemanticProposal
from memoryos.memory.canonical.current_head import artifact_root_for, load_current_head
from memoryos.memory.canonical.review_command import (
    PendingReviewCommandStore,
    PendingReviewIdempotencyConflict,
    validate_pending_review_record,
)
from memoryos.security.trusted_context import AUTHORITATIVE_REMEMBER, TrustedRequestContext


class PendingReviewService(ApplicationService):
    def review_pending(
        self,
        *,
        user_id: str,
        pending_uri: str,
        decision: str,
        expected_lifecycle_revision: int,
        expected_proposal_fingerprint: str,
        command_id: str,
        tenant_id: str | None = None,
        reason: str = "",
        corrected_proposal: MemorySemanticProposal | dict[str, Any] | None = None,
        caller: TrustedRequestContext | None = None,
        review_locked: Any | None = None,
    ) -> dict[str, Any]:
        """Apply a user-owned structured review without accepting arbitrary operations or targets."""

        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        if expected_lifecycle_revision < 1:
            raise ValueError("expected_lifecycle_revision must be positive")
        if not expected_proposal_fingerprint or not command_id:
            raise ValueError("pending review requires proposal fingerprint and command_id")
        normalized_decision = str(decision or "").strip().upper()
        allowed_decisions = {
            "CONFIRM",
            "CONFIRM_AND_APPLY",
            "CORRECT",
            "REJECT",
            "EXPIRE",
            "RETRY",
        }
        if normalized_decision not in allowed_decisions:
            raise ValueError(
                "pending review decision must be CONFIRM, CONFIRM_AND_APPLY, CORRECT, REJECT, EXPIRE, or RETRY"
            )
        if corrected_proposal is not None and not isinstance(corrected_proposal, MemorySemanticProposal | dict):
            raise ValueError("corrected_proposal must be a semantic proposal object")
        correction: MemorySemanticProposal | None = (
            corrected_proposal
            if isinstance(corrected_proposal, MemorySemanticProposal)
            else MemorySemanticProposal.from_dict(corrected_proposal)
            if isinstance(corrected_proposal, dict)
            else None
        )
        if (normalized_decision == "CORRECT") != (correction is not None):
            raise ValueError("CORRECT requires corrected_proposal and other decisions forbid it")
        correction_digest = stable_hash([correction.to_dict()], length=64) if correction is not None else ""
        review_store = PendingReviewCommandStore(self.root, tenant_id=tenant_id)
        lock_key = f"pending-review:{tenant_id}:{pending_uri}"
        with PathLock(self.lock_store).acquire(lock_key, ttl_seconds=120) as guard:
            guard.checkpoint()
            committed_pending = CanonicalMemoryRepository(
                self.source_store,
                self.relation_store,
            ).load_pending(
                pending_uri,
                tenant_id=tenant_id,
                owner_user_id=user_id,
            )
            if caller is not None:
                caller.require(AUTHORITATIVE_REMEMBER)
                caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
                self._require_exact_workspace(
                    {"scope": committed_pending.scope.to_dict()},
                    caller,
                    pending_uri,
                )
            if committed_pending.lifecycle_state == LifecycleState.CONFIRMED:
                for raw_transition in reversed(committed_pending.lifecycle_history):
                    transition = dict(raw_transition)
                    owning_command = str(transition.get("review_command_id") or "")
                    if (
                        str(transition.get("to") or "").casefold() != "confirmed"
                        or str(transition.get("review_decision") or "").upper() != "CONFIRM_AND_APPLY"
                        or not owning_command
                        or owning_command == command_id
                    ):
                        continue
                    owning_record = review_store.load(owning_command)
                    if owning_record.get("status") == "running":
                        raise PendingReviewIdempotencyConflict(
                            "another CONFIRM_AND_APPLY command owns the in-flight resolution"
                        )
                    break
            command_proof_preexisting = review_store.path(command_id).exists()
            command = review_store.begin(
                command_id,
                owner_user_id=user_id,
                pending_uri=pending_uri,
                decision=normalized_decision,
                expected_lifecycle_revision=expected_lifecycle_revision,
                expected_proposal_fingerprint=expected_proposal_fingerprint,
                reason=reason,
                correction_proposal_digest=correction_digest,
            )
            if command["status"] == "completed":
                validate_pending_review_record(command, committed_pending)
                return dict(command["result"])
            if command["status"] == "failed":
                error = dict(command.get("error", {}) or {})
                raise ValueError(
                    "pending review command previously failed: "
                    f"{error.get('type', 'UnknownError')}: {error.get('message', '')}"
                )
            self.committer.recover_pending_regular_memory(
                user_id,
                commit_group_id=f"pending-review:{command_id}",
            )
            self.committer.recover_pending_canonical(
                user_id,
                commit_group_id=f"pending-resolution:{command_id}",
            )
            self.committer.recover_pending_canonical(
                user_id,
                commit_group_id=f"pending-correction:{command_id}",
            )
            try:
                handler = review_locked or self._review_pending_locked
                result = handler(
                    user_id=user_id,
                    pending_uri=pending_uri,
                    normalized_decision=normalized_decision,
                    expected_lifecycle_revision=expected_lifecycle_revision,
                    expected_proposal_fingerprint=expected_proposal_fingerprint,
                    command_id=command_id,
                    tenant_id=tenant_id,
                    reason=reason,
                    corrected_proposal=correction,
                    caller=caller,
                    review_request_digest=str(command["request_digest"]),
                    command_proof_preexisting=command_proof_preexisting,
                )
            except (FileNotFoundError, PermissionError, KeyError, TypeError, ValueError) as exc:
                review_store.fail(command_id, exc)
                raise
            except (OSError, TimeoutError, RuntimeError):
                # The durable command stays ``running``.  A retry first
                # recovers receipt/head/redo state and then returns or
                # completes the exact same command effect.
                raise
            guard.checkpoint()
            review_store.complete(command_id, result)
            return result

    def _review_pending_locked(
        self,
        *,
        user_id: str,
        pending_uri: str,
        normalized_decision: str,
        expected_lifecycle_revision: int,
        expected_proposal_fingerprint: str,
        command_id: str,
        tenant_id: str,
        reason: str,
        corrected_proposal: MemorySemanticProposal | None,
        caller: TrustedRequestContext | None,
        review_request_digest: str,
        command_proof_preexisting: bool,
    ) -> dict[str, Any]:
        repository = CanonicalMemoryRepository(self.source_store, self.relation_store)
        pending = repository.load_pending(
            pending_uri,
            tenant_id=tenant_id,
            owner_user_id=user_id,
        )
        if caller is not None:
            caller.require(AUTHORITATIVE_REMEMBER)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            self._require_exact_workspace({"scope": pending.scope.to_dict()}, caller, pending_uri)
        if pending.proposal.fingerprint != expected_proposal_fingerprint:
            raise ValueError("pending review expected revision or proposal fingerprint mismatch")
        command_reason_prefix = f"structured_review:{command_id}"
        structured_history = [
            dict(item)
            for item in pending.lifecycle_history
            if str(dict(item).get("review_command_id") or "") == command_id
        ]
        if any(
            str(item.get("review_decision") or "").strip().upper() != normalized_decision
            or str(item.get("review_request_digest") or "") != review_request_digest
            for item in structured_history
        ):
            raise PendingReviewIdempotencyConflict(
                "pending review command_id is already bound by a receipt to a different decision or effect"
            )
        legacy_command_history = any(
            str(dict(item).get("reason") or "").startswith(
                (command_reason_prefix, f"structured_correction:{command_id}")
            )
            and not dict(item).get("review_command_id")
            for item in pending.lifecycle_history
        )
        if legacy_command_history and not command_proof_preexisting:
            raise PendingReviewIdempotencyConflict(
                "legacy pending review history has no durable request binding; command_id cannot be recreated"
            )
        command_history = bool(structured_history or legacy_command_history)
        if pending.lifecycle_revision != expected_lifecycle_revision and not command_history:
            raise ValueError("pending review expected revision or proposal fingerprint mismatch")
        pending.assert_review_decision(normalized_decision)
        formation = self.session_commit_service.memory_planner.formation
        review_reason = f"structured_review:{command_id}:{reason}".rstrip(":")
        if normalized_decision == "CORRECT":
            assert corrected_proposal is not None
            correction_prefix = f"structured_correction:{command_id}"
            correction_history = any(
                str(dict(item).get("reason") or "").startswith(correction_prefix) for item in pending.lifecycle_history
            )
            if pending.lifecycle_state == LifecycleState.REJECTED and correction_history:
                return self._pending_review_recovered_result(pending_uri, pending, ())
            evidence = corrected_proposal.atomic_evidence_ref or (
                corrected_proposal.evidence_refs[0] if corrected_proposal.evidence_refs else None
            )
            if evidence is None or not evidence.source_uri:
                raise ValueError("corrected proposal has no durable source archive")
            archive = self.session_archive_store.read_archive(evidence.source_uri, tenant_id=tenant_id)
            episode = self.session_commit_service.memory_planner.episode_adapter.adapt(archive)
            corrected = formation.plan_pending_correction(
                pending_uri,
                corrected_proposal,
                archive=archive,
                episode=episode,
                tenant_id=tenant_id,
                owner_user_id=user_id,
                commit_group_id=f"pending-correction:{command_id}",
                retrieval_views=list(pending.retrieval_views),
                reason=correction_prefix,
                review_command_id=command_id,
                review_decision=normalized_decision,
                review_request_digest=review_request_digest,
            )
            diff = self.committer.commit(user_id, list(corrected.operations))
            self._process_memory_projections_or_raise()
            final = repository.load_pending(
                pending_uri,
                tenant_id=tenant_id,
                owner_user_id=user_id,
            )
            corrected_claim_uris = tuple(
                str(operation.target_uri)
                for operation in corrected.operations
                if isinstance((payload := operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
                and dict(payload.get("metadata", {}) or {}).get("state") == "ACTIVE"
            )
            return {
                "uri": pending_uri,
                "status": final.lifecycle_state.value,
                "lifecycle_revision": final.lifecycle_revision,
                "corrected_claim_uris": list(corrected_claim_uris),
                "corrected_proposal_fingerprint": corrected.proposal.fingerprint,
                "diff_id": diff.diff_id,
            }
        terminal = {
            "REJECT": LifecycleState.REJECTED,
            "EXPIRE": LifecycleState.EXPIRED,
            "RETRY": LifecycleState.RETRYABLE,
            "CONFIRM": LifecycleState.CONFIRMED,
        }
        if normalized_decision in terminal:
            if pending.lifecycle_state == terminal[normalized_decision] and command_history:
                return self._pending_review_recovered_result(pending_uri, pending, ())
            operation = formation.plan_pending_lifecycle_transition(
                pending_uri,
                terminal[normalized_decision],
                tenant_id=tenant_id,
                owner_user_id=user_id,
                commit_group_id=f"pending-review:{command_id}",
                reason=review_reason,
                retry_increment=normalized_decision == "RETRY",
                review_command_id=command_id,
                review_decision=normalized_decision,
                review_request_digest=review_request_digest,
            )
            diff = self.committer.commit(user_id, [operation])
            updated = repository.load_pending(
                pending_uri,
                tenant_id=tenant_id,
                owner_user_id=user_id,
            )
            return {
                "uri": pending_uri,
                "status": updated.lifecycle_state.value,
                "lifecycle_revision": updated.lifecycle_revision,
                "diff_id": diff.diff_id,
            }
        if pending.lifecycle_state == LifecycleState.RESOLVED and command_history:
            return self._pending_review_recovered_result(pending_uri, pending, ())
        if pending.lifecycle_state in {LifecycleState.PENDING, LifecycleState.RETRYABLE}:
            confirmation = formation.plan_pending_lifecycle_transition(
                pending_uri,
                LifecycleState.CONFIRMED,
                tenant_id=tenant_id,
                owner_user_id=user_id,
                commit_group_id=f"pending-review:{command_id}",
                reason=review_reason,
                review_command_id=command_id,
                review_decision=normalized_decision,
                review_request_digest=review_request_digest,
            )
            self.committer.commit(user_id, [confirmation])
            pending = repository.load_pending(
                pending_uri,
                tenant_id=tenant_id,
                owner_user_id=user_id,
            )
        elif pending.lifecycle_state != LifecycleState.CONFIRMED:
            raise ValueError("only PENDING, RETRYABLE, or CONFIRMED proposals can be applied")
        evidence = pending.proposal.atomic_evidence_ref or (
            pending.proposal.evidence_refs[0] if pending.proposal.evidence_refs else None
        )
        if evidence is None or not evidence.source_uri:
            raise ValueError("confirmed pending proposal has no durable source archive")
        archive = self.session_archive_store.read_archive(evidence.source_uri, tenant_id=tenant_id)
        episode = self.session_commit_service.memory_planner.episode_adapter.adapt(archive)
        resolved = formation.plan_confirmed_pending_resolution(
            pending_uri,
            pending.proposal,
            archive=archive,
            episode=episode,
            tenant_id=tenant_id,
            owner_user_id=user_id,
            commit_group_id=f"pending-resolution:{command_id}",
            retrieval_views=list(pending.retrieval_views),
            reason=review_reason,
            review_command_id=command_id,
            review_decision=normalized_decision,
            review_request_digest=review_request_digest,
        )
        diff = self.committer.commit(user_id, list(resolved.operations))
        self._process_memory_projections_or_raise()
        final = repository.load_pending(
            pending_uri,
            tenant_id=tenant_id,
            owner_user_id=user_id,
        )
        claim_uris = [
            str(operation.target_uri)
            for operation in resolved.operations
            if isinstance((payload := operation.payload.get("context_object")), dict)
            and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
            and dict(payload.get("metadata", {}) or {}).get("state") == "ACTIVE"
        ]
        return {
            "uri": pending_uri,
            "status": final.lifecycle_state.value,
            "lifecycle_revision": final.lifecycle_revision,
            "resolved_claim_uris": claim_uris,
            "diff_id": diff.diff_id,
        }

    def _pending_review_recovered_result(
        self,
        pending_uri: str,
        pending: Any,
        claim_uris: tuple[str, ...],
    ) -> dict[str, Any]:
        diff_id = ""
        resolved_claim_uris = list(claim_uris)
        artifact_root = artifact_root_for(self.source_store)
        if artifact_root is not None:
            _head, receipt, _snapshot = load_current_head(
                artifact_root,
                pending_uri,
                canonical_kind="pending_proposal",
            )
            diff_id = str(dict(receipt.get("diff", {}) or {}).get("diff_id") or "")
            for operation in receipt.get("operations", []):
                if not isinstance(operation, dict) or operation.get("target_uri") != pending_uri:
                    continue
                resolved_claim_uris.extend(
                    str(item) for item in dict(operation.get("payload", {}) or {}).get("resolved_claim_uris", []) or []
                )
                corrected_claim_uris = [
                    str(item) for item in dict(operation.get("payload", {}) or {}).get("corrected_claim_uris", []) or []
                ]
                if corrected_claim_uris:
                    result_correction = {
                        "corrected_claim_uris": corrected_claim_uris,
                        "corrected_proposal_fingerprint": str(
                            dict(operation.get("payload", {}) or {}).get("corrected_proposal_fingerprint") or ""
                        ),
                    }
                    break
            else:
                result_correction = {}
        result: dict[str, Any] = {
            "uri": pending_uri,
            "status": pending.lifecycle_state.value,
            "lifecycle_revision": pending.lifecycle_revision,
            "diff_id": diff_id,
        }
        if pending.lifecycle_state == LifecycleState.RESOLVED:
            result["resolved_claim_uris"] = list(dict.fromkeys(resolved_claim_uris))
        result.update(result_correction if artifact_root is not None else {})
        return result



__all__ = ["PendingReviewService"]
