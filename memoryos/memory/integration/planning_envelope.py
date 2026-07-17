"""Create-only durable envelopes for deterministic memory planning retry."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memoryos.core.durable_io import atomic_create_json
from memoryos.core.file_lock import open_private_lock
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.integration.planning_context import (
    PlanningContext,
    PrefetchSnapshot,
    ProposalPlanningInput,
    ProposalPlanningOutcome,
    StagedObjectSnapshot,
)

try:  # pragma: no cover - production Unix platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

PLANNING_ENVELOPE_SCHEMA_VERSION = "memory_planning_envelope_v2"
PLANNING_ENVELOPE_ANCHOR_SCHEMA_VERSION = "memory_planning_envelope_anchor_v1"


class PlanningEnvelopeIntegrityError(RuntimeError):
    """A planning envelope is corrupt or reused for another proposal set."""


def validate_planning_envelope_payload(
    payload: object,
    *,
    tenant_id: str,
    task_id: str,
) -> dict[str, Any]:
    """Validate one immutable envelope independently of its directory scan."""

    if not isinstance(payload, dict) or payload.get("schema_version") != PLANNING_ENVELOPE_SCHEMA_VERSION:
        raise PlanningEnvelopeIntegrityError("planning envelope schema is unsupported")
    digest = payload.get("envelope_digest")
    core = {key: value for key, value in payload.items() if key != "envelope_digest"}
    if not isinstance(digest, str) or digest != canonical_digest(core):
        raise PlanningEnvelopeIntegrityError("planning envelope digest is corrupt")
    if payload.get("tenant_id") != tenant_id or payload.get("task_id") != task_id:
        raise PlanningEnvelopeIntegrityError("planning envelope crosses task or tenant boundary")
    proposal_inputs = payload.get("proposal_inputs")
    if not isinstance(proposal_inputs, list) or any(not isinstance(item, dict) for item in proposal_inputs):
        raise PlanningEnvelopeIntegrityError("planning envelope proposal set is invalid")
    if any(not isinstance(item.get("proposal"), dict) for item in proposal_inputs):
        raise PlanningEnvelopeIntegrityError("planning envelope contains an invalid normalized proposal")
    proposal_set_digest = canonical_digest(
        [
            {
                "proposal": item.get("proposal"),
                "retrieval_views": item.get("retrieval_views", []),
                "forced_pending_reason": item.get("forced_pending_reason", ""),
            }
            for item in proposal_inputs
        ]
    )
    if proposal_set_digest != payload.get("proposal_set_digest"):
        raise PlanningEnvelopeIntegrityError("planning envelope proposal set was modified")
    if payload.get("planning_digest") != PlanningEnvelopeStore.planning_digest(core):
        raise PlanningEnvelopeIntegrityError("planning envelope planning digest is corrupt")
    required_strings = (
        "user_id",
        "archive_uri",
        "archive_digest",
        "manifest_digest",
        "episode_id",
        "session_id",
        "planning_id",
        "operation_group_identity",
        "extractor_version",
    )
    if any(not isinstance(payload.get(field), str) or not payload.get(field) for field in required_strings):
        raise PlanningEnvelopeIntegrityError("planning envelope identity or extractor version is missing")
    if payload.get("commit_group_id") != payload.get("operation_group_identity"):
        raise PlanningEnvelopeIntegrityError("planning envelope commit group identity disagrees")
    reservation_digest = payload.get("salience_reservation_digest")
    if not isinstance(reservation_digest, str) or len(reservation_digest) != 64:
        raise PlanningEnvelopeIntegrityError("planning envelope has no durable salience reservation binding")
    egress_decision = payload.get("egress_decision")
    if egress_decision not in {"ALLOW", "ALLOW_REDACTED", "LOCAL_ONLY", "DENY"}:
        raise PlanningEnvelopeIntegrityError("planning envelope egress decision is invalid")
    egress_audit = payload.get("egress_audit")
    if not isinstance(egress_audit, dict) or set(egress_audit) != {
        "outbound_digest",
        "decision",
        "provider",
        "model",
    }:
        raise PlanningEnvelopeIntegrityError("planning envelope egress audit fields are invalid")
    outbound_digest = egress_audit.get("outbound_digest")
    if (
        egress_audit.get("decision") != egress_decision
        or not all(isinstance(egress_audit.get(field), str) for field in ("provider", "model"))
        or not isinstance(outbound_digest, str)
        or (
            outbound_digest != ""
            and (
                len(outbound_digest) != 64 or any(character not in "0123456789abcdef" for character in outbound_digest)
            )
        )
        or (egress_decision in {"LOCAL_ONLY", "DENY"} and outbound_digest != "")
        or (egress_decision in {"ALLOW", "ALLOW_REDACTED"} and len(outbound_digest) != 64)
    ):
        raise PlanningEnvelopeIntegrityError("planning envelope egress audit binding is invalid")
    for field in ("prefetch_snapshot", "staged_objects"):
        rows = payload.get(field)
        if not isinstance(rows, list):
            raise PlanningEnvelopeIntegrityError(f"planning envelope {field} is invalid")
        expected_keys = (
            {"uri", "revision", "object_digest", "content_digest", "relation_digest"}
            if field == "prefetch_snapshot"
            else {"uri", "revision", "object_digest"}
        )
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != expected_keys
                or any(
                    key.endswith("_digest") and (not isinstance(value, str) or len(value) != 64)
                    for key, value in row.items()
                )
            ):
                raise PlanningEnvelopeIntegrityError(f"planning envelope {field} contains raw or invalid staged state")
    if any(
        "payload_json" in item for item in [*payload.get("prefetch_snapshot", []), *payload.get("staged_objects", [])]
    ):
        raise PlanningEnvelopeIntegrityError("planning envelope contains duplicated raw Source payload")
    outcomes = payload.get("candidate_outcomes")
    if not isinstance(outcomes, list) or any(not isinstance(item, dict) for item in outcomes):
        raise PlanningEnvelopeIntegrityError("planning envelope candidate outcomes are invalid")
    decisions = {str(item.get("proposal_id") or ""): str(item.get("decision") or "") for item in outcomes}
    accepted = [
        item["proposal"]
        for item in proposal_inputs
        if decisions.get(str(dict(item["proposal"]).get("proposal_id") or "")) == "ACCEPT_FOR_RECONCILE"
    ]
    pending = [
        {"proposal": item["proposal"], "reason": item.get("forced_pending_reason", "")}
        for item in proposal_inputs
        if item.get("forced_pending_reason")
    ]
    if payload.get("normalized_accepted_proposals") != accepted or payload.get("pending_proposal_inputs") != pending:
        raise PlanningEnvelopeIntegrityError("planning envelope admitted proposal classification is invalid")
    return payload


def canonical_direct_planning_digest(operations: Sequence[Any]) -> str:
    """Content-bound proof for non-model direct canonical commands."""

    normalized = [item.to_dict() if callable(getattr(item, "to_dict", None)) else dict(item) for item in operations]
    if not normalized:
        raise ValueError("direct canonical planning requires operations")
    return canonical_digest(
        {
            "schema_version": "canonical_implicit_planning_v1",
            "transaction_id": str(dict(normalized[0].get("payload", {}) or {}).get("transaction_id") or ""),
            "proposal_fingerprints": sorted(
                {
                    str(value)
                    for operation in normalized
                    for value in dict(operation.get("payload", {}) or {}).get("proposal_fingerprints", []) or []
                }
            ),
            "operations": [
                {
                    "operation_id": str(operation.get("operation_id") or ""),
                    "target_uri": operation.get("target_uri"),
                    "expected_revision": dict(operation.get("payload", {}) or {}).get("expected_revision", 0),
                }
                for operation in sorted(normalized, key=lambda item: str(item.get("operation_id") or ""))
            ],
        }
    )


def pending_direct_planning_digest(operation: Any) -> str:
    normalized = operation.to_dict() if callable(getattr(operation, "to_dict", None)) else dict(operation)
    payload = dict(normalized.get("payload", {}) or {})
    context_object = payload.get("context_object")
    metadata = dict(context_object.get("metadata", {}) or {}) if isinstance(context_object, dict) else {}
    return canonical_digest(
        {
            "schema_version": "pending_planning_v1",
            "planning_task_id": payload.get("planning_task_id"),
            "proposal_id": payload.get("pending_proposal_id"),
            "proposal_fingerprint": metadata.get("proposal_fingerprint"),
        }
    )


class PlanningEnvelopeStore:
    def __init__(self, root: str | Path, *, tenant_id: str = "default") -> None:
        require_safe_path_segment(tenant_id, "planning envelope tenant_id")
        shared = Path(root)
        self.artifact_root = shared if tenant_id == "default" else shared / "tenants" / tenant_id
        self.root = self.artifact_root / "system" / "planning-envelopes"
        self.tenant_id = tenant_id

    def path(self, task_id: str) -> Path:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("planning envelope task_id is required")
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def anchor_path(self, task_id: str) -> Path:
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        return self.artifact_root / "system" / "planning-envelope-anchors" / f"{digest}.json"

    def create(
        self,
        context: PlanningContext,
        *,
        archive_uri: str,
        assume_locked: bool = False,
    ) -> dict[str, Any]:
        payload = self._payload(context, archive_uri=archive_uri)
        validate_planning_envelope_payload(payload, tenant_id=self.tenant_id, task_id=context.task_id)
        if assume_locked:
            return self._create_locked(context.task_id, payload)
        with self.task_lock(context.task_id):
            return self._create_locked(context.task_id, payload)

    def _create_locked(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self.path(task_id)
        if path.is_symlink():
            raise PlanningEnvelopeIntegrityError("planning envelope artifact path cannot be a symbolic link")
        if path.exists():
            existing = self.load_validated_payload(task_id)
            if existing.get("envelope_digest") != payload.get("envelope_digest"):
                raise PlanningEnvelopeIntegrityError("same task_id is already bound to another immutable proposal set")
            self._ensure_anchor(existing)
            return existing
        # Publish the independent identity anchor first.  A process death
        # between these two atomic writes leaves an explicit NOT_READY proof
        # instead of silently allowing the model to be called again.
        self._ensure_anchor(payload)
        atomic_create_json(path, payload, artifact_root=self.artifact_root)
        return self.load_payload(task_id)

    def load(self, task_id: str) -> PlanningContext | None:
        path = self.path(task_id)
        if path.is_symlink():
            raise PlanningEnvelopeIntegrityError("planning envelope artifact path cannot be a symbolic link")
        if not path.exists():
            anchor_path = self.anchor_path(task_id)
            if anchor_path.is_symlink():
                raise PlanningEnvelopeIntegrityError("planning envelope anchor path cannot be a symbolic link")
            if anchor_path.exists():
                raise PlanningEnvelopeIntegrityError(
                    "planning envelope is missing after its immutable identity anchor was published"
                )
            return None
        return self._context(self.load_validated_payload(task_id))

    def load_payload(self, task_id: str) -> dict[str, Any]:
        path = self.path(task_id)
        if path.is_symlink():
            raise PlanningEnvelopeIntegrityError("planning envelope artifact path cannot be a symbolic link")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlanningEnvelopeIntegrityError("planning envelope is unreadable") from exc
        return validate_planning_envelope_payload(payload, tenant_id=self.tenant_id, task_id=task_id)

    def load_validated_payload(self, task_id: str) -> dict[str, Any]:
        """Load one immutable envelope together with its independent identity anchor."""

        envelope = self.load_payload(task_id)
        anchor_path = self.anchor_path(task_id)
        if anchor_path.is_symlink() or not anchor_path.exists():
            raise PlanningEnvelopeIntegrityError("planning envelope immutable identity anchor is missing or invalid")
        anchor = self._load_anchor(anchor_path)
        if (
            anchor.get("envelope_digest") != envelope.get("envelope_digest")
            or anchor.get("planning_digest") != envelope.get("planning_digest")
            or anchor.get("proposal_set_digest") != envelope.get("proposal_set_digest")
        ):
            raise PlanningEnvelopeIntegrityError("planning envelope anchor disagrees with its envelope")
        return envelope

    def validate_all(self) -> dict[str, int]:
        """Fail startup closed if any immutable task envelope is unreadable or detached."""

        validated = 0
        envelope_task_ids: set[str] = set()
        for path in sorted(self.root.glob("*.json")) if self.root.exists() else ():
            if path.is_symlink():
                raise PlanningEnvelopeIntegrityError(
                    f"planning envelope artifact has an invalid task path: {path.name}"
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PlanningEnvelopeIntegrityError(f"planning envelope artifact is unreadable: {path.name}") from exc
            task_id = str(payload.get("task_id") or "") if isinstance(payload, dict) else ""
            if not task_id or path.is_symlink() or self.path(task_id).resolve() != path.resolve():
                raise PlanningEnvelopeIntegrityError(
                    f"planning envelope artifact has an invalid task path: {path.name}"
                )
            envelope = self.load_validated_payload(task_id)
            anchor_path = self.anchor_path(task_id)
            anchor = self._load_anchor(anchor_path)
            if (
                anchor.get("envelope_digest") != envelope.get("envelope_digest")
                or anchor.get("planning_digest") != envelope.get("planning_digest")
                or anchor.get("proposal_set_digest") != envelope.get("proposal_set_digest")
            ):
                raise PlanningEnvelopeIntegrityError("planning envelope anchor disagrees with its envelope")
            envelope_task_ids.add(task_id)
            validated += 1
        anchors = 0
        anchor_task_ids: set[str] = set()
        anchor_root = self.artifact_root / "system" / "planning-envelope-anchors"
        for path in sorted(anchor_root.glob("*.json")) if anchor_root.exists() else ():
            anchor = self._load_anchor(path)
            task_id = str(anchor["task_id"])
            expected_path = self.anchor_path(task_id)
            if path.is_symlink() or expected_path.resolve() != path.resolve():
                raise PlanningEnvelopeIntegrityError("planning envelope anchor has an invalid task path")
            try:
                envelope = self.load_payload(task_id)
            except PlanningEnvelopeIntegrityError:
                raise
            except FileNotFoundError as exc:
                raise PlanningEnvelopeIntegrityError(
                    f"planning envelope referenced by anchor is missing: {task_id}"
                ) from exc
            if (
                anchor.get("envelope_digest") != envelope.get("envelope_digest")
                or anchor.get("planning_digest") != envelope.get("planning_digest")
                or anchor.get("proposal_set_digest") != envelope.get("proposal_set_digest")
            ):
                raise PlanningEnvelopeIntegrityError("planning envelope anchor disagrees with its envelope")
            anchor_task_ids.add(task_id)
            anchors += 1
        if anchor_task_ids != envelope_task_ids:
            raise PlanningEnvelopeIntegrityError("planning envelope and immutable anchor sets disagree")
        return {"validated": validated, "anchors": anchors}

    def iter_payloads(self) -> tuple[dict[str, Any], ...]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")) if self.root.exists() else ():
            if path.is_symlink():
                raise PlanningEnvelopeIntegrityError("planning envelope artifact has an invalid task path")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PlanningEnvelopeIntegrityError("planning envelope artifact is unreadable") from exc
            task_id = str(raw.get("task_id") or "") if isinstance(raw, dict) else ""
            if not task_id or path.is_symlink() or path.resolve() != self.path(task_id).resolve():
                raise PlanningEnvelopeIntegrityError("planning envelope artifact has an invalid task path")
            payloads.append(self.load_payload(task_id))
        return tuple(payloads)

    def _ensure_anchor(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = str(payload["task_id"])
        core = {
            "schema_version": PLANNING_ENVELOPE_ANCHOR_SCHEMA_VERSION,
            "task_id": task_id,
            "tenant_id": self.tenant_id,
            "envelope_path": str(self.path(task_id).relative_to(self.artifact_root)),
            "envelope_digest": str(payload["envelope_digest"]),
            "planning_digest": str(payload["planning_digest"]),
            "proposal_set_digest": str(payload["proposal_set_digest"]),
        }
        anchor = {**core, "anchor_digest": canonical_digest(core)}
        path = self.anchor_path(task_id)
        if path.is_symlink():
            raise PlanningEnvelopeIntegrityError("planning envelope anchor path cannot be a symbolic link")
        if path.exists():
            existing = self._load_anchor(path)
            if existing != anchor:
                raise PlanningEnvelopeIntegrityError("planning envelope anchor conflicts with immutable envelope")
            return existing
        atomic_create_json(path, anchor, artifact_root=self.artifact_root)
        return self._load_anchor(path)

    def _load_anchor(self, path: Path) -> dict[str, Any]:
        if path.is_symlink():
            raise PlanningEnvelopeIntegrityError("planning envelope anchor path cannot be a symbolic link")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlanningEnvelopeIntegrityError("planning envelope anchor is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != PLANNING_ENVELOPE_ANCHOR_SCHEMA_VERSION:
            raise PlanningEnvelopeIntegrityError("planning envelope anchor schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "anchor_digest"}
        if payload.get("anchor_digest") != canonical_digest(core) or payload.get("tenant_id") != self.tenant_id:
            raise PlanningEnvelopeIntegrityError("planning envelope anchor integrity check failed")
        task_id = str(payload.get("task_id") or "")
        expected_anchor = self.anchor_path(task_id) if task_id else None
        expected_envelope = self.path(task_id).relative_to(self.artifact_root).as_posix() if task_id else ""
        if (
            expected_anchor is None
            or path.resolve() != expected_anchor.resolve()
            or payload.get("envelope_path") != expected_envelope
        ):
            raise PlanningEnvelopeIntegrityError("planning envelope anchor is detached from its unique envelope path")
        return payload

    @staticmethod
    def planning_digest(payload_without_envelope_digest: dict[str, Any]) -> str:
        core = {key: value for key, value in payload_without_envelope_digest.items() if key not in {"planning_digest"}}
        return canonical_digest(core)

    def _payload(self, context: PlanningContext, *, archive_uri: str) -> dict[str, Any]:
        proposal_inputs = [
            {
                "proposal": item.proposal.to_dict(),
                "retrieval_views": list(item.retrieval_views),
                "forced_pending_reason": item.forced_pending_reason,
            }
            for item in context.proposal_inputs
        ]
        proposal_set_digest = canonical_digest(proposal_inputs)
        outcome_decisions = {item.proposal_id: item.decision for item in context.proposal_outcomes}
        core: dict[str, Any] = {
            "schema_version": PLANNING_ENVELOPE_SCHEMA_VERSION,
            "task_id": context.task_id,
            "commit_group_id": context.operation_group_identity,
            "archive_uri": archive_uri,
            "archive_digest": context.archive_digest,
            "manifest_digest": context.manifest_digest,
            "episode_id": context.episode_id,
            "session_id": context.session_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "planning_id": context.planning_id,
            "extractor_version": context.extractor_version,
            "model_id": context.model_id,
            "prompt_version": context.prompt_version,
            "semantic_contract_version": context.semantic_contract_version,
            "proposal_inputs": proposal_inputs,
            "normalized_accepted_proposals": [
                item.proposal.to_dict()
                for item in context.proposal_inputs
                if outcome_decisions.get(item.proposal.proposal_id) == "ACCEPT_FOR_RECONCILE"
            ],
            "pending_proposal_inputs": [
                {"proposal": item.proposal.to_dict(), "reason": item.forced_pending_reason}
                for item in context.proposal_inputs
                if item.forced_pending_reason
            ],
            "candidate_outcomes": [asdict(item) for item in context.proposal_outcomes],
            "proposal_fingerprints": [item.proposal.fingerprint for item in context.proposal_inputs],
            "proposal_set_digest": proposal_set_digest,
            "prefetch_snapshot": [asdict(item) for item in context.prefetch_snapshot],
            "prefetch_revisions": [list(item) for item in context.planned_against_revisions],
            "staged_objects": [asdict(item) for item in context.staged_objects],
            "scope_candidates": list(context.scope_candidates),
            "evidence_references": [item.to_dict() for item in context.evidence_references],
            "operation_group_identity": context.operation_group_identity,
            "admission_summary": [list(item) for item in context.admission_summary],
            "security_flags": list(context.extraction_security_flags),
            "salience_decision": {
                "episode_fingerprint": context.salience_fingerprint,
                "reasons": list(context.salience_reasons),
                "score": context.salience_score,
                "budget_cost": context.salience_budget_cost,
                "duplicate": context.salience_duplicate,
                "privacy_risk": context.salience_privacy_risk,
                "factors": [
                    {"name": name, "weight": weight, "event_ids": list(event_ids)}
                    for name, weight, event_ids in context.salience_factors
                ],
            },
            "salience_reservation_digest": context.salience_reservation_digest,
            "egress_decision": context.egress_decision,
            "egress_audit": dict(context.egress_audit),
            "created_at": context.created_at,
        }
        core["planning_digest"] = self.planning_digest(core)
        return {**core, "envelope_digest": canonical_digest(core)}

    def _context(self, payload: dict[str, Any]) -> PlanningContext:
        salience = dict(payload.get("salience_decision", {}) or {})
        return PlanningContext(
            planning_id=str(payload["planning_id"]),
            task_id=str(payload["task_id"]),
            archive_digest=str(payload.get("archive_digest") or ""),
            manifest_digest=str(payload.get("manifest_digest") or ""),
            episode_id=str(payload["episode_id"]),
            session_id=str(payload["session_id"]),
            tenant_id=str(payload["tenant_id"]),
            proposal_inputs=tuple(
                ProposalPlanningInput(
                    MemorySemanticProposal.from_dict(dict(item["proposal"])),
                    tuple(str(value) for value in item.get("retrieval_views", []) or []),
                    str(item.get("forced_pending_reason") or ""),
                )
                for item in payload.get("proposal_inputs", [])
            ),
            prefetch_snapshot=tuple(
                PrefetchSnapshot(
                    uri=str(item["uri"]),
                    revision=int(item["revision"]),
                    object_digest=str(item["object_digest"]),
                    content_digest=str(item["content_digest"]),
                    relation_digest=str(item["relation_digest"]),
                )
                for item in payload.get("prefetch_snapshot", [])
            ),
            planned_against_revisions=tuple(
                (str(item[0]), int(item[1])) for item in payload.get("prefetch_revisions", [])
            ),
            staged_objects=tuple(
                StagedObjectSnapshot(
                    uri=str(item["uri"]),
                    revision=int(item["revision"]),
                    object_digest=str(item["object_digest"]),
                )
                for item in payload.get("staged_objects", [])
            ),
            scope_candidates=tuple(str(item) for item in payload.get("scope_candidates", []) or []),
            evidence_references=tuple(EvidenceRef(**dict(item)) for item in payload.get("evidence_references", [])),
            operation_group_identity=str(payload["operation_group_identity"]),
            admission_summary=tuple((str(item[0]), int(item[1])) for item in payload.get("admission_summary", [])),
            proposal_outcomes=tuple(
                ProposalPlanningOutcome(
                    proposal_id=str(item["proposal_id"]),
                    decision=str(item["decision"]),
                    reason=str(item["reason"]),
                    candidate_index=(int(item["candidate_index"]) if item.get("candidate_index") is not None else None),
                    security_flags=tuple(str(value) for value in item.get("security_flags", []) or []),
                )
                for item in payload.get("candidate_outcomes", [])
            ),
            extraction_security_flags=tuple(str(item) for item in payload.get("security_flags", []) or []),
            salience_fingerprint=str(salience.get("episode_fingerprint") or ""),
            salience_reasons=tuple(str(item) for item in salience.get("reasons", []) or []),
            salience_score=int(salience.get("score", 0) or 0),
            salience_budget_cost=int(salience.get("budget_cost", 0) or 0),
            salience_duplicate=bool(salience.get("duplicate", False)),
            salience_privacy_risk=bool(salience.get("privacy_risk", False)),
            salience_reservation_digest=str(payload.get("salience_reservation_digest") or ""),
            salience_factors=tuple(
                (
                    str(item.get("name") or ""),
                    int(item.get("weight", 0) or 0),
                    tuple(str(value) for value in item.get("event_ids", []) or []),
                )
                for item in salience.get("factors", []) or []
                if isinstance(item, dict)
            ),
            proposal_set_digest=str(payload["proposal_set_digest"]),
            planning_digest=str(payload["planning_digest"]),
            egress_decision=str(payload.get("egress_decision") or "LOCAL_ONLY"),
            egress_audit=tuple(
                sorted((str(key), str(value)) for key, value in dict(payload.get("egress_audit", {}) or {}).items())
            ),
            user_id=str(payload["user_id"]),
            extractor_version=str(payload["extractor_version"]),
            model_id=str(payload.get("model_id") or ""),
            prompt_version=str(payload.get("prompt_version") or ""),
            semantic_contract_version=str(payload.get("semantic_contract_version") or ""),
            created_at=str(payload.get("created_at") or ""),
        )

    @contextmanager
    def task_lock(self, task_id: str) -> Iterator[None]:
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        lock_path = self.root / ".locks" / f"{digest}.lock"
        descriptor = open_private_lock(lock_path, root=self.artifact_root)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
