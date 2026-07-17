"""Proofs responsibilities for canonical projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.catalog import (
    CatalogRecord,
    CatalogRecordKind,
)
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.queue_store import (
    QueueJob,
)
from memoryos.contextdb.store.vector import vector_row_id
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest
from memoryos.memory.canonical.current_head import (
    CurrentHeadIntegrityError,
    artifact_root_for,
    head_from_receipt_snapshot,
    iter_current_head_uris,
    load_current_head,
)
from memoryos.memory.canonical.projection_proof import (
    PROJECTION_COMPLETION_PROOF_SCHEMA_VERSION,
    PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    AuthoritativeProjectionIntegrityError,
    projection_publication_record_digest,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionStatus,
    ProjectionStepStatus,
)
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    materialized_current_revision_payload,
    revision_payload_with_effective_validity,
)
from memoryos.memory.canonical.visibility import (
    CommittedCanonicalRead,
    CommittedStateIntegrityError,
    read_committed_canonical,
)
from memoryos.operations.commit.outbox_envelope import (
    prepared_intent_digest,
)
from memoryos.operations.commit.receipt import (
    ReceiptIntegrityError,
    load_transaction_receipt,
    receipt_snapshot,
)

from .models import (
    ProjectionOutboxIntegrityError,
)

if TYPE_CHECKING:
    from .worker import MemoryProjectionWorker


def verify_current_projections(self: MemoryProjectionWorker) -> dict[str, int]:
    artifact_root = artifact_root_for(self.projector.source_store)
    claim_uris = tuple(iter_current_head_uris(artifact_root, kinds=("claim",)) if artifact_root is not None else ())
    current_records = {record.claim_uri: record for record in self.projector.record_store.iter_current()}
    if set(current_records) != set(claim_uris):
        dangling = sorted(set(current_records) - set(claim_uris))
        missing = sorted(set(claim_uris) - set(current_records))
        raise ProjectionIntegrityError(
            f"projection current/head closure mismatch; dangling={dangling}; missing={missing}"
        )
    verified = 0
    for claim_uri in claim_uris:
        committed = read_committed_canonical(
            self.projector.source_store,
            claim_uri,
            self.projector.relation_store,
        )
        revision = int(dict(committed.object.metadata or {}).get("revision", 0))
        if current_records[claim_uri].source_revision != revision:
            raise ProjectionIntegrityError(
                f"projection current revision does not match committed Claim head: {claim_uri}"
            )
        self._verify_claim_projection(claim_uri, revision)
        verified += 1
    return {"verified": verified}


def validate_projection_proofs(self: MemoryProjectionWorker) -> dict[str, int]:
    """Reverse-bind every proof artifact, including crash-orphaned receipts."""

    structural = self.proof_store.validate_all()
    verified = 0
    for publication in self.proof_store.iter_publications():
        transaction_id = str(publication["transaction_id"])
        group_id = str(publication["commit_group_id"])
        result = self.verify_commit_group_completion(group_id, (transaction_id,))
        failures = [str(item) for item in result["failures"]]
        if failures:
            raise ProjectionIntegrityError(
                f"projection publication has no durable completion: {transaction_id}: {failures}"
            )
        if len(result["proofs"]) != 1:
            raise ProjectionIntegrityError("projection transaction has an invalid completion proof count")
        verified += 1
    final = self.proof_store.validate_all()
    return {
        "publications": final["publications"],
        "completions": final["completions"],
        "verified": verified,
        "completed_during_validation": final["completions"] - structural["completions"],
    }


def migrate_legacy_completion_proof(
    self: MemoryProjectionWorker,
    group_id: str,
    transaction_id: str,
    legacy_proof: dict[str, Any],
) -> bool:
    """Promote a validated v1 group result into the create-only proof DAG.

    The v1 result was emitted only after the old verifier had checked the
    live index/vector/views, but it did not persist their metadata digests.
    Migration binds that attestation digest to the immutable receipt and
    all still-present revision-scoped projection artifacts.  It never
    invents proof when the legacy result or historical artifacts are
    missing.
    """

    if self.proof_store.load_publication(transaction_id) is not None:
        return False
    if not isinstance(legacy_proof, dict) or legacy_proof.get("schema_version") != "projection_completion_proof_v1":
        raise ProjectionIntegrityError("legacy projection completion proof schema is unsupported")
    legacy_core = {key: value for key, value in legacy_proof.items() if key != "proof_digest"}
    legacy_digest = str(legacy_proof.get("proof_digest") or "")
    if legacy_digest != canonical_digest(legacy_core):
        raise ProjectionIntegrityError("legacy projection completion proof digest is corrupt")
    job_id = f"outbox_{transaction_id}"
    job = self.queue_store.get(job_id)
    if job is None or job.status != "done":
        raise ProjectionIntegrityError("legacy projection completion queue state is not durable")
    outbox = self._load_projection_job_outbox(job, expected_transaction_id=transaction_id)
    receipt = self._load_bound_receipt(outbox, transaction_id, group_id)
    if (
        legacy_proof.get("commit_group_id") != group_id
        or legacy_proof.get("transaction_id") != transaction_id
        or legacy_proof.get("job_id") != job_id
        or legacy_proof.get("queue_status") != "done"
        or legacy_proof.get("outbox_digest") != outbox.get("outbox_digest")
        or legacy_proof.get("receipt_digest") != receipt.get("receipt_digest")
    ):
        raise ProjectionIntegrityError("legacy projection completion proof crosses its transaction boundary")
    raw_claims = legacy_proof.get("claims")
    if not isinstance(raw_claims, list):
        raise ProjectionIntegrityError("legacy projection completion proof has no Claim set")
    legacy_by_identity = {
        (str(item.get("claim_uri") or ""), int(item.get("source_revision", 0))): item
        for item in raw_claims
        if isinstance(item, dict)
    }
    expected_identities = {(str(item["uri"]), int(item["revision"])) for item in self._claim_revisions(outbox)}
    if set(legacy_by_identity) != expected_identities:
        raise ProjectionIntegrityError("legacy projection completion Claim set differs from outbox")
    claims = [
        self._migrated_legacy_claim_proof(
            legacy_by_identity[identity],
            receipt,
            legacy_digest,
        )
        for identity in sorted(expected_identities)
    ]
    publication_core = {
        "schema_version": PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        "commit_group_id": group_id,
        "transaction_id": transaction_id,
        "job_id": job_id,
        "tenant_id": str(receipt["tenant_id"]),
        "user_id": str(receipt["user_id"]),
        "queue_identity_digest": self._queue_identity_digest(job),
        "outbox_digest": str(outbox["outbox_digest"]),
        "receipt_digest": str(receipt["receipt_digest"]),
        "prepared_intent_digest": str(receipt["prepared_intent_digest"]),
        "operation_ids": [str(item) for item in outbox["operation_ids"]],
        "claim_revisions": [{"uri": item["claim_uri"], "revision": item["source_revision"]} for item in claims],
        "claims": claims,
        "migration_source_schema": "projection_completion_proof_v1",
        "legacy_completion_proof_digest": legacy_digest,
    }
    publication = self.proof_store.ensure_publication(
        {
            **publication_core,
            "publication_digest": canonical_digest(publication_core),
        }
    )
    self._verify_projection_publication(publication, outbox, receipt, job)
    return True


def _migrated_legacy_claim_proof(
    self: MemoryProjectionWorker,
    legacy: dict[str, Any],
    receipt: dict[str, Any],
    legacy_digest: str,
) -> dict[str, Any]:
    claim_uri = str(legacy.get("claim_uri") or "")
    source_revision = int(legacy.get("source_revision", 0))
    attempt_id = str(legacy.get("projection_attempt_id") or "")
    record = self.projector.record_store.load(
        claim_uri,
        source_revision,
        projection_attempt_id=attempt_id,
    )
    if record is None or record.status not in {
        ProjectionStatus.COMPLETED.value,
        ProjectionStatus.STALE.value,
    }:
        raise ProjectionIntegrityError("legacy projection attempt record is missing")
    for key, expected in (
        ("claim_uri", record.claim_uri),
        ("source_revision", record.source_revision),
        ("projection_revision", record.projection_revision),
        ("projection_attempt_id", record.projection_attempt_id),
        ("input_effect_hash", record.input_effect_hash),
        ("publish_token", record.publish_token),
        ("projected_content_digest", record.projected_content_digest),
        ("projected_relation_digest", record.projected_relation_digest),
    ):
        if legacy.get(key) != expected:
            raise ProjectionIntegrityError("legacy projection Claim proof differs from attempt record")
    legacy_record_digest = str(legacy.get("record_digest") or "")
    if len(legacy_record_digest) != 64:
        raise ProjectionIntegrityError("legacy projection record digest is invalid")
    try:
        snapshot = receipt_snapshot(receipt, claim_uri)
        obj = ContextObject.from_dict(dict(snapshot["object"]))
        metadata = dict(obj.metadata or {})
        materialized = materialized_current_revision_payload(metadata)
    except (KeyError, TypeError, ValueError, ReceiptIntegrityError) as exc:
        raise ProjectionIntegrityError("legacy projection has no immutable Source snapshot") from exc
    domain_identity = {
        "claim_uri": claim_uri,
        "tenant_id": str(obj.tenant_id or "default"),
        "owner_user_id": str(obj.owner_user_id or ""),
        "canonical_kind": "claim",
        "claim_state": str(materialized.get("state") or ""),
        "canonical_head_digest": str(head_from_receipt_snapshot(snapshot, receipt)["head_digest"]),
        "current_transaction_id": str(receipt["transaction_id"]),
        "current_receipt_digest": str(receipt["receipt_digest"]),
        "current_claim_revision": int(materialized["revision"]),
    }
    historical_committed = CommittedCanonicalRead(obj, receipt=receipt)
    if (
        int(metadata.get("revision", 0)) != source_revision
        or self.projector._input_effect_hash(historical_committed, source_revision) != record.input_effect_hash
    ):
        raise ProjectionIntegrityError("legacy projection input effect differs from receipt")
    layer_values = {
        "L0": self.projector.source_store.read_content(record.l0_uri),
        "L1": self.projector.source_store.read_content(record.l1_uri),
        "L2": self.projector.source_store.read_content(record.l2_uri),
    }
    try:
        relation_payload = json.loads(self.projector.source_store.read_content(record.relations_uri))
        manifest = json.loads(self.projector.source_store.read_content(record.manifest_uri))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProjectionIntegrityError("legacy projection artifact is malformed") from exc
    if not isinstance(relation_payload, dict) or not isinstance(manifest, dict):
        raise ProjectionIntegrityError("legacy projection artifact is not an object")
    if record.projected_content_digest != canonical_digest(
        layer_values
    ) or record.projected_relation_digest != canonical_digest(relation_payload.get("relations", [])):
        raise ProjectionIntegrityError("legacy projection artifact digest is corrupt")
    self._assert_projection_identity(
        relation_payload,
        record,
        label="legacy relation",
        domain_identity=domain_identity,
    )
    self._assert_projection_identity(
        manifest,
        record,
        label="legacy manifest",
        domain_identity=domain_identity,
    )
    attested = lambda component: canonical_digest(  # noqa: E731
        {
            "schema_version": "legacy_projection_component_attestation_v1",
            "legacy_completion_proof_digest": legacy_digest,
            "component": component,
        }
    )
    claim_core = {
        "claim_uri": record.claim_uri,
        "source_revision": record.source_revision,
        "projection_revision": record.projection_revision,
        "projection_attempt_id": record.projection_attempt_id,
        "input_effect_hash": record.input_effect_hash,
        "publish_token": record.publish_token,
        "projected_content_digest": record.projected_content_digest,
        "projected_relation_digest": record.projected_relation_digest,
        "record_digest": legacy_record_digest,
        "publication_record_digest": projection_publication_record_digest(record),
        "layer_uris": {
            "L0": record.l0_uri,
            "L1": record.l1_uri,
            "L2": record.l2_uri,
            "manifest": record.manifest_uri,
            "relations": record.relations_uri,
        },
        "layer_digests": {name: canonical_digest(value) for name, value in layer_values.items()},
        "relation_artifact_digest": canonical_digest(relation_payload),
        "manifest_digest": canonical_digest(manifest),
        "index_metadata_digest": attested("index"),
        "vector_metadata_digest": attested("vector"),
        "scope_view_digests": [attested("scope")],
        "taxonomy_view_digests": [attested("taxonomy")],
        "domain_identity": domain_identity,
        "migration_source_schema": "projection_completion_proof_v1",
        "legacy_completion_proof_digest": legacy_digest,
    }
    return {**claim_core, "claim_proof_digest": canonical_digest(claim_core)}


def _verify_projection_completion(
    self: MemoryProjectionWorker,
    group_id: str,
    transaction_ids: tuple[str, ...],
) -> list[str]:
    """Prove durable queue and every derived publication before completion."""

    return self.verify_commit_group_completion(group_id, transaction_ids)["failures"]


def verify_commit_group_completion(
    self: MemoryProjectionWorker,
    group_id: str,
    transaction_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Return immutable publication-bound proofs for one commit group.

    A current transaction is checked against every live derived row.  Once
    its Claim revision advances, the create-only publication receipt and
    revision-scoped projection artifacts become the historical proof;
    current index/vector/view pointers are then validated by the newer
    transaction instead of being incorrectly compared with the old one.
    """

    failures: list[str] = []
    proofs: list[dict[str, Any]] = []
    for transaction_id in transaction_ids:
        job_id = f"outbox_{transaction_id}"
        job = self.queue_store.get(job_id)
        if job is None:
            failures.append(f"{job_id}:missing_job")
            continue
        if job.status != "done":
            failures.append(f"{job_id}:queue_{job.status}")
            continue
        try:
            outbox = self._load_projection_job_outbox(
                job,
                expected_transaction_id=transaction_id,
            )
            if (
                str(outbox.get("transaction_id") or "") != transaction_id
                or str(outbox.get("commit_group_id") or "") != group_id
            ):
                raise ProjectionIntegrityError("projection outbox crosses commit group")
            receipt = self._load_bound_receipt(outbox, transaction_id, group_id)
            publication = self.proof_store.load_publication(transaction_id)
            if publication is None:
                # Compatibility/recovery path for a job ACKed by an older
                # process or a crash between ACK and completion proof.  It
                # remains safe only while every projected revision can
                # still pass the strict current-state verifier.
                publication = self._ensure_projection_publication(outbox, job)
            self._verify_projection_publication(publication, outbox, receipt, job)
            proof_core = {
                "schema_version": PROJECTION_COMPLETION_PROOF_SCHEMA_VERSION,
                "commit_group_id": group_id,
                "transaction_id": transaction_id,
                "job_id": job_id,
                "tenant_id": str(receipt["tenant_id"]),
                "user_id": str(receipt["user_id"]),
                "queue_status": job.status,
                "queue_identity_digest": self._queue_identity_digest(job),
                "outbox_digest": str(outbox["outbox_digest"]),
                "receipt_digest": str(receipt["receipt_digest"]),
                "prepared_intent_digest": str(receipt["prepared_intent_digest"]),
                "operation_ids": [str(item) for item in outbox["operation_ids"]],
                "claim_revisions": list(publication["claim_revisions"]),
                "claims": list(publication["claims"]),
                "publication_digest": str(publication["publication_digest"]),
            }
            completion = self.proof_store.ensure_completion(
                {**proof_core, "proof_digest": canonical_digest(proof_core)}
            )
            proofs.append(completion)
        except AuthoritativeProjectionIntegrityError as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact="projection_proof",
                identifiers={"job_id": job_id, "commit_group_id": group_id},
            )
            failures.append(f"{job_id}:ProjectionIntegrityError")
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            ProjectionIntegrityError,
            ProjectionOutboxIntegrityError,
            CommittedStateIntegrityError,
            CurrentHeadIntegrityError,
            ReceiptIntegrityError,
        ) as exc:
            failures.append(f"{job_id}:{type(exc).__name__}")
    return {"failures": failures, "proofs": proofs}


def _ensure_projection_publication(
    self: MemoryProjectionWorker,
    outbox: dict[str, Any],
    job: QueueJob,
) -> dict[str, Any]:
    transaction_id = str(outbox.get("transaction_id") or "")
    group_id = str(outbox.get("commit_group_id") or "")
    receipt = self._load_bound_receipt(outbox, transaction_id, group_id)
    existing = self.proof_store.load_publication(transaction_id)
    if existing is not None:
        self._verify_projection_publication(existing, outbox, receipt, job)
        return existing
    claim_proofs: list[dict[str, Any]] = []
    for item in self._claim_revisions(outbox):
        claim_uri = str(item["uri"])
        source_revision = int(item["revision"])
        head, _current_receipt, _current_snapshot = load_current_head(
            self.projector.root,
            claim_uri,
            canonical_kind="claim",
        )
        current_revision = int(head.get("current_revision", 0))
        if current_revision == source_revision:
            claim_proofs.append(self._verify_claim_projection(claim_uri, source_revision))
        elif current_revision > source_revision:
            claim_proofs.append(
                self._materialize_historical_claim_projection(
                    receipt,
                    claim_uri,
                    source_revision,
                )
            )
        else:
            raise ProjectionIntegrityError("projection outbox refers to a future Claim revision")
    publication_core = {
        "schema_version": PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        "commit_group_id": group_id,
        "transaction_id": transaction_id,
        "job_id": job.job_id,
        "tenant_id": str(receipt["tenant_id"]),
        "user_id": str(receipt["user_id"]),
        "queue_identity_digest": self._queue_identity_digest(job),
        "outbox_digest": str(outbox["outbox_digest"]),
        "receipt_digest": str(receipt["receipt_digest"]),
        "prepared_intent_digest": str(receipt["prepared_intent_digest"]),
        "operation_ids": [str(item) for item in outbox["operation_ids"]],
        "claim_revisions": [{"uri": item["claim_uri"], "revision": item["source_revision"]} for item in claim_proofs],
        "claims": claim_proofs,
    }
    return self.proof_store.ensure_publication(
        {
            **publication_core,
            "publication_digest": canonical_digest(publication_core),
        }
    )


def _verify_projection_publication(
    self: MemoryProjectionWorker,
    publication: dict[str, Any],
    outbox: dict[str, Any],
    receipt: dict[str, Any],
    job: QueueJob,
) -> None:
    self._verify_projection_publication_boundary(publication, outbox, receipt, job)
    claim_proofs = publication.get("claims")
    assert isinstance(claim_proofs, list)
    by_identity = {
        (str(item.get("claim_uri") or ""), int(item.get("source_revision", 0))): item
        for item in claim_proofs
        if isinstance(item, dict)
    }
    expected_identities = {(str(item["uri"]), int(item["revision"])) for item in self._claim_revisions(outbox)}
    for claim_uri, source_revision in sorted(expected_identities):
        claim_proof = by_identity[(claim_uri, source_revision)]
        head, _current_receipt, _current_snapshot = load_current_head(
            self.projector.root,
            claim_uri,
            canonical_kind="claim",
        )
        current_revision = int(head.get("current_revision", 0))
        if current_revision == source_revision:
            # Rebuild may legitimately publish a new attempt for the same
            # committed Source effect.  Verify that mutable current attempt
            # in full, then independently verify the original immutable
            # publication instead of requiring their attempt ids to match.
            self._verify_claim_projection(claim_uri, source_revision)
            self._verify_historical_claim_projection(claim_proof, receipt)
        elif current_revision > source_revision:
            self._verify_historical_claim_projection(claim_proof, receipt)
        else:
            raise ProjectionIntegrityError("projection publication refers to a future Claim revision")


def _verify_projection_publication_boundary(
    self: MemoryProjectionWorker,
    publication: dict[str, Any],
    outbox: dict[str, Any],
    receipt: dict[str, Any],
    job: QueueJob,
) -> None:
    transaction_id = str(outbox["transaction_id"])
    group_id = str(outbox["commit_group_id"])
    expected_boundary = {
        "commit_group_id": group_id,
        "transaction_id": transaction_id,
        "job_id": job.job_id,
        "tenant_id": str(receipt["tenant_id"]),
        "user_id": str(receipt["user_id"]),
        "queue_identity_digest": self._queue_identity_digest(job),
        "outbox_digest": str(outbox["outbox_digest"]),
        "receipt_digest": str(receipt["receipt_digest"]),
        "prepared_intent_digest": str(receipt["prepared_intent_digest"]),
        "operation_ids": [str(item) for item in outbox["operation_ids"]],
        "claim_revisions": self._claim_revisions(outbox),
    }
    actual_boundary = {
        **{key: publication.get(key) for key in expected_boundary if key != "claim_revisions"},
        "claim_revisions": [
            {"uri": str(item.get("uri") or ""), "revision": int(item["revision"])}
            for item in publication.get("claim_revisions", [])
            if isinstance(item, dict) and item.get("revision") is not None
        ],
    }
    if actual_boundary != expected_boundary:
        raise AuthoritativeProjectionIntegrityError("projection publication receipt crosses its transaction boundary")
    claim_proofs = publication.get("claims")
    if not isinstance(claim_proofs, list):
        raise AuthoritativeProjectionIntegrityError("projection publication receipt has no Claim proofs")
    by_identity = {
        (str(item.get("claim_uri") or ""), int(item.get("source_revision", 0))): item
        for item in claim_proofs
        if isinstance(item, dict)
    }
    expected_identities = {(str(item["uri"]), int(item["revision"])) for item in self._claim_revisions(outbox)}
    if set(by_identity) != expected_identities:
        raise AuthoritativeProjectionIntegrityError("projection publication Claim proof set differs from outbox")


def _load_bound_receipt(
    self: MemoryProjectionWorker,
    outbox: dict[str, Any],
    transaction_id: str,
    group_id: str,
) -> dict[str, Any]:
    raw_relative = str(outbox.get("receipt_path") or "")
    try:
        idempotency_key = require_safe_path_segment(
            outbox.get("idempotency_key"),
            "projection outbox idempotency_key",
        )
    except (TypeError, ValueError) as exc:
        raise AuthoritativeProjectionIntegrityError(
            "projection outbox does not identify its unique immutable receipt"
        ) from exc
    normalized_relative = Path(raw_relative)
    expected_relative = Path("system") / "transactions" / f"{idempotency_key}.json"
    candidate = self.projector.root / normalized_relative
    if (
        normalized_relative.is_absolute()
        or normalized_relative.as_posix() != expected_relative.as_posix()
        or candidate.is_symlink()
    ):
        raise AuthoritativeProjectionIntegrityError("projection outbox does not reference its unique immutable receipt")
    try:
        receipt_path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AuthoritativeProjectionIntegrityError(
            "projection outbox immutable receipt is missing or unreadable"
        ) from exc
    root = self.projector.root.resolve()
    if receipt_path == root or root not in receipt_path.parents:
        raise AuthoritativeProjectionIntegrityError("projection receipt path escapes tenant root")
    try:
        receipt = load_transaction_receipt(receipt_path)
    except (OSError, ReceiptIntegrityError) as exc:
        raise AuthoritativeProjectionIntegrityError("projection immutable receipt is corrupt") from exc
    if (
        receipt.get("receipt_digest") != outbox.get("receipt_digest")
        or receipt.get("transaction_id") != transaction_id
        or receipt.get("commit_group_id") != group_id
        or receipt.get("prepared_intent_digest") != prepared_intent_digest(outbox)
    ):
        raise AuthoritativeProjectionIntegrityError("projection receipt does not bind its outbox")
    return receipt


def _queue_identity_digest(job: QueueJob) -> str:
    return canonical_digest(
        {
            "job_id": job.job_id,
            "queue_name": job.queue_name,
            "action": job.action,
            "target_uri": job.target_uri,
            "payload": job.payload,
        }
    )


def _claim_revisions(outbox: dict[str, Any]) -> list[dict[str, Any]]:
    revisions: list[dict[str, Any]] = []
    for item in outbox.get("claim_revisions", []) or []:
        if not isinstance(item, dict) or not item.get("uri") or item.get("revision") is None:
            raise ProjectionIntegrityError("projection claim revision is invalid")
        revisions.append({"uri": str(item["uri"]), "revision": int(item["revision"])})
    return sorted(revisions, key=lambda item: (item["uri"], item["revision"]))


def _verify_claim_projection(self: MemoryProjectionWorker, claim_uri: str, source_revision: int) -> dict[str, Any]:
    if not claim_uri:
        raise ProjectionIntegrityError("projection claim URI is missing")
    committed = read_committed_canonical(
        self.projector.source_store,
        claim_uri,
        self.projector.relation_store,
    )
    metadata = dict(committed.object.metadata or {})
    if committed.from_before_image or int(metadata.get("revision", 0)) != source_revision:
        raise ProjectionIntegrityError("projection source revision is not current committed state")
    record = self.projector.record_store.load_current(
        claim_uri,
        source_revision=source_revision,
    )
    if record is None or not record.usable:
        raise ProjectionIntegrityError("projection current record is missing or incomplete")
    try:
        materialized_current = materialized_current_revision_payload(metadata)
    except CanonicalMemoryInvariantError as exc:
        raise ProjectionIntegrityError("projection source domain state is invalid") from exc
    domain_identity = self.projector._projection_domain_identity(
        committed,
        materialized_current,
    )
    expected_effect = self.projector._input_effect_hash(committed, source_revision)
    if record.input_effect_hash != expected_effect or record.projection_revision != source_revision:
        raise ProjectionIntegrityError("projection input effect does not match committed Source")
    layer_values = {
        "L0": self.projector.source_store.read_content(record.l0_uri),
        "L1": self.projector.source_store.read_content(record.l1_uri),
        "L2": self.projector.source_store.read_content(record.l2_uri),
    }
    if record.projected_content_digest != canonical_digest(layer_values):
        raise ProjectionIntegrityError("projection layer content digest does not match record")
    relation_payload = json.loads(self.projector.source_store.read_content(record.relations_uri))
    if not isinstance(relation_payload, dict) or record.projected_relation_digest != canonical_digest(
        relation_payload.get("relations", [])
    ):
        raise ProjectionIntegrityError("projection relation artifact does not match record")
    self._assert_projection_identity(
        relation_payload,
        record,
        label="relation",
        domain_identity=domain_identity,
    )
    manifest = json.loads(self.projector.source_store.read_content(record.manifest_uri))
    self._assert_projection_identity(
        manifest,
        record,
        label="manifest",
        domain_identity=domain_identity,
    )
    get_catalog = getattr(self.projector.index_store, "get_catalog", None)
    if callable(get_catalog):
        # Unified Catalog rows are revision-scoped and may intentionally
        # differ from legacy materialized-current artifacts for a late
        # historical transaction.  Verify the exact record key rather
        # than asking the URI compatibility API to choose one row.
        catalog = get_catalog(
            self.projector._claim_catalog_record_key(metadata, source_revision),
            tenant_id=str(committed.object.tenant_id or "default"),
        )
        if not isinstance(catalog, CatalogRecord):
            raise ProjectionIntegrityError("projection Claim Revision Catalog row is missing")
        self._assert_projection_identity(
            dict(catalog.metadata),
            record,
            label="index",
            domain_identity=domain_identity,
        )
        revisions = self.projector._bounded_claim_revisions(metadata)
        requested_revision = revision_payload_with_effective_validity(
            revisions,
            source_revision,
        )
        expected_l0, expected_l1, expected_l2 = self.projector._sanitized_revision_layers(
            committed.object,
            metadata,
            requested_revision,
            source_revision,
        )
        expected_serving_digest = canonical_digest({"L0": expected_l0, "L1": expected_l1, "L2": expected_l2})
        expected_catalog_identity = {
            "record_key": self.projector._claim_catalog_record_key(metadata, source_revision),
            "uri": claim_uri,
            "tenant_id": str(committed.object.tenant_id or "default"),
            "owner_user_id": str(committed.object.owner_user_id or ""),
            "record_kind": CatalogRecordKind.CLAIM_REVISION.value,
            "source_revision": source_revision,
            "canonical_slot_id": str(metadata.get("slot_id") or ""),
            "canonical_claim_id": str(metadata.get("claim_id") or ""),
            "canonical_revision": source_revision,
            "canonical_state": str(requested_revision.get("state") or ""),
            "canonical_head_digest": str(domain_identity["canonical_head_digest"]),
            "receipt_digest": str(domain_identity["current_receipt_digest"]),
            "projection_effect_hash": record.input_effect_hash,
            "l0_text": expected_l0,
            "l1_text": expected_l1,
            "source_digest": expected_serving_digest,
        }
        for field, expected in expected_catalog_identity.items():
            if getattr(catalog, field) != expected:
                raise ProjectionIntegrityError(f"projection Claim Revision Catalog {field} mismatch")
        if self.projector.source_store.read_content(catalog.l2_uri) != expected_l2:
            raise ProjectionIntegrityError("projection Claim Revision Catalog L2 content mismatch")
        # This is the exact row attested in the immutable publication.
        # It intentionally mirrors the legacy metadata shape without a
        # URI-level row selection that can pick another revision.
        index_metadata = {
            **dict(catalog.metadata),
            "record_key": catalog.record_key,
            "tenant_id": catalog.tenant_id,
            "owner_user_id": catalog.owner_user_id,
            "context_type": catalog.context_type,
            "claim_state": str(catalog.metadata.get("claim_state") or ""),
            "slot_id": catalog.canonical_slot_id,
            "memory_type": str(catalog.metadata.get("memory_type") or ""),
            "index_content_digest": canonical_digest(catalog.l1_text),
        }
    else:
        legacy_index_metadata = self.projector.index_store.get_index_metadata(claim_uri)
        if legacy_index_metadata is None:
            raise ProjectionIntegrityError("projection index row is missing")
        index_metadata = legacy_index_metadata
        self._assert_projection_identity(
            index_metadata,
            record,
            label="index",
            domain_identity=domain_identity,
        )
        if index_metadata.get("index_content_digest") != canonical_digest(
            "\n".join((layer_values["L0"], layer_values["L1"], layer_values["L2"]))
        ):
            raise ProjectionIntegrityError("projection index content digest does not match layers")
    vector_metadata: dict[str, Any] | None = None
    if self.projector.vector_store is not None:
        vector_metadata = self.projector.vector_store.get_vector_metadata(
            vector_row_id(
                str(committed.object.tenant_id or "default"),
                self.projector._claim_catalog_record_key(metadata, source_revision),
            )
        )
        if vector_metadata is None:
            raise ProjectionIntegrityError("projection vector row is missing")
        self._assert_projection_identity(
            vector_metadata,
            record,
            label="vector",
            domain_identity=domain_identity,
        )
    elif record.vector_status != ProjectionStepStatus.SKIPPED.value:
        raise ProjectionIntegrityError("projection vector status cannot be completed without a store")
    scope_views = self._matching_current_views("scope", record, domain_identity)
    taxonomy_views = self._matching_current_views("taxonomy", record, domain_identity)
    if not scope_views or not taxonomy_views:
        raise ProjectionIntegrityError("projection scope or taxonomy publication is missing")
    claim_core = {
        "claim_uri": record.claim_uri,
        "source_revision": record.source_revision,
        "projection_revision": record.projection_revision,
        "projection_attempt_id": record.projection_attempt_id,
        "input_effect_hash": record.input_effect_hash,
        "publish_token": record.publish_token,
        "projected_content_digest": record.projected_content_digest,
        "projected_relation_digest": record.projected_relation_digest,
        "record_digest": str(record.to_dict()["record_digest"]),
        "publication_record_digest": projection_publication_record_digest(record),
        "layer_uris": {
            "L0": record.l0_uri,
            "L1": record.l1_uri,
            "L2": record.l2_uri,
            "manifest": record.manifest_uri,
            "relations": record.relations_uri,
        },
        "layer_digests": {name: canonical_digest(value) for name, value in layer_values.items()},
        "relation_artifact_digest": canonical_digest(relation_payload),
        "manifest_digest": canonical_digest(manifest),
        "index_metadata_digest": canonical_digest(index_metadata),
        "vector_metadata_digest": (canonical_digest(vector_metadata) if vector_metadata is not None else ""),
        "scope_view_digests": sorted(canonical_digest(item) for item in scope_views),
        "taxonomy_view_digests": sorted(canonical_digest(item) for item in taxonomy_views),
        "domain_identity": domain_identity,
    }
    return {**claim_core, "claim_proof_digest": canonical_digest(claim_core)}
