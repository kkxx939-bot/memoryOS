"""Validation responsibilities for canonical projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.core.integrity import canonical_digest, canonical_json
from memoryos.memory.canonical.current_head import (
    head_from_receipt_snapshot,
)
from memoryos.memory.canonical.projection_proof import (
    ProjectionProofStore,
    projection_publication_record_digest,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
)
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    materialized_current_revision_payload,
)
from memoryos.memory.canonical.visibility import (
    CommittedCanonicalRead,
    CommittedStateIntegrityError,
    committed_content,
    committed_relations,
    read_committed_canonical,
)
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    validate_outbox,
)
from memoryos.operations.commit.receipt import (
    ReceiptIntegrityError,
    load_transaction_receipt,
    receipt_snapshot,
)

if TYPE_CHECKING:
    from .service import CanonicalMemoryProjector


def _verified_rebuild_claim_proofs(
    self: CanonicalMemoryProjector,
    proof_store: ProjectionProofStore,
) -> dict[tuple[str, int], tuple[dict[str, Any], ProjectionRecord]]:
    """Preflight immutable publication/receipt/record closure for rebuild."""

    verified: dict[tuple[str, int], tuple[dict[str, Any], ProjectionRecord]] = {}
    for publication in proof_store.iter_publications():
        receipt = self._verified_publication_receipt(publication)
        claims = publication.get("claims")
        if not isinstance(claims, list):
            raise ProjectionIntegrityError("projection rebuild publication Claim set is invalid")
        for raw_proof in claims:
            if not isinstance(raw_proof, dict):
                raise ProjectionIntegrityError("projection rebuild Claim proof is invalid")
            claim_uri = str(raw_proof.get("claim_uri") or "")
            source_revision = int(raw_proof.get("source_revision", 0) or 0)
            identity = (claim_uri, source_revision)
            if identity in verified:
                raise ProjectionIntegrityError("projection rebuild Claim revision proof is duplicated")
            record = self._verified_projection_record_from_publication(
                raw_proof,
                receipt=receipt,
            )
            verified[identity] = (dict(raw_proof), record)
    return verified


def _verified_publication_receipt(self: CanonicalMemoryProjector, publication: dict[str, Any]) -> dict[str, Any]:
    """Verify one publication against its immutable outbox, receipt, and completion.

    The helper is deliberately transaction-bounded.  Online projection
    reconciliation uses it to recover the exact published attempt for one
    historical Claim revision without scanning every publication.
    """

    transaction_id = str(publication["transaction_id"])
    resolved_root = self.root.resolve()
    outbox_path = self.root / "system" / "outbox" / f"{transaction_id}.json"
    expected_outbox = resolved_root / "system" / "outbox" / f"{transaction_id}.json"
    try:
        resolved_outbox = outbox_path.resolve(strict=True)
    except OSError as exc:
        raise ProjectionIntegrityError("projection rebuild outbox is missing") from exc
    if outbox_path.is_symlink() or resolved_outbox != expected_outbox:
        raise ProjectionIntegrityError("projection rebuild outbox path is unsafe")
    try:
        outbox = validate_outbox(
            json.loads(outbox_path.read_text(encoding="utf-8")),
            transaction_id=transaction_id,
            tenant_id=str(publication["tenant_id"]),
            allowed_statuses={"committed"},
        )
    except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
        raise ProjectionIntegrityError("projection rebuild has no valid immutable outbox") from exc
    receipt_relative = Path(str(outbox.get("receipt_path") or ""))
    receipt_path = self.root / receipt_relative
    try:
        resolved_receipt = receipt_path.resolve(strict=True)
    except OSError as exc:
        raise ProjectionIntegrityError("projection rebuild receipt is missing") from exc
    if (
        receipt_relative.is_absolute()
        or receipt_path.is_symlink()
        or resolved_receipt == resolved_root
        or resolved_root not in resolved_receipt.parents
    ):
        raise ProjectionIntegrityError("projection rebuild receipt path is unsafe")
    try:
        receipt = load_transaction_receipt(resolved_receipt)
    except ReceiptIntegrityError as exc:
        raise ProjectionIntegrityError("projection rebuild receipt is invalid") from exc
    if (
        str(receipt.get("transaction_id") or "") != transaction_id
        or str(receipt.get("receipt_digest") or "") != str(publication["receipt_digest"])
        or str(outbox.get("receipt_digest") or "") != str(publication["receipt_digest"])
        or str(outbox.get("outbox_digest") or "") != str(publication["outbox_digest"])
        or str(receipt.get("prepared_intent_digest") or "") != str(publication["prepared_intent_digest"])
    ):
        raise ProjectionIntegrityError("projection rebuild publication differs from its receipt/outbox")
    completion = ProjectionProofStore(self.root).load_completion(transaction_id)
    if completion is not None:
        for field in (
            "commit_group_id",
            "transaction_id",
            "job_id",
            "tenant_id",
            "user_id",
            "queue_identity_digest",
            "outbox_digest",
            "receipt_digest",
            "prepared_intent_digest",
            "operation_ids",
            "claim_revisions",
            "claims",
            "publication_digest",
        ):
            if completion.get(field) != publication.get(field):
                raise ProjectionIntegrityError("projection completion proof differs from its publication receipt")
    return receipt


def _verified_projection_record_from_publication(
    self: CanonicalMemoryProjector,
    claim_proof: dict[str, Any],
    *,
    receipt: dict[str, Any],
) -> ProjectionRecord:
    claim_uri = str(claim_proof.get("claim_uri") or "")
    source_revision = int(claim_proof.get("source_revision", 0) or 0)
    try:
        snapshot = receipt_snapshot(receipt, claim_uri)
        snapshot_obj = ContextObject.from_dict(dict(snapshot["object"]))
    except (KeyError, TypeError, ValueError, ReceiptIntegrityError) as exc:
        raise ProjectionIntegrityError("projection rebuild Claim proof has no Source snapshot") from exc
    snapshot_metadata = dict(snapshot_obj.metadata or {})
    if (
        str(snapshot.get("canonical_kind") or snapshot_metadata.get("canonical_kind") or "") != "claim"
        or int(snapshot.get("after_revision", 0) or 0) != source_revision
        or int(snapshot_metadata.get("revision", 0) or 0) != source_revision
    ):
        raise ProjectionIntegrityError("projection rebuild Claim snapshot revision is inconsistent")
    historical = CommittedCanonicalRead(snapshot_obj, receipt=receipt)
    if self._input_effect_hash(historical, source_revision) != str(claim_proof.get("input_effect_hash") or ""):
        raise ProjectionIntegrityError("projection rebuild Claim effect differs from its receipt")
    domain = claim_proof.get("domain_identity")
    expected_head = head_from_receipt_snapshot(snapshot, receipt)
    try:
        snapshot_current = materialized_current_revision_payload(snapshot_metadata)
    except CanonicalMemoryInvariantError as exc:
        raise ProjectionIntegrityError("projection rebuild Claim snapshot state is invalid") from exc
    expected_domain = {
        "claim_uri": claim_uri,
        "tenant_id": str(snapshot_obj.tenant_id or "default"),
        "owner_user_id": str(snapshot_obj.owner_user_id or ""),
        "canonical_kind": "claim",
        "claim_state": str(snapshot_current.get("state") or ""),
        "canonical_head_digest": str(expected_head.get("head_digest") or ""),
        "current_transaction_id": str(receipt.get("transaction_id") or ""),
        "current_receipt_digest": str(receipt.get("receipt_digest") or ""),
        "current_claim_revision": int(snapshot_current.get("revision", 0) or 0),
    }
    if domain != expected_domain:
        raise ProjectionIntegrityError("projection rebuild Claim domain proof is inconsistent")
    try:
        current = read_committed_canonical(
            self.source_store,
            claim_uri,
            self.relation_store,
        )
    except (FileNotFoundError, CommittedStateIntegrityError) as exc:
        raise ProjectionIntegrityError("projection rebuild current Claim is unavailable") from exc
    current_revision_payload = next(
        (
            item
            for item in dict(current.object.metadata or {}).get("revisions", ()) or ()
            if isinstance(item, dict) and int(item.get("revision", 0) or 0) == source_revision
        ),
        None,
    )
    snapshot_revision_payload = next(
        (
            item
            for item in snapshot_metadata.get("revisions", ()) or ()
            if isinstance(item, dict) and int(item.get("revision", 0) or 0) == source_revision
        ),
        None,
    )
    immutable_revision_fields = (
        "revision",
        "value_fields",
        "evidence_refs",
        "proposal_id",
        "relation",
        "epistemic_status",
    )
    if (
        current_revision_payload is None
        or snapshot_revision_payload is None
        or any(
            canonical_digest(current_revision_payload.get(field))
            != canonical_digest(snapshot_revision_payload.get(field))
            for field in immutable_revision_fields
        )
    ):
        raise ProjectionIntegrityError("projection rebuild current Claim revision differs from receipt")
    layer_uris = claim_proof.get("layer_uris")
    layer_digests = claim_proof.get("layer_digests")
    if not isinstance(layer_uris, dict) or not isinstance(layer_digests, dict):
        raise ProjectionIntegrityError("projection rebuild Claim artifacts are incomplete")
    try:
        layer_values = {level: self.source_store.read_content(str(layer_uris[level])) for level in ("L0", "L1", "L2")}
        manifest = json.loads(self.source_store.read_content(str(layer_uris["manifest"])))
        relations = json.loads(self.source_store.read_content(str(layer_uris["relations"])))
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectionIntegrityError("projection rebuild Claim artifacts are unreadable") from exc
    if (
        not isinstance(manifest, dict)
        or not isinstance(relations, dict)
        or {level: canonical_digest(value) for level, value in layer_values.items()} != layer_digests
        or canonical_digest(manifest) != str(claim_proof.get("manifest_digest") or "")
        or canonical_digest(relations) != str(claim_proof.get("relation_artifact_digest") or "")
    ):
        raise ProjectionIntegrityError("projection rebuild Claim artifact digest is corrupt")
    record_payload = {key: manifest.get(key) for key in ProjectionRecord.__dataclass_fields__}
    record_payload["record_digest"] = manifest.get("record_digest")
    try:
        record = ProjectionRecord.from_dict(record_payload)
    except (KeyError, TypeError, ValueError, ProjectionIntegrityError) as exc:
        raise ProjectionIntegrityError("projection rebuild manifest has no valid projection record") from exc
    if (
        record.claim_uri != claim_uri
        or record.source_revision != source_revision
        or projection_publication_record_digest(record) != str(claim_proof.get("publication_record_digest") or "")
        or record.projected_content_digest != canonical_digest(layer_values)
        or record.projected_relation_digest != canonical_digest(relations.get("relations", []))
    ):
        raise ProjectionIntegrityError("projection rebuild record differs from publication proof")
    persisted = self.record_store.load(
        claim_uri,
        source_revision,
        projection_attempt_id=record.projection_attempt_id,
    )
    if persisted is None:
        self.record_store.save(record)
    elif projection_publication_record_digest(persisted) != projection_publication_record_digest(record):
        raise ProjectionIntegrityError("projection rebuild durable record differs from publication proof")
    return persisted or record


def _projection_domain_identity(
    self: CanonicalMemoryProjector,
    committed: CommittedCanonicalRead,
    current_revision: dict[str, Any],
) -> dict[str, Any]:
    obj = committed.object
    metadata = dict(obj.metadata or {})
    head = dict(committed.head or {})
    if (
        not head
        or head.get("uri") != obj.uri
        or head.get("canonical_kind") != "claim"
        or head.get("tenant_id") != str(obj.tenant_id or "default")
        or head.get("owner_user_id") != str(obj.owner_user_id or "")
        or not str(head.get("current_transaction_id") or "")
        or not str(head.get("receipt_digest") or "")
    ):
        raise ProjectionIntegrityError("projection Source has no complete current-head identity")
    state = str(current_revision.get("state") or "")
    if not state or state != str(metadata.get("state") or ""):
        raise ProjectionIntegrityError("projection Claim state mirror is inconsistent")
    return {
        "claim_uri": obj.uri,
        "tenant_id": str(obj.tenant_id or "default"),
        "owner_user_id": str(obj.owner_user_id or ""),
        "canonical_kind": "claim",
        "claim_state": state,
        "canonical_head_digest": str(head["head_digest"]),
        "current_transaction_id": str(head["current_transaction_id"]),
        "current_receipt_digest": str(head["receipt_digest"]),
        "current_claim_revision": int(current_revision["revision"]),
    }


def _is_current(self: CanonicalMemoryProjector, claim_uri: str, revision: int, expected_effect_hash: str) -> bool:
    try:
        committed = read_committed_canonical(
            self.source_store,
            claim_uri,
            self.relation_store,
        )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return False
    if committed.from_before_image:
        return False
    metadata = dict(committed.object.metadata or {})
    return bool(
        not committed.from_before_image
        and metadata.get("canonical_kind") == "claim"
        and int(metadata.get("revision", 0)) == revision
        and self._input_effect_hash(committed, revision) == expected_effect_hash
    )


def _input_effect_hash(
    self: CanonicalMemoryProjector,
    committed: CommittedCanonicalRead,
    source_revision: int,
) -> str:
    obj = committed.object
    content = committed_content(committed)
    relations = sorted(
        (relation.to_dict() for relation in committed_relations(committed)),
        key=canonical_json,
    )
    return canonical_digest(
        {
            "claim_uri": obj.uri,
            "source_revision": source_revision,
            "object": obj.to_dict(),
            "content": content,
            "relations": relations,
        }
    )
