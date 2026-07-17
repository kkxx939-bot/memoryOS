"""Historical responsibilities for canonical projection."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.core.integrity import canonical_digest
from memoryos.memory.canonical.current_head import (
    head_from_receipt_snapshot,
)
from memoryos.memory.canonical.projection_proof import (
    projection_publication_record_digest,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
    ProjectionStatus,
    ProjectionStepStatus,
)
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    materialized_current_revision_payload,
)
from memoryos.memory.canonical.visibility import (
    CommittedCanonicalRead,
    committed_relations,
)
from memoryos.operations.commit.receipt import (
    ReceiptIntegrityError,
    receipt_snapshot,
)

if TYPE_CHECKING:
    from .worker import MemoryProjectionWorker


def _verify_historical_claim_projection(
    self: MemoryProjectionWorker,
    claim_proof: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    """Validate a retired projection without consulting mutable current rows."""

    claim_uri = str(claim_proof.get("claim_uri") or "")
    source_revision = int(claim_proof.get("source_revision", 0))
    try:
        snapshot = receipt_snapshot(receipt, claim_uri)
        obj = ContextObject.from_dict(dict(snapshot["object"]))
    except (KeyError, TypeError, ValueError, ReceiptIntegrityError) as exc:
        raise ProjectionIntegrityError("historical projection has no matching immutable Source effect") from exc
    metadata = dict(obj.metadata or {})
    if (
        str(snapshot.get("canonical_kind") or metadata.get("canonical_kind") or "") != "claim"
        or int(metadata.get("revision", 0)) != source_revision
        or int(snapshot.get("after_revision", 0)) != source_revision
    ):
        raise ProjectionIntegrityError("historical projection Source revision is inconsistent")
    try:
        materialized_current = materialized_current_revision_payload(metadata)
    except CanonicalMemoryInvariantError as exc:
        raise ProjectionIntegrityError("historical projection Source domain state is invalid") from exc
    expected_domain_identity = {
        "claim_uri": claim_uri,
        "tenant_id": str(obj.tenant_id or "default"),
        "owner_user_id": str(obj.owner_user_id or ""),
        "canonical_kind": "claim",
        "claim_state": str(materialized_current.get("state") or ""),
        "canonical_head_digest": str(head_from_receipt_snapshot(snapshot, receipt)["head_digest"]),
        "current_transaction_id": str(receipt["transaction_id"]),
        "current_receipt_digest": str(receipt["receipt_digest"]),
        "current_claim_revision": int(materialized_current["revision"]),
    }
    if claim_proof.get("domain_identity") != expected_domain_identity or expected_domain_identity["claim_state"] != str(
        metadata.get("state") or ""
    ):
        raise ProjectionIntegrityError("historical projection domain identity differs from receipt")
    historical_committed = CommittedCanonicalRead(obj, receipt=receipt)
    expected_effect_hash = self.projector._input_effect_hash(
        historical_committed,
        source_revision,
    )
    if claim_proof.get("input_effect_hash") != expected_effect_hash:
        raise ProjectionIntegrityError("historical projection input effect differs from receipt")
    attempt_id = str(claim_proof.get("projection_attempt_id") or "")
    expected_base = f"{claim_uri}/projections/rev-{source_revision}/attempt-{attempt_id}"
    expected_uris = {
        "L0": f"{expected_base}/l0.md",
        "L1": f"{expected_base}/l1.md",
        "L2": f"{expected_base}/l2.json",
        "manifest": f"{expected_base}/manifest.json",
        "relations": f"{expected_base}/relations.json",
    }
    if not attempt_id or claim_proof.get("layer_uris") != expected_uris:
        raise ProjectionIntegrityError("historical projection artifact URIs are inconsistent")
    layer_values = {
        "L0": self.projector.source_store.read_content(expected_uris["L0"]),
        "L1": self.projector.source_store.read_content(expected_uris["L1"]),
        "L2": self.projector.source_store.read_content(expected_uris["L2"]),
    }
    layer_digests = {name: canonical_digest(value) for name, value in layer_values.items()}
    if claim_proof.get("layer_digests") != layer_digests or claim_proof.get(
        "projected_content_digest"
    ) != canonical_digest(layer_values):
        raise ProjectionIntegrityError("historical projection layer content is corrupt")
    try:
        relation_payload = json.loads(self.projector.source_store.read_content(expected_uris["relations"]))
        manifest = json.loads(self.projector.source_store.read_content(expected_uris["manifest"]))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProjectionIntegrityError("historical projection artifact is malformed") from exc
    if not isinstance(relation_payload, dict) or not isinstance(manifest, dict):
        raise ProjectionIntegrityError("historical projection artifact is not an object")
    record_payload = {key: manifest.get(key) for key in ProjectionRecord.__dataclass_fields__}
    record_payload["record_digest"] = manifest.get("record_digest")
    try:
        record = ProjectionRecord.from_dict(record_payload)
    except (KeyError, TypeError, ValueError, ProjectionIntegrityError) as exc:
        raise ProjectionIntegrityError("historical projection manifest has no valid attempt snapshot") from exc
    if record.status not in {ProjectionStatus.COMPLETED.value, ProjectionStatus.STALE.value}:
        raise ProjectionIntegrityError("historical projection attempt has an invalid terminal state")
    if projection_publication_record_digest(record) != claim_proof.get("publication_record_digest"):
        raise ProjectionIntegrityError("historical projection attempt differs from publication receipt")
    expected_record_fields = {
        "claim_uri": record.claim_uri,
        "source_revision": record.source_revision,
        "projection_revision": record.projection_revision,
        "projection_attempt_id": record.projection_attempt_id,
        "input_effect_hash": record.input_effect_hash,
        "publish_token": record.publish_token,
        "projected_content_digest": record.projected_content_digest,
        "projected_relation_digest": record.projected_relation_digest,
    }
    if any(claim_proof.get(key) != value for key, value in expected_record_fields.items()):
        raise ProjectionIntegrityError("historical projection record identity is inconsistent")
    if (
        claim_proof.get("relation_artifact_digest") != canonical_digest(relation_payload)
        or claim_proof.get("manifest_digest") != canonical_digest(manifest)
        or record.projected_relation_digest != canonical_digest(relation_payload.get("relations", []))
    ):
        raise ProjectionIntegrityError("historical projection artifact digest is corrupt")
    self._assert_projection_identity(
        relation_payload,
        record,
        label="historical relation",
        domain_identity=expected_domain_identity,
    )
    self._assert_projection_identity(
        manifest,
        record,
        label="historical manifest",
        domain_identity=expected_domain_identity,
    )
    if claim_proof.get("historical_only") is True:
        expected_attestations = {
            component: self._historical_component_attestation(
                component,
                claim_uri=claim_uri,
                source_revision=source_revision,
                transaction_id=str(receipt["transaction_id"]),
                receipt_digest=str(receipt["receipt_digest"]),
            )
            for component in ("index", "vector", "scope", "taxonomy")
        }
        if (
            claim_proof.get("index_metadata_digest") != expected_attestations["index"]
            or claim_proof.get("vector_metadata_digest") != expected_attestations["vector"]
            or claim_proof.get("scope_view_digests") != [expected_attestations["scope"]]
            or claim_proof.get("taxonomy_view_digests") != [expected_attestations["taxonomy"]]
        ):
            raise ProjectionIntegrityError("historical projection component attestation is inconsistent")
    self._restore_historical_projection_record(record)


def _materialize_historical_claim_projection(
    self: MemoryProjectionWorker,
    receipt: dict[str, Any],
    claim_uri: str,
    source_revision: int,
) -> dict[str, Any]:
    """Project an immutable receipt snapshot without replacing current rows."""

    try:
        snapshot = receipt_snapshot(receipt, claim_uri)
        obj = ContextObject.from_dict(dict(snapshot["object"]))
    except (KeyError, TypeError, ValueError, ReceiptIntegrityError) as exc:
        raise ProjectionIntegrityError("historical projection has no matching immutable Source effect") from exc
    metadata = dict(obj.metadata or {})
    if (
        str(snapshot.get("canonical_kind") or metadata.get("canonical_kind") or "") != "claim"
        or int(metadata.get("revision", 0)) != source_revision
        or int(snapshot.get("after_revision", 0)) != source_revision
    ):
        raise ProjectionIntegrityError("historical projection Source revision is inconsistent")
    try:
        materialized = materialized_current_revision_payload(metadata)
        revision = self.projector._revision_payload(
            metadata,
            int(materialized["revision"]),
        )
    except (CanonicalMemoryInvariantError, KeyError, TypeError, ValueError) as exc:
        raise ProjectionIntegrityError("historical projection Source domain state is invalid") from exc
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
    if not domain_identity["claim_state"] or domain_identity["claim_state"] != str(metadata.get("state") or ""):
        raise ProjectionIntegrityError("historical projection Claim state is inconsistent")
    committed = CommittedCanonicalRead(obj, receipt=receipt)
    input_effect_hash = self.projector._input_effect_hash(committed, source_revision)
    attempt_id = canonical_digest(
        {
            "schema_version": "historical_projection_attempt_v1",
            "transaction_id": str(receipt["transaction_id"]),
            "receipt_digest": str(receipt["receipt_digest"]),
            "claim_uri": claim_uri,
            "source_revision": source_revision,
        }
    )[:32]
    slot_uri = claim_uri.rsplit("/claims/", 1)[0]
    base = f"{claim_uri}/projections/rev-{source_revision}/attempt-{attempt_id}"
    record = self.projector.record_store.start(
        claim_uri=claim_uri,
        slot_uri=slot_uri,
        source_revision=source_revision,
        projection_revision=source_revision,
        projection_attempt_id=attempt_id,
        input_effect_hash=input_effect_hash,
        l0_uri=f"{base}/l0.md",
        l1_uri=f"{base}/l1.md",
        l2_uri=f"{base}/l2.json",
        relations_uri=f"{base}/relations.json",
        manifest_uri=f"{base}/manifest.json",
        current_claim_revision=int(materialized["revision"]),
    )
    record = self.projector.record_store.update(
        record,
        index_status=ProjectionStepStatus.SKIPPED.value,
        vector_status=ProjectionStepStatus.SKIPPED.value,
        relation_status=ProjectionStepStatus.RUNNING.value,
        scope_status=ProjectionStepStatus.SKIPPED.value,
        taxonomy_status=ProjectionStepStatus.SKIPPED.value,
        status=ProjectionStatus.RUNNING.value,
        failure_reason="",
        retryable=True,
        current=False,
    )
    l0, l1, l2 = self.projector._layers(
        obj,
        metadata,
        revision,
        source_revision,
    )
    relations = [item.to_dict() for item in committed_relations(committed)]
    record = self.projector.record_store.update(
        record,
        projected_content_digest=canonical_digest({"L0": l0, "L1": l1, "L2": l2}),
        projected_relation_digest=canonical_digest(relations),
    )
    self.projector.source_store.write_content(record.l0_uri, l0)
    self.projector.source_store.write_content(record.l1_uri, l1)
    self.projector.source_store.write_content(record.l2_uri, l2)
    relation_payload = {
        **domain_identity,
        "claim_uri": claim_uri,
        "slot_uri": slot_uri,
        "source_revision": source_revision,
        "projection_revision": record.projection_revision,
        "projection_attempt_id": record.projection_attempt_id,
        "input_effect_hash": record.input_effect_hash,
        "publish_token": record.publish_token,
        "projected_content_digest": record.projected_content_digest,
        "projected_relation_digest": record.projected_relation_digest,
        "relations": relations,
    }
    self.projector.source_store.write_content(
        record.relations_uri,
        json.dumps(relation_payload, ensure_ascii=False, indent=2, sort_keys=True),
    )
    record = self.projector.record_store.update(
        record,
        relation_status=ProjectionStepStatus.COMPLETED.value,
    )
    record = self.projector.record_store.stale(
        record,
        "canonical revision advanced before projection publication",
    )
    manifest = self.projector._manifest(
        record,
        metadata,
        record.relations_uri,
        domain_identity=domain_identity,
    )
    self.projector.source_store.write_content(
        record.manifest_uri,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
    )
    layer_values = {"L0": l0, "L1": l1, "L2": l2}
    attestations = {
        component: self._historical_component_attestation(
            component,
            claim_uri=claim_uri,
            source_revision=source_revision,
            transaction_id=str(receipt["transaction_id"]),
            receipt_digest=str(receipt["receipt_digest"]),
        )
        for component in ("index", "vector", "scope", "taxonomy")
    }
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
        "index_metadata_digest": attestations["index"],
        "vector_metadata_digest": attestations["vector"],
        "scope_view_digests": [attestations["scope"]],
        "taxonomy_view_digests": [attestations["taxonomy"]],
        "domain_identity": domain_identity,
        "historical_only": True,
    }
    proof = {**claim_core, "claim_proof_digest": canonical_digest(claim_core)}
    self._verify_historical_claim_projection(proof, receipt)
    return proof


def _historical_component_attestation(
    component: str,
    *,
    claim_uri: str,
    source_revision: int,
    transaction_id: str,
    receipt_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema_version": "historical_projection_component_attestation_v1",
            "component": component,
            "status": "skipped_superseded",
            "claim_uri": claim_uri,
            "source_revision": source_revision,
            "transaction_id": transaction_id,
            "receipt_digest": receipt_digest,
        }
    )


def _restore_historical_projection_record(self: MemoryProjectionWorker, record: ProjectionRecord) -> None:
    """Restore disposable attempt state from its immutable manifest snapshot."""

    path = self.projector.record_store.attempt_path_for(record)
    try:
        persisted = self.projector.record_store.load(
            record.claim_uri,
            record.source_revision,
            projection_attempt_id=record.projection_attempt_id,
        )
    except ProjectionIntegrityError:
        persisted = None
    if persisted is not None:
        if projection_publication_record_digest(persisted) == (projection_publication_record_digest(record)):
            return
        quarantine_control_file(
            self.projector.root,
            path,
            kind="projection_record",
            error=ValueError("projection attempt differs from immutable publication manifest"),
            identifiers={
                "record_id": path.stem,
                "claim_uri": record.claim_uri,
                "projection_attempt_id": record.projection_attempt_id,
            },
        )
    self.projector.record_store.save(record)


def _matching_current_views(
    self: MemoryProjectionWorker,
    kind: str,
    record: ProjectionRecord,
    domain_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in self.projector.root.glob(f"views/{kind}/**/current.json"):
        payload = self.projector._read_json_optional(path)
        if payload is None or str(payload.get("claim_uri") or "") != record.claim_uri:
            continue
        self._assert_projection_identity(
            payload,
            record,
            label=kind,
            domain_identity=domain_identity,
        )
        matches.append(payload)
    return matches


def _assert_projection_identity(
    payload: dict[str, Any],
    record: ProjectionRecord,
    *,
    label: str,
    domain_identity: dict[str, Any],
) -> None:
    expected = {
        "source_revision": record.source_revision,
        "projection_revision": record.projection_revision,
        "projection_attempt_id": record.projection_attempt_id,
        "input_effect_hash": record.input_effect_hash,
        "publish_token": record.publish_token,
        "projected_content_digest": record.projected_content_digest,
        "projected_relation_digest": record.projected_relation_digest,
        **domain_identity,
    }
    aliases = {
        "source_revision": ("source_revision", "projection_source_revision"),
        "projection_revision": ("projection_revision",),
        "projection_attempt_id": ("projection_attempt_id",),
        "input_effect_hash": ("input_effect_hash", "projection_input_effect_hash"),
        "publish_token": ("publish_token", "projection_publish_token"),
        "projected_content_digest": (
            "projected_content_digest",
            "projection_content_digest",
        ),
        "projected_relation_digest": (
            "projected_relation_digest",
            "projection_relation_digest",
        ),
        "claim_uri": ("claim_uri",),
        "tenant_id": ("tenant_id",),
        "owner_user_id": ("owner_user_id",),
        "canonical_kind": ("canonical_kind",),
        "claim_state": ("claim_state",),
        "canonical_head_digest": ("canonical_head_digest",),
        "current_transaction_id": ("current_transaction_id",),
        "current_receipt_digest": ("current_receipt_digest",),
        "current_claim_revision": ("current_claim_revision",),
    }
    for field_name, expected_value in expected.items():
        actual = next((payload.get(key) for key in aliases[field_name] if key in payload), None)
        if actual != expected_value:
            raise ProjectionIntegrityError(f"projection {label} {field_name} mismatch")
