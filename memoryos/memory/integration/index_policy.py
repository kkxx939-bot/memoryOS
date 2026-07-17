"""Canonical-memory protection policy for generic ContextDB index rebuilds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.integrity import canonical_digest, canonical_json
from memoryos.memory.canonical.current_head import (
    CurrentHeadIntegrityError,
    artifact_root_for,
    load_current_head,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecordStore,
)
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    materialized_current_revision_payload,
)
from memoryos.memory.canonical.visibility import (
    committed_content,
    committed_relations,
    read_committed_canonical,
)
from memoryos.memory.integration.classification import (
    is_canonical_memory_object,
    is_canonical_memory_uri,
)


class CanonicalMemoryIndexPolicy:
    """Recognize and prove the serving rows owned by canonical memory."""

    def owns_index_entry(
        self,
        source_store: SourceStore,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        row_metadata = dict(metadata or {})
        if (
            is_canonical_memory_uri(uri)
            or str(row_metadata.get("canonical_kind") or "")
            in {"slot", "claim", "pending_proposal"}
            or str(row_metadata.get("schema_version") or "").startswith("canonical_")
        ):
            return True
        try:
            return is_canonical_memory_object(source_store.read_object(uri))
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return False

    def preserve_index_entry(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        return _is_current_canonical_projection(
            source_store,
            index_store,
            uri,
            metadata,
        )


def _is_current_canonical_projection(
    source_store: SourceStore,
    index_store: IndexStore,
    uri: str,
    index_metadata: dict[str, Any] | None,
) -> bool:
    artifact_root = artifact_root_for(source_store)
    if artifact_root is None:
        return False
    try:
        load_current_head(artifact_root, uri)
    except FileNotFoundError:
        return False
    except CurrentHeadIntegrityError as exc:
        raise ProjectionIntegrityError(
            f"generic rebuild found an invalid canonical current head: {uri}"
        ) from exc

    try:
        committed = read_committed_canonical(source_store, uri)
    except FileNotFoundError as exc:
        raise ProjectionIntegrityError(
            f"generic rebuild cannot validate committed canonical Source: {uri}"
        ) from exc
    source_metadata = dict(committed.object.metadata or {})
    if source_metadata.get("canonical_kind") != "claim":
        return False

    revision = int(source_metadata.get("revision", 0) or 0)
    try:
        current_revision = materialized_current_revision_payload(source_metadata)
    except CanonicalMemoryInvariantError as exc:
        raise ProjectionIntegrityError(
            f"generic rebuild found an invalid committed Claim state: {uri}"
        ) from exc
    record_store = ProjectionRecordStore(artifact_root)
    record = record_store.load_current(uri, source_revision=revision)
    if record is None:
        raise ProjectionIntegrityError(
            "generic rebuild found a committed Claim index row without a "
            f"current projection record: {uri}"
        )

    expected_effect_hash = canonical_digest(
        {
            "claim_uri": uri,
            "source_revision": revision,
            "object": committed.object.to_dict(),
            "content": committed_content(committed),
            "relations": sorted(
                (relation.to_dict() for relation in committed_relations(committed)),
                key=canonical_json,
            ),
        }
    )
    if (
        record.source_revision != revision
        or record.projection_revision != revision
        or record.current_claim_revision != int(current_revision["revision"])
        or record.input_effect_hash != expected_effect_hash
    ):
        raise ProjectionIntegrityError(
            f"generic rebuild found a projection detached from committed Claim state: {uri}"
        )

    try:
        layer_values = {
            "L0": source_store.read_content(record.l0_uri),
            "L1": source_store.read_content(record.l1_uri),
            "L2": source_store.read_content(record.l2_uri),
        }
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
        raise ProjectionIntegrityError(
            f"generic rebuild found a committed projection with missing layer content: {uri}"
        ) from exc
    if record.projected_content_digest != canonical_digest(layer_values):
        raise ProjectionIntegrityError(
            f"generic rebuild found a projection layer digest mismatch: {uri}"
        )

    head = dict(committed.head or {})
    expected_index_identity: dict[str, object] = {
        "claim_uri": uri,
        "tenant_id": str(committed.object.tenant_id or "default"),
        "owner_user_id": str(committed.object.owner_user_id or ""),
        "canonical_kind": "claim",
        "claim_state": str(current_revision.get("state") or ""),
        "current_transaction_id": str(head.get("current_transaction_id") or ""),
        "current_receipt_digest": str(head.get("receipt_digest") or ""),
        "current_claim_revision": int(current_revision["revision"]),
        "projection_source_revision": record.source_revision,
        "projection_revision": record.projection_revision,
        "projection_attempt_id": record.projection_attempt_id,
        "projection_input_effect_hash": record.input_effect_hash,
        "projection_publish_token": record.publish_token,
        "projection_content_digest": record.projected_content_digest,
        "projection_relation_digest": record.projected_relation_digest,
        "projection_manifest_uri": record.manifest_uri,
    }
    get_catalog = getattr(index_store, "get_catalog", None)
    if callable(get_catalog):
        from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind

        record_key = f"claim:{source_metadata.get('claim_id')}:revision:{revision}"
        compatibility_row = dict(index_metadata or {})
        if str(compatibility_row.get("record_key") or "") == uri:
            raise ProjectionIntegrityError(
                f"generic rebuild found an invalid canonical index projection: {uri}: ['record_key']"
            )
        catalog = get_catalog(
            record_key,
            tenant_id=str(committed.object.tenant_id or "default"),
        )
        if not isinstance(catalog, CatalogRecord):
            raise ProjectionIntegrityError(
                f"generic rebuild found no exact Claim Revision Catalog row: {uri}"
            )
        typed_identity = {
            "record_key": record_key,
            "uri": uri,
            "record_kind": CatalogRecordKind.CLAIM_REVISION.value,
            "source_revision": revision,
            "canonical_claim_id": str(source_metadata.get("claim_id") or ""),
            "canonical_slot_id": str(source_metadata.get("slot_id") or ""),
            "canonical_revision": revision,
            "canonical_head_digest": str(head.get("head_digest") or ""),
            "receipt_digest": str(head.get("receipt_digest") or ""),
            "projection_effect_hash": record.input_effect_hash,
        }
        if any(
            getattr(catalog, field) != expected
            for field, expected in typed_identity.items()
        ):
            raise ProjectionIntegrityError(
                f"generic rebuild found a detached Claim Revision Catalog row: {uri}"
            )
        row = {
            **dict(catalog.metadata),
            "record_key": catalog.record_key,
            "tenant_id": catalog.tenant_id,
            "owner_user_id": catalog.owner_user_id,
            "index_content_digest": canonical_digest(catalog.l1_text),
        }
        expected_index_identity["index_content_digest"] = canonical_digest(
            catalog.l1_text
        )
    else:
        row = dict(index_metadata or {})
        expected_index_identity.update(
            {
                "projection_record_path": str(record_store.attempt_path_for(record)),
                "index_content_digest": canonical_digest(
                    "\n".join(
                        (layer_values["L0"], layer_values["L1"], layer_values["L2"])
                    )
                ),
            }
        )
    mismatched = []
    for field_name, expected_value in expected_index_identity.items():
        actual_value = row.get(field_name)
        if field_name == "projection_record_path":
            if _same_path(actual_value, expected_value):
                continue
        elif actual_value == expected_value:
            continue
        mismatched.append(field_name)
    if mismatched:
        raise ProjectionIntegrityError(
            f"generic rebuild found an invalid canonical index projection: {uri}: {mismatched}"
        )
    return True


def _same_path(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    try:
        return Path(left).expanduser().resolve(strict=False) == Path(
            right
        ).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False


__all__ = ["CanonicalMemoryIndexPolicy"]
