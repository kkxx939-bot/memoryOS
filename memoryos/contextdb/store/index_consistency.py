"""上下文数据库里的索引一致性检查。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import (
    IndexStore,
    RelationStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)


def prepare_generic_index_rebuild(source_store: SourceStore, index_store: IndexStore) -> dict[str, int]:
    """Remove generic and unproved rows while retaining valid canonical projections.

    A generic rebuild has no authority to republish canonical memory.  It must
    therefore leave an already-published, current projection byte-for-byte
    intact.  Canonical-looking rows without a current head are raw/stale rows
    and are removed.  A row for a committed Claim that cannot be bound to its
    current projection record is an integrity failure, not cleanup material.
    """

    indexed = tuple(index_store.indexed_uris())
    preserved: set[str] = set()
    for uri in indexed:
        metadata = index_store.get_index_metadata(uri)
        if not _canonical_index_entry(source_store, uri, metadata):
            continue
        if _is_current_canonical_projection(source_store, index_store, uri, metadata):
            preserved.add(uri)

    # Classify every row before mutating anything.  If a committed projection
    # is corrupt, the caller fails closed without leaving a partially-cleared
    # generic index behind.
    removed = 0
    for uri in indexed:
        if uri in preserved:
            continue
        index_store.delete_index(uri)
        removed += 1
    return {"removed": removed, "canonical_preserved": len(preserved)}


def validate_canonical_authoritative_state(
    source_store: SourceStore,
    relation_store: RelationStore | None,
    projection_store: Any | None,
) -> dict[str, int]:
    """Validate non-rebuildable canonical truth before derived mutation.

    Source bundles, immutable receipts/current heads and the current projection
    record are authoritative inputs to a rebuild.  Index/vector/view rows are
    deliberately not checked here because they are the rebuildable outputs.
    """

    from memoryos.memory.canonical.event import canonical_digest, canonical_json
    from memoryos.memory.canonical.projection_state import ProjectionIntegrityError
    from memoryos.memory.canonical.state import (
        CanonicalMemoryInvariantError,
        materialized_current_revision_payload,
    )
    from memoryos.memory.canonical.visibility import (
        capture_committed_canonical_snapshot,
        committed_content,
        committed_relations,
    )

    snapshot = capture_committed_canonical_snapshot(source_store, relation_store)
    claims = {
        uri: committed
        for uri, committed in snapshot.records.items()
        if str(dict(committed.object.metadata or {}).get("canonical_kind") or "") == "claim"
    }
    if projection_store is None:
        if claims:
            raise ProjectionIntegrityError("committed canonical Claims have no projection record store")
        return {
            "canonical_objects": len(snapshot.records),
            "canonical_claims": 0,
            "projection_records": 0,
        }

    current_records = {record.claim_uri: record for record in projection_store.iter_current()}
    if set(current_records) != set(claims):
        dangling = sorted(set(current_records) - set(claims))
        missing = sorted(set(claims) - set(current_records))
        raise ProjectionIntegrityError(
            f"projection current/head closure mismatch; dangling={dangling}; missing={missing}"
        )

    for claim_uri, committed in claims.items():
        metadata = dict(committed.object.metadata or {})
        source_revision = int(metadata.get("revision", 0) or 0)
        try:
            current_revision = materialized_current_revision_payload(metadata)
        except CanonicalMemoryInvariantError as exc:
            raise ProjectionIntegrityError(
                f"committed Claim has an invalid materialized revision: {claim_uri}"
            ) from exc
        expected_effect_hash = canonical_digest(
            {
                "claim_uri": claim_uri,
                "source_revision": source_revision,
                "object": committed.object.to_dict(),
                "content": committed_content(committed),
                "relations": sorted(
                    (relation.to_dict() for relation in committed_relations(committed)),
                    key=canonical_json,
                ),
            }
        )
        record = current_records[claim_uri]
        if (
            record.slot_uri != claim_uri.rsplit("/claims/", 1)[0]
            or record.source_revision != source_revision
            or record.projection_revision != source_revision
            or record.current_claim_revision != int(current_revision["revision"])
            or record.input_effect_hash != expected_effect_hash
            or not record.current
            or not record.usable
        ):
            raise ProjectionIntegrityError(
                f"projection current record is detached from committed Claim state: {claim_uri}"
            )
    return {
        "canonical_objects": len(snapshot.records),
        "canonical_claims": len(claims),
        "projection_records": len(current_records),
    }


def _canonical_index_entry(
    source_store: SourceStore,
    uri: str,
    metadata: dict | None,
) -> bool:
    row_metadata = dict(metadata or {})
    if (
        is_canonical_memory_uri(uri)
        or str(row_metadata.get("canonical_kind") or "") in {"slot", "claim", "pending_proposal"}
        or str(row_metadata.get("schema_version") or "").startswith("canonical_")
    ):
        return True
    try:
        return is_canonical_memory_object(source_store.read_object(uri))
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return False


def _is_current_canonical_projection(
    source_store: SourceStore,
    index_store: IndexStore,
    uri: str,
    index_metadata: dict | None,
) -> bool:
    # Keep ContextDB's foundational store package importable while canonical
    # session modules are initializing; these dependencies are needed only
    # when a rebuild actually encounters a canonical-looking row.
    from memoryos.memory.canonical.current_head import (
        CurrentHeadIntegrityError,
        artifact_root_for,
        load_current_head,
    )
    from memoryos.memory.canonical.event import canonical_digest, canonical_json
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

    artifact_root = artifact_root_for(source_store)
    if artifact_root is None:
        return False
    try:
        load_current_head(artifact_root, uri)
    except FileNotFoundError:
        return False
    except CurrentHeadIntegrityError as exc:
        raise ProjectionIntegrityError(f"generic rebuild found an invalid canonical current head: {uri}") from exc

    try:
        committed = read_committed_canonical(source_store, uri)
    except FileNotFoundError as exc:
        raise ProjectionIntegrityError(f"generic rebuild cannot validate committed canonical Source: {uri}") from exc
    source_metadata = dict(committed.object.metadata or {})
    if source_metadata.get("canonical_kind") != "claim":
        # Slots and Pending Proposals never have generic index publication
        # authority, even when their Source state is committed.
        return False

    revision = int(source_metadata.get("revision", 0) or 0)
    try:
        current_revision = materialized_current_revision_payload(source_metadata)
    except CanonicalMemoryInvariantError as exc:
        raise ProjectionIntegrityError(f"generic rebuild found an invalid committed Claim state: {uri}") from exc
    record_store = ProjectionRecordStore(artifact_root)
    record = record_store.load_current(uri, source_revision=revision)
    if record is None:
        raise ProjectionIntegrityError(
            f"generic rebuild found a committed Claim index row without a current projection record: {uri}"
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
        raise ProjectionIntegrityError(f"generic rebuild found a projection detached from committed Claim state: {uri}")

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
        raise ProjectionIntegrityError(f"generic rebuild found a projection layer digest mismatch: {uri}")

    head = dict(committed.head or {})
    expected_index_identity = {
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
        "projection_record_path": str(record_store.attempt_path_for(record)),
        "index_content_digest": canonical_digest(
            "\n".join((layer_values["L0"], layer_values["L1"], layer_values["L2"]))
        ),
    }
    row = dict(index_metadata or {})
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
        return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False


@dataclass(frozen=True)
class IndexConsistencyResult:
    source_count: int
    index_count: int
    missing_in_index: list[str]
    orphan_index: list[str] = field(default_factory=list)
    deleted_or_archived_in_default_search: list[str] = field(default_factory=list)
    broken_relations: list[dict] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not (
            self.missing_in_index
            or self.orphan_index
            or self.deleted_or_archived_in_default_search
            or self.broken_relations
        )


class IndexConsistencyService:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store

    def verify(self) -> IndexConsistencyResult:
        objects = [obj for obj in self.source_store.list_objects() if not self._canonical(obj)]
        source_uris = {obj.uri for obj in objects}
        indexed_uris = {uri for uri in self.index_store.indexed_uris() if "/memories/canonical/" not in uri}
        missing = sorted(
            obj.uri
            for obj in objects
            if obj.lifecycle_state not in {LifecycleState.DELETED, LifecycleState.ARCHIVED, LifecycleState.OBSOLETE}
            and obj.uri not in indexed_uris
        )
        orphan = sorted(uri for uri in indexed_uris if uri not in source_uris)
        hot_in_default = []
        for obj in objects:
            if obj.lifecycle_state not in {LifecycleState.DELETED, LifecycleState.ARCHIVED, LifecycleState.OBSOLETE}:
                continue
            hits = self.index_store.search(
                obj.title or obj.uri,
                filters={"owner_user_id": obj.owner_user_id, "context_type": obj.context_type.value},
                limit=50,
            )
            if obj.uri in {hit.uri for hit in hits}:
                hot_in_default.append(obj.uri)
        return IndexConsistencyResult(
            source_count=len(objects),
            index_count=len(indexed_uris),
            missing_in_index=missing,
            orphan_index=orphan,
            deleted_or_archived_in_default_search=sorted(hot_in_default),
            broken_relations=self._broken_relations(source_uris),
        )

    def rebuild(self) -> IndexConsistencyResult:
        prepare_generic_index_rebuild(self.source_store, self.index_store)
        self._rebuild_regular_rows()
        return self.verify()

    def rebuild_for_canonical_reprojection(self) -> IndexConsistencyResult:
        """Rebuild generic rows after ContextDB validated canonical truth.

        Unlike the generic rebuild, a full canonical rebuild intentionally
        removes all derived index rows so the formal projector can republish
        Claims.  Callers must run ``validate_canonical_authoritative_state``
        before invoking this mutation phase.
        """

        self.index_store.clear()
        self._rebuild_regular_rows()
        return self.verify()

    def _rebuild_regular_rows(self) -> None:
        for obj in self.source_store.list_objects():
            if self._canonical(obj):
                continue
            if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.ARCHIVED, LifecycleState.OBSOLETE}:
                self.index_store.delete_index(obj.uri)
                continue
            try:
                content = self.source_store.read_content(obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = ""
            self.index_store.upsert_index(obj, content=content)

    def _canonical(self, obj) -> bool:  # noqa: ANN001
        return is_canonical_memory_object(obj)

    def _broken_relations(self, source_uris: set[str]) -> list[dict]:
        if self.relation_store is None:
            return []
        broken: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for uri in source_uris:
            for relation in self.relation_store.relations_of(uri):
                key = (relation.source_uri, relation.relation_type, relation.target_uri)
                if key in seen:
                    continue
                seen.add(key)
                if self._global_uri(relation.source_uri) or self._global_uri(relation.target_uri):
                    continue
                if self._canonical_uri(relation.source_uri) or self._canonical_uri(relation.target_uri):
                    continue
                if relation.source_uri not in source_uris or relation.target_uri not in source_uris:
                    broken.append(relation.to_dict())
        return broken

    def _global_uri(self, uri: str) -> bool:
        return uri.startswith(("memoryos://resources/", "memoryos://skills/"))

    @staticmethod
    def _canonical_uri(uri: str) -> bool:
        return is_canonical_memory_uri(uri)
