"""上下文数据库里的索引一致性检查。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from memoryos.contextdb.extensions import (
    ContextDomainOverlay,
    ContextIndexPolicy,
    NoContextIndexPolicy,
    NoDomainOverlay,
)
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.ordinary_relations import (
    NoRelationDomainPolicy,
    RelationDomainPolicy,
    ordinary_relation_serving_eligibility,
    ordinary_relation_specs_for_object,
)
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore


@contextmanager
def _projection_fence(
    source_store: SourceStore,
    migration_gate: Any | None,
    *,
    projection_fence_held: bool,
) -> Iterator[Any | None]:
    """Serialize exported repair writers with a tenant serving rebuild."""

    if projection_fence_held:
        yield None
        return
    gate = migration_gate or getattr(source_store, "migration_gate", None)
    acquire = getattr(gate, "acquire_projection_fence", None)
    release = getattr(gate, "release_projection_fence", None)
    fence = acquire() if callable(acquire) else None
    try:
        yield fence
    finally:
        if callable(release):
            release(fence)


def _checkpoint_projection_fence(fence: Any | None) -> None:
    checkpoint = getattr(fence, "checkpoint", None)
    if callable(checkpoint):
        checkpoint()


def prepare_generic_index_rebuild(
    source_store: SourceStore,
    index_store: IndexStore,
    *,
    migration_gate: Any | None = None,
    projection_fence_held: bool = False,
    index_policy: ContextIndexPolicy | None = None,
) -> dict[str, int]:
    """Safely prepare a generic index rebuild from a direct repair entry."""

    with _projection_fence(
        source_store,
        migration_gate,
        projection_fence_held=projection_fence_held,
    ) as fence:
        return _prepare_generic_index_rebuild_unfenced(
            source_store,
            index_store,
            fence=fence,
            index_policy=index_policy,
        )


def _prepare_generic_index_rebuild_unfenced(
    source_store: SourceStore,
    index_store: IndexStore,
    *,
    fence: Any | None,
    index_policy: ContextIndexPolicy | None = None,
) -> dict[str, int]:
    """Remove generic rows while preserving proved domain-owned projections."""

    policy = index_policy or NoContextIndexPolicy()
    indexed = tuple(index_store.indexed_uris())
    preserved: set[str] = set()
    for offset, uri in enumerate(indexed):
        if offset % 256 == 0:
            _checkpoint_projection_fence(fence)
        metadata = index_store.get_index_metadata(uri)
        if not policy.owns_index_entry(source_store, uri, metadata):
            # Unified Session/Resource projections have a stable record key
            # distinct from their serving URI and cannot be reconstructed by
            # scanning SourceStore. Preserve them until the SessionArchive
            # projector or an explicit migration/retention task replaces them.
            if metadata and str(metadata.get("record_key") or uri) != uri:
                preserved.add(uri)
            continue
        if policy.preserve_index_entry(source_store, index_store, uri, metadata):
            preserved.add(uri)

    # Classify every row before mutating anything.  If a committed projection
    # is corrupt, the caller fails closed without leaving a partially-cleared
    # generic index behind.
    removed = 0
    for offset, uri in enumerate(indexed):
        if offset % 256 == 0:
            _checkpoint_projection_fence(fence)
        if uri in preserved:
            continue
        index_store.delete_index(uri)
        removed += 1
    return {"removed": removed, "canonical_preserved": len(preserved)}


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


@dataclass(frozen=True)
class OrdinaryRelationRebuildBatch:
    processed_objects: int
    projected_relations: int
    written_relations: int
    skipped_relations: int
    checkpoint: str
    complete: bool


class IndexConsistencyService:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore | None = None,
        *,
        migration_gate: Any | None = None,
        domain_overlay: ContextDomainOverlay | None = None,
        index_policy: ContextIndexPolicy | None = None,
        relation_domain_policy: RelationDomainPolicy | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.migration_gate = migration_gate or getattr(source_store, "migration_gate", None)
        self.domain_overlay = domain_overlay or NoDomainOverlay()
        self.index_policy = index_policy or NoContextIndexPolicy()
        self.relation_domain_policy = relation_domain_policy or NoRelationDomainPolicy()

    def verify(self) -> IndexConsistencyResult:
        objects = [
            obj
            for obj in self.source_store.list_objects()
            if not self.domain_overlay.owns_object(obj)
        ]
        source_uris = {obj.uri for obj in objects}
        indexed_uris = {
            uri
            for uri in self.index_store.indexed_uris()
            if not self.index_policy.owns_index_entry(
                self.source_store,
                uri,
                self.index_store.get_index_metadata(uri),
            )
        }
        derived_uris = {
            uri
            for uri in indexed_uris
            if (
                (metadata := self.index_store.get_index_metadata(uri)) is not None
                and str(metadata.get("record_key") or uri) != uri
            )
        }
        missing = sorted(
            obj.uri
            for obj in objects
            if obj.lifecycle_state not in {LifecycleState.DELETED, LifecycleState.ARCHIVED, LifecycleState.OBSOLETE}
            and obj.uri not in indexed_uris
        )
        # SessionArchive and Unified Catalog projections intentionally have no
        # ordinary SourceStore object at their serving URI.  They are derived,
        # rebuildable rows, not orphaned generic Source indexes.
        orphan = sorted(uri for uri in indexed_uris - derived_uris if uri not in source_uris)
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
            broken_relations=self._broken_relations(
                source_uris,
                tenant_id=str(getattr(self.source_store, "tenant_id", "default") or "default"),
            ),
        )

    def rebuild(self, *, projection_fence_held: bool = False) -> IndexConsistencyResult:
        with _projection_fence(
            self.source_store,
            self.migration_gate,
            projection_fence_held=projection_fence_held,
        ) as fence:
            return self._rebuild_unfenced(fence=fence)

    def _rebuild_unfenced(self, *, fence: Any | None) -> IndexConsistencyResult:
        _prepare_generic_index_rebuild_unfenced(
            self.source_store,
            self.index_store,
            fence=fence,
            index_policy=self.index_policy,
        )
        self._rebuild_regular_rows(fence=fence)
        _checkpoint_projection_fence(fence)
        return self.verify()

    def rebuild_for_canonical_reprojection(
        self,
        *,
        projection_fence_held: bool = False,
    ) -> IndexConsistencyResult:
        """Rebuild generic rows after a domain owner validated its truth.

        The Unified Catalog may also contain SessionArchive projections that
        SourceStore cannot recreate.  Preserve proved derived rows in place,
        repair regular Source rows, and let the formal canonical projectors
        idempotently overwrite their own stable keys. Callers must run
        its authoritative state before this mutation phase.
        """

        with _projection_fence(
            self.source_store,
            self.migration_gate,
            projection_fence_held=projection_fence_held,
        ) as fence:
            return self._rebuild_unfenced(fence=fence)

    def rebuild_ordinary_relations_next_batch(
        self,
        *,
        tenant_id: str,
        after_uri: str = "",
        batch_size: int = 256,
        projection_fence_held: bool = False,
    ) -> OrdinaryRelationRebuildBatch:
        """Publish one offline Source-authoritative ordinary relation batch."""

        with _projection_fence(
            self.source_store,
            self.migration_gate,
            projection_fence_held=projection_fence_held,
        ) as fence:
            return self._rebuild_ordinary_relations_next_batch_unfenced(
                tenant_id=tenant_id,
                after_uri=after_uri,
                batch_size=batch_size,
                fence=fence,
            )

    def _rebuild_ordinary_relations_next_batch_unfenced(
        self,
        *,
        tenant_id: str,
        after_uri: str,
        batch_size: int,
        fence: Any | None,
    ) -> OrdinaryRelationRebuildBatch:
        """Publish one batch while the caller owns the projection fence."""

        if self.relation_store is None:
            raise RuntimeError("ordinary relation rebuild requires RelationStore")
        relation_store = self.relation_store
        reconcile = getattr(relation_store, "reconcile_ordinary_relations", None)
        if not callable(reconcile):
            raise RuntimeError("RelationStore has no ordinary reconcile capability")
        bounded = max(1, min(int(batch_size), 1_000))
        candidates = sorted(
            (
                obj
                for obj in self.source_store.list_objects()
                if not self.domain_overlay.owns_object(obj)
                and str(obj.tenant_id or "default") == tenant_id
                and obj.uri > after_uri
            ),
            key=lambda obj: obj.uri,
        )
        batch = candidates[:bounded]
        if not batch:
            return OrdinaryRelationRebuildBatch(0, 0, 0, 0, after_uri, True)

        projected: dict[tuple[str, str, str], ContextRelation] = {}
        for offset, obj in enumerate(batch):
            if offset % 256 == 0:
                _checkpoint_projection_fence(fence)
            if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.ARCHIVED}:
                continue
            created_at_by_identity = {
                (relation.source_uri, relation.relation_type, relation.target_uri): relation.created_at
                for relation in obj.relations
            }
            for spec in ordinary_relation_specs_for_object(obj):
                eligibility = ordinary_relation_serving_eligibility(
                    spec,
                    authority_uri=obj.uri,
                    tenant_id=tenant_id,
                    source_store=self.source_store,
                    index_store=self.index_store,
                    domain_policy=self.relation_domain_policy,
                    domain_reader=lambda uri: self.domain_overlay.read_object(
                        self.source_store,
                        relation_store,
                        uri,
                    ),
                    allow_virtual_targets=True,
                )
                if not eligibility.allowed:
                    continue
                identity = (
                    str(spec["source_uri"]),
                    str(spec["relation_type"]),
                    str(spec["target_uri"]),
                )
                relation = ContextRelation(
                    source_uri=identity[0],
                    relation_type=identity[1],
                    target_uri=identity[2],
                    weight=float(spec.get("weight", 1.0)),
                    metadata=dict(spec.get("metadata", {}) or {}),
                    created_at=created_at_by_identity.get(identity, ""),
                )
                existing = projected.get(identity)
                if existing is not None and not self._ordinary_relation_equal(existing, relation):
                    raise RuntimeError("Source objects contain a conflicting ordinary relation identity")
                projected[identity] = relation
        _checkpoint_projection_fence(fence)
        result = reconcile(tuple(projected[key] for key in sorted(projected)), tenant_id=tenant_id)
        if not isinstance(result, dict):
            raise TypeError("ordinary RelationStore reconcile returned an invalid result")
        processed = int(result.get("processed", -1))
        written = int(result.get("written", -1))
        skipped = int(result.get("skipped", -1))
        if (
            processed != len(projected)
            or min(written, skipped) < 0
            or written + skipped != processed
        ):
            raise RuntimeError("ordinary RelationStore reconcile counters are inconsistent")
        checkpoint = batch[-1].uri
        return OrdinaryRelationRebuildBatch(
            processed_objects=len(batch),
            projected_relations=processed,
            written_relations=written,
            skipped_relations=skipped,
            checkpoint=checkpoint,
            complete=len(candidates) <= bounded,
        )

    def _rebuild_regular_rows(self, *, fence: Any | None) -> None:
        for offset, obj in enumerate(self.source_store.list_objects()):
            if offset % 256 == 0:
                _checkpoint_projection_fence(fence)
            if self.domain_overlay.owns_object(obj):
                continue
            if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.ARCHIVED, LifecycleState.OBSOLETE}:
                self.index_store.delete_index(obj.uri)
                continue
            try:
                content = self.source_store.read_content(obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = ""
            self.index_store.upsert_index(obj, content=content)

    @staticmethod
    def _ordinary_relation_equal(left: ContextRelation, right: ContextRelation) -> bool:
        return (
            left.source_uri == right.source_uri
            and left.relation_type == right.relation_type
            and left.target_uri == right.target_uri
            and left.weight == right.weight
            and dict(left.metadata or {}) == dict(right.metadata or {})
        )

    def _broken_relations(self, source_uris: set[str], *, tenant_id: str) -> list[dict]:
        if self.relation_store is None:
            return []
        broken: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for uri in source_uris:
            for relation in self.relation_store.relations_of(uri, tenant_id=tenant_id):
                key = (relation.source_uri, relation.relation_type, relation.target_uri)
                if key in seen:
                    continue
                seen.add(key)
                if self._global_uri(relation.source_uri) or self._global_uri(relation.target_uri):
                    continue
                if self.domain_overlay.owns_uri(
                    relation.source_uri
                ) or self.domain_overlay.owns_uri(relation.target_uri):
                    continue
                if relation.source_uri not in source_uris or relation.target_uri not in source_uris:
                    broken.append(relation.to_dict())
        return broken

    def _global_uri(self, uri: str) -> bool:
        return uri.startswith(("memoryos://resources/", "memoryos://skills/"))
