"""上下文数据库里的上下文数据库入口。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from threading import RLock
from typing import TYPE_CHECKING, Any, TypeAlias

from memoryos.contextdb.commit_protocol import ContextCommitter
from memoryos.contextdb.extensions import (
    ContextDomainOverlay,
    ContextIndexPolicy,
    NoContextIndexPolicy,
    NoDomainOverlay,
)
from memoryos.contextdb.maintenance import (
    ContextDBAdministration,
    GenericContextMaintenance,
)
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.ordinary_relations import (
    NoRelationDomainPolicy,
    RelationDomainPolicy,
    ordinary_relation_serving_eligibility,
)
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult
from memoryos.contextdb.store.index_store import IndexHit, IndexStore
from memoryos.contextdb.store.queue_store import QueueStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore

if TYPE_CHECKING:
    # Preserve the public annotation spelling without introducing a runtime
    # dependency on the operations implementation.
    CommitResult: TypeAlias = Any
    OperationCommitter: TypeAlias = ContextCommitter
    from memoryos.operations.model.context_operation import ContextOperation


class ContextDB:
    """统一封装上下文读写、检索、关系和会话提交。"""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore,
        queue_store: QueueStore | None = None,
        session_commit_service=None,
        committer: OperationCommitter | None = None,
        document_overlay: Any | None = None,
        tombstone_service=None,
        retention_manager=None,
        readiness=None,
        tenant_id: str = "",
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.session_commit_service = session_commit_service
        self.committer = committer
        self.tenant_id = str(tenant_id or getattr(source_store, "tenant_id", "default") or "default")
        self.tombstone_service = tombstone_service
        if (
            self.committer is not None
            and tombstone_service is not None
            and getattr(self.committer, "tombstone_service", None) is None
        ):
            # ContextDB is also assembled directly by embedders and tests;
            # preserve the same DELETE outbox boundary as the runtime factory.
            self.committer.tombstone_service = tombstone_service
        self.retention_manager = retention_manager
        self.readiness = readiness
        # Markdown documents are not SourceStore objects.  Retrieval receives
        # a Catalog candidate and asks this exact-byte reader to validate its
        # live path/digest before exposing content.
        self.memory_document_overlay = document_overlay
        # Administrative rebuilds and online retrieval share one publication
        # boundary so a query observes either the pre-rebuild or post-rebuild
        # serving state, never a partially republished Catalog.
        self.serving_lock = RLock()
        self._configure_extensions()

    def _configure_extensions(
        self,
        *,
        domain_overlay: ContextDomainOverlay | None = None,
        index_policy: ContextIndexPolicy | None = None,
        administration_service: ContextDBAdministration | None = None,
        relation_domain_policy: RelationDomainPolicy | None = None,
    ) -> None:
        """Install composition-owned domain hooks without widening the public constructor."""

        self.domain_overlay = domain_overlay or NoDomainOverlay()
        self.index_policy = index_policy or NoContextIndexPolicy()
        self.relation_domain_policy = relation_domain_policy or NoRelationDomainPolicy()
        self.administration_service = administration_service or GenericContextMaintenance(
            self.source_store,
            self.index_store,
            self.relation_store,
            tenant_id=self.tenant_id,
            domain_overlay=self.domain_overlay,
            index_policy=self.index_policy,
            readiness=self.readiness,
            serving_lock=self.serving_lock,
            relation_domain_policy=self.relation_domain_policy,
        )

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

    @contextmanager
    def _mutation_fence(self) -> Iterator[None]:
        """Serialize Source and serving publication within this composition."""

        with self.serving_lock:
            yield

    def read_object(self, uri: str) -> ContextObject:
        self._require_ready()
        if self.domain_overlay.owns_uri(uri):
            return self.domain_overlay.read_object(self.source_store, self.relation_store, uri)
        obj = self.source_store.read_object(uri)
        if self.domain_overlay.owns_object(obj):
            return self.domain_overlay.read_object(self.source_store, self.relation_store, uri)
        return obj

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """写入 object。"""
        self._require_ready()
        self.seed_object(obj, content=content)

    def seed_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """处理 seed object 这一步。"""
        with self._mutation_fence():
            self._require_ready()
            if self._domain_owned_object(obj) or self._document_owned_object(obj):
                raise PermissionError(
                    "domain-owned context cannot be seeded through ContextDB; use its domain committer "
                    "or a startup-owned recovery SourceStore"
                )
            self.source_store.write_object(obj, content=content)
            self.index_store.upsert_index(
                obj,
                content=self._index_content(obj, content),
                tenant_id=str(obj.tenant_id or self.tenant_id),
            )

    import_object = seed_object

    def delete_context(self, uri: str, *, reason: str = "context_deleted") -> dict[str, Any]:
        with self._mutation_fence():
            return self._delete_context_unfenced(uri, reason=reason)

    def _delete_context_unfenced(self, uri: str, *, reason: str) -> dict[str, Any]:
        """Retire Source, then replay its pre-enqueued derived tombstones.

        A crash before Source retirement leaves a durable but ineligible
        tombstone; replay fails closed until Source is actually non-serving.
        """

        self._require_ready()
        if self.tombstone_service is None:
            raise RuntimeError("ContextDB.delete_context requires ProjectionTombstoneService")
        obj = self.source_store.read_object(uri)
        tenant_id = str(obj.tenant_id or "default")
        tombstones = self.enqueue_context_tombstones(uri, tenant_id=tenant_id, reason=reason)
        self.source_store.soft_delete(uri, reason)
        result = self.process_projection_tombstones(tombstones)
        if result.failed:
            raise RuntimeError("derived projection tombstone cleanup is retryable but incomplete")
        return {
            "uri": uri,
            "tenant_id": tenant_id,
            "tombstone_ids": list(tombstones),
            "processed": list(result.processed),
            "stale": list(result.stale),
        }

    def delete_session_context(self, session_id: str, *, reason: str = "session_deleted") -> dict[str, Any]:
        with self._mutation_fence():
            return self._delete_session_context_unfenced(session_id, reason=reason)

    def _delete_session_context_unfenced(self, session_id: str, *, reason: str) -> dict[str, Any]:
        """Remove Session serving projections while retaining immutable evidence."""

        self._require_ready()
        if self.tombstone_service is None:
            raise RuntimeError("ContextDB.delete_session_context requires ProjectionTombstoneService")
        tenant_id = str(getattr(self.source_store, "tenant_id", "default") or "default")
        tombstones = self.tombstone_service.enqueue_session(
            session_id,
            tenant_id=tenant_id,
            reason=reason,
        )
        result = self.process_projection_tombstones(tombstones)
        if result.failed:
            raise RuntimeError("Session projection tombstone cleanup is retryable but incomplete")
        return {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "tombstone_ids": list(tombstones),
            "processed": list(result.processed),
            "stale": list(result.stale),
            "evidence_retained": True,
        }

    def run_retention_cycle(self, *, now: datetime | None = None) -> dict[str, Any]:
        with self._mutation_fence():
            return self._run_retention_cycle_unfenced(now=now)

    def _run_retention_cycle_unfenced(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Run one bounded lifecycle cycle through durable projection tombstones."""

        self._require_ready()
        if self.retention_manager is None:
            raise RuntimeError("ContextDB.run_retention_cycle requires CatalogRetentionManager")
        tenant_id = str(getattr(self.source_store, "tenant_id", "default") or "default")
        tiers = self.retention_manager.apply_serving_tiers(tenant_id=tenant_id, now=now)
        vectors = self.retention_manager.gc_vectors(tenant_id=tenant_id)
        stale = self.retention_manager.gc_stale_projections(tenant_id=tenant_id)
        auxiliary = self.retention_manager.gc_auxiliary_state(tenant_id=tenant_id, now=now)
        return {
            "tenant_id": tenant_id,
            "tiers": asdict(tiers),
            "vectors": asdict(vectors),
            "stale": asdict(stale),
            "auxiliary": asdict(auxiliary),
        }

    def compact_session_context(
        self,
        session_id: str,
        *,
        owner_user_id: str = "",
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        with self._mutation_fence():
            return self._compact_session_context_unfenced(
                session_id,
                owner_user_id=owner_user_id,
                now=now,
            )

    def _compact_session_context_unfenced(
        self,
        session_id: str,
        *,
        owner_user_id: str = "",
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Create a bounded Session L1 projection without deleting Archive evidence."""

        self._require_ready()
        if self.retention_manager is None:
            raise RuntimeError("ContextDB.compact_session_context requires CatalogRetentionManager")
        tenant_id = str(getattr(self.source_store, "tenant_id", "default") or "default")
        record = self.retention_manager.compact_session(
            tenant_id=tenant_id,
            session_id=session_id,
            owner_user_id=owner_user_id,
            now=now,
        )
        if record is None:
            return None
        self.retention_manager.gc_vectors(tenant_id=tenant_id)
        return record.to_dict()

    def restore_cold_context(
        self,
        record_key: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        with self._mutation_fence():
            return self._restore_cold_context_unfenced(record_key, now=now)

    def _restore_cold_context_unfenced(
        self,
        record_key: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Restore one exact COLD/ARCHIVED serving record to bounded WARM access."""

        self._require_ready()
        if self.retention_manager is None:
            raise RuntimeError("ContextDB.restore_cold_context requires CatalogRetentionManager")
        tenant_id = str(getattr(self.source_store, "tenant_id", "default") or "default")
        return self.retention_manager.restore_cold_record(
            record_key,
            tenant_id=tenant_id,
            now=now,
        ).to_dict()

    def compact_timeline_context(
        self,
        timeline_path: str,
        *,
        owner_user_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        with self._mutation_fence():
            return self._compact_timeline_context_unfenced(
                timeline_path,
                owner_user_id=owner_user_id,
                now=now,
            )

    def _compact_timeline_context_unfenced(
        self,
        timeline_path: str,
        *,
        owner_user_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Create one derived day overview from database-filtered path members."""

        self._require_ready()
        if self.retention_manager is None:
            raise RuntimeError("ContextDB.compact_timeline_context requires CatalogRetentionManager")
        tenant_id = str(getattr(self.source_store, "tenant_id", "default") or "default")
        record = self.retention_manager.compact_timeline(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            timeline_path=timeline_path,
            now=now,
        )
        return record.to_dict() if record is not None else None

    def enqueue_context_tombstones(
        self,
        uri: str,
        *,
        tenant_id: str,
        reason: str,
    ) -> tuple[str, ...]:
        """Durably journal all projections before a normal Context is retired."""

        with self._mutation_fence():
            if self.tombstone_service is None:
                raise RuntimeError("Context deletion requires ProjectionTombstoneService")
            return self.tombstone_service.enqueue_uri(
                uri,
                tenant_id=tenant_id,
                reason=reason,
                require_source_retired=True,
            )

    def process_projection_tombstones(
        self,
        tombstone_ids: list[str] | tuple[str, ...],
        *,
        tenant_id: str = "",
    ):
        """Apply one delete request's exact journal set without queue starvation."""

        with self._mutation_fence():
            if self.tombstone_service is None:
                raise RuntimeError("Context deletion requires ProjectionTombstoneService")
            return self.tombstone_service.process_tombstones(
                tombstone_ids,
                tenant_id=str(tenant_id or self.tenant_id),
            )

    def commit_operation(self, operation: ContextOperation) -> CommitResult:
        self._require_ready()
        if self.committer is None:
            raise RuntimeError("ContextDB.commit_operation requires OperationCommitter")
        return self.committer.commit(operation.user_id, [operation])

    def commit_operations(self, operations: list[ContextOperation]) -> list[CommitResult]:
        self._require_ready()
        if self.committer is None:
            raise RuntimeError("ContextDB.commit_operations requires OperationCommitter")
        results: list[Any] = []
        operations_by_user: dict[str, list[Any]] = defaultdict(list)
        for operation in operations:
            operations_by_user[operation.user_id].append(operation)
        for user_id, user_operations in operations_by_user.items():
            results.append(self.committer.commit(user_id, user_operations))
        return results

    def add_relation(self, relation: ContextRelation) -> None:
        with self._mutation_fence():
            self._add_relation_unfenced(relation)

    def _add_relation_unfenced(self, relation: ContextRelation) -> None:
        self._require_ready()
        if self._document_owned_uri(relation.source_uri) or self._document_owned_uri(
            relation.target_uri
        ):
            raise PermissionError(
                "Markdown document relations can only be published by the document projector"
            )
        domain_source = self._domain_owned_uri(relation.source_uri)
        endpoint_objects: dict[str, ContextObject] = {}
        for uri in (relation.source_uri, relation.target_uri):
            try:
                endpoint = self.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            endpoint_objects[uri] = endpoint
            if self._document_owned_object(endpoint):
                raise PermissionError(
                    "Markdown document relations can only be published by the document projector"
                )
            if uri == relation.source_uri and self._domain_owned_object(endpoint):
                domain_source = True
        if domain_source:
            raise PermissionError(
                "domain-owned relations can only be published by their authoritative handler"
            )
        if not endpoint_objects:
            raise FileNotFoundError("ordinary relation requires at least one durable Source endpoint")

        identity = (relation.source_uri, relation.relation_type, relation.target_uri)
        existing_authorities = [
            obj
            for obj in endpoint_objects.values()
            if any(
                (item.source_uri, item.relation_type, item.target_uri) == identity
                for item in obj.relations
            )
        ]
        if len(existing_authorities) > 1:
            raise RuntimeError("ordinary relation has multiple Source authorities")
        authority = (
            existing_authorities[0]
            if existing_authorities
            else endpoint_objects.get(relation.source_uri) or endpoint_objects[relation.target_uri]
        )
        if self._domain_owned_object(authority):
            raise PermissionError("ordinary relation requires a generic Source authority")
        tenant_id = str(authority.tenant_id or "default")
        declared_tenant = relation.metadata.get("tenant_id")
        if declared_tenant not in (None, "", tenant_id):
            raise PermissionError("ordinary relation metadata crosses its Source tenant")
        owner_user_id = str(authority.owner_user_id or relation.metadata.get("owner_user_id") or "")
        if not owner_user_id:
            raise ValueError("ordinary relation Source authority requires an owner_user_id")
        declared_owner = relation.metadata.get("owner_user_id")
        if declared_owner not in (None, "", owner_user_id):
            raise PermissionError("ordinary relation metadata crosses its Source owner")
        for uri, endpoint in endpoint_objects.items():
            if uri == authority.uri or ContextURI.parse(uri).authority != "user":
                continue
            if str(endpoint.tenant_id or "default") != tenant_id:
                raise PermissionError("ordinary relation endpoints cross a tenant boundary")
            endpoint_owner = str(endpoint.owner_user_id or "")
            if (
                not self._domain_owned_object(endpoint)
                and endpoint_owner
                and endpoint_owner != owner_user_id
            ):
                raise PermissionError("ordinary relation endpoints cross an owner boundary")
        normalized_metadata = dict(relation.metadata or {})
        normalized_metadata.pop("catalog_record_key", None)
        normalized_metadata.update({"tenant_id": tenant_id, "owner_user_id": owner_user_id})
        existing_relation = next(
            (
                item
                for item in authority.relations
                if (item.source_uri, item.relation_type, item.target_uri) == identity
            ),
            None,
        )
        desired = ContextRelation(
            source_uri=relation.source_uri,
            relation_type=relation.relation_type,
            target_uri=relation.target_uri,
            weight=relation.weight,
            metadata=normalized_metadata,
            created_at=(existing_relation.created_at if existing_relation is not None else relation.created_at),
        )
        eligibility = ordinary_relation_serving_eligibility(
            {
                "source_uri": desired.source_uri,
                "relation_type": desired.relation_type,
                "target_uri": desired.target_uri,
                "weight": desired.weight,
                "metadata": dict(desired.metadata or {}),
            },
            authority_uri=authority.uri,
            tenant_id=tenant_id,
            source_store=self.source_store,
            index_store=self.index_store,
            domain_policy=self.relation_domain_policy,
            domain_reader=lambda uri: self.domain_overlay.read_object(
                self.source_store, self.relation_store, uri
            ),
        )
        if not eligibility.allowed:
            raise ValueError(
                "ordinary relation is not serving-eligible: "
                f"{eligibility.reason or 'endpoint is unavailable'}"
            )
        if (
            existing_relation is not None
            and self._ordinary_relation_equal(existing_relation, desired)
        ):
            self._repair_ordinary_relation_projection(desired, tenant_id=tenant_id)
            return

        desired_authority = ContextObject.from_dict(authority.to_dict())
        updated_relations: list[ContextRelation] = []
        replaced = False
        for item in desired_authority.relations:
            if (item.source_uri, item.relation_type, item.target_uri) != identity:
                updated_relations.append(item)
                continue
            if not replaced:
                updated_relations.append(desired)
                replaced = True
        if not replaced:
            updated_relations.append(desired)
        desired_authority.relations = updated_relations
        desired_authority.updated_at = datetime.now(timezone.utc).isoformat()
        try:
            content = self.source_store.read_content(authority.layers.l2_uri or authority.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            content = ""
        committer = self._ordinary_relation_committer(tenant_id)
        committer.commit_ordinary_relation_update(
            owner_user_id=owner_user_id,
            desired_authority=desired_authority,
            content=content,
            tenant_id=tenant_id,
        )

    def _ordinary_relation_committer(self, tenant_id: str) -> ContextCommitter:
        committer = self.committer
        if committer is None:
            raise RuntimeError("ContextDB.add_relation requires an injected ContextCommitter")
        if not callable(getattr(committer, "commit_ordinary_relation_update", None)):
            raise RuntimeError(
                "ContextDB.add_relation requires a committer with ordinary relation support"
            )
        if (
            committer.source_store is not self.source_store
            or committer.index_store is not self.index_store
            or committer.relation_store is not self.relation_store
            or committer.tenant_id != tenant_id
        ):
            raise RuntimeError("ContextDB.add_relation committer differs from its bound stores")
        return committer

    def _repair_ordinary_relation_projection(self, relation: ContextRelation, *, tenant_id: str) -> None:
        reconcile = getattr(self.relation_store, "reconcile_ordinary_relations", None)
        if callable(reconcile):
            reconcile((relation,), tenant_id=tenant_id)
            return
        self.relation_store.add_relation(relation, tenant_id=tenant_id)

    @staticmethod
    def _ordinary_relation_equal(left: ContextRelation, right: ContextRelation) -> bool:
        return (
            left.source_uri == right.source_uri
            and left.relation_type == right.relation_type
            and left.target_uri == right.target_uri
            and left.weight == right.weight
            and dict(left.metadata or {}) == dict(right.metadata or {})
        )

    def search(
        self,
        query: str,
        *,
        owner_user_id: str | None = None,
        context_type: ContextType | None = None,
        limit: int = 10,
        tenant_id: str = "",
        project_id: str = "",
        adapter_id: str = "",
        admission_status: str = "",
        allowed_uris: list[str] | tuple[str, ...] | None = None,
    ) -> list[IndexHit]:
        """按给定条件查找匹配结果。"""

        self._require_ready()
        filters: dict[str, Any] = {}
        if owner_user_id is not None:
            filters["owner_user_id"] = owner_user_id
        if tenant_id:
            filters["tenant_id"] = tenant_id
        if context_type is not None:
            filters["context_type"] = context_type.value
        if project_id:
            filters["project_id"] = project_id
        if adapter_id:
            filters["adapter_id"] = adapter_id
        if admission_status:
            filters["admission_status"] = admission_status
        if allowed_uris is not None:
            filters["allowed_uris"] = tuple(allowed_uris)
        effective_tenant = str(tenant_id or self.tenant_id)
        filters["tenant_id"] = effective_tenant
        hits = self.index_store.search(
            query,
            tenant_id=effective_tenant,
            filters=filters,
            limit=limit,
        )
        visible: list[IndexHit] = []
        for hit in hits:
            if self.index_policy.owns_index_entry(
                self.source_store,
                hit.uri,
                dict(hit.metadata or {}),
            ):
                continue
            try:
                obj = self.source_store.read_object(hit.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            if self._domain_owned_object(obj):
                continue
            visible.append(hit)
        return visible

    def relations_of(
        self,
        uri: str,
        *,
        owner_user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[ContextRelation]:
        self._require_ready()
        effective_tenant = str(tenant_id or self.tenant_id)
        domain_target = self._domain_owned_uri(uri)
        if not domain_target:
            try:
                domain_target = self._domain_owned_object(
                    self.source_store.read_object(uri)
                )
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                domain_target = False
        if domain_target:
            return self.domain_overlay.relations_of(
                self.source_store,
                self.relation_store,
                uri,
                owner_user_id=owner_user_id,
                tenant_id=effective_tenant,
            )
        return self.relation_store.relations_of(
            uri,
            tenant_id=effective_tenant,
            owner_user_id=owner_user_id,
        )

    def commit_session(self, archive: SessionArchive, *, async_commit: bool = True) -> SessionCommitResult:
        self._require_ready()
        if self.session_commit_service is None:
            raise RuntimeError("ContextDB requires SessionCommitService to commit sessions")
        return self.session_commit_service.commit_session(archive, async_commit=async_commit)

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict:
        return self.administration_service.rebuild_index(owner_user_id=owner_user_id)

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict:
        return self.administration_service.verify_consistency(owner_user_id=owner_user_id)

    def _index_content(self, obj: ContextObject, content: str | bytes) -> str:
        if isinstance(content, bytes):
            return obj.title
        return content or obj.metadata.get("summary", obj.title)

    def _domain_owned_object(self, obj: ContextObject) -> bool:
        return self.domain_overlay.owns_object(obj)

    def _domain_owned_uri(self, uri: str) -> bool:
        return self.domain_overlay.owns_uri(uri)

    @staticmethod
    def _document_owned_object(obj: ContextObject) -> bool:
        return obj.context_type is ContextType.MEMORY or ContextDB._document_owned_uri(obj.uri)

    @staticmethod
    def _document_owned_uri(uri: str) -> bool:
        raw = str(uri or "")
        return raw.startswith("memoryos://user/") and "/memory/documents/" in raw


__all__ = ["ContextDB"]
