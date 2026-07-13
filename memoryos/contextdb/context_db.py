"""上下文数据库里的上下文数据库入口。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult
from memoryos.contextdb.store.index_consistency import (
    IndexConsistencyService,
    validate_canonical_authoritative_state,
)
from memoryos.contextdb.store.source_store import (
    IndexHit,
    IndexStore,
    QueueStore,
    RelationStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)
from memoryos.memory.canonical.projection_state import ProjectionRecordStore
from memoryos.memory.canonical.visibility import list_committed_relations, read_committed_canonical
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation

CommitResult = ContextDiff


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
        projection_store: ProjectionRecordStore | None = None,
        canonical_projector=None,
        readiness=None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.session_commit_service = session_commit_service
        self.committer = committer
        self.projection_store = projection_store
        self.canonical_projector = canonical_projector
        self.readiness = readiness

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

    def _mark_not_ready(self, error: BaseException, *, artifact: str) -> None:
        mark_not_ready = getattr(self.readiness, "mark_not_ready", None)
        if callable(mark_not_ready):
            mark_not_ready(
                f"canonical consistency failure: {type(error).__name__}: {error}",
                details={"artifact": artifact, "error_type": type(error).__name__},
            )

    def _canonical_preflight(self) -> tuple[dict[str, int], Any | None]:
        """Prove authoritative canonical inputs before any rebuild mutation."""

        try:
            result = validate_canonical_authoritative_state(
                self.source_store,
                self.relation_store,
                self.projection_store,
            )
            worker = None
            if self.canonical_projector is not None and self.queue_store is not None:
                from memoryos.memory.canonical.projection import MemoryProjectionWorker

                worker = MemoryProjectionWorker(self.canonical_projector, self.queue_store)
                dispatched = worker.dispatch_outbox()
                # This validates only immutable publication/outbox/receipt/queue
                # bindings.  Index/vector/views are rebuildable and are checked
                # separately after a rebuild.
                worker._validate_authoritative_projection_proofs()
                queue_stats = self.queue_store.stats(queue_name="memory_projection")
                if queue_stats.get("dead_letter", 0) or queue_stats.get("quarantine", 0):
                    raise RuntimeError("canonical rebuild queue contains terminal failed work")
                if queue_stats.get("leased", 0):
                    raise RuntimeError("canonical rebuild queue contains an active lease")
                result = {
                    **result,
                    "projection_outbox_transactions": len(dispatched),
                    "projection_queue_pending": int(queue_stats.get("pending", 0) or 0),
                    "projection_queue_done": int(queue_stats.get("done", 0) or 0),
                }
            elif result["canonical_claims"]:
                raise RuntimeError("canonical projection proof validation is unavailable")
            return result, worker
        except Exception as exc:
            self._mark_not_ready(exc, artifact="canonical_rebuild_preflight")
            raise

    def _verify_canonical_projection(self, worker: Any | None) -> tuple[dict[str, Any], str]:
        if worker is None:
            return {"verified": 0, "publications": 0, "completions": 0}, ""
        try:
            current = worker.verify_current_projections()
            proofs = worker.validate_projection_proofs()
            return {**current, **proofs}, ""
        except Exception as exc:
            # A concurrent authoritative failure marks readiness in the
            # committed-read/proof layer and must propagate.  Missing or stale
            # derived rows remain a reportable, rebuildable inconsistency.
            state = str(getattr(getattr(self.readiness, "state", None), "value", "READY"))
            if state != "READY":
                self._require_ready()
            return {}, f"{type(exc).__name__}: {exc}"

    def read_object(self, uri: str) -> ContextObject:
        self._require_ready()
        if "/memories/canonical/" in uri or "/memories/pending/" in uri:
            return read_committed_canonical(self.source_store, uri, self.relation_store).object
        obj = self.source_store.read_object(uri)
        if is_canonical_memory_object(obj):
            return read_committed_canonical(self.source_store, uri, self.relation_store).object
        return obj

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """写入 object。"""
        self._require_ready()
        self.seed_object(obj, content=content)

    def seed_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """处理 seed object 这一步。"""
        self._require_ready()
        if self._canonical_object(obj):
            raise PermissionError(
                "canonical memory cannot be seeded through ContextDB; use the canonical committer "
                "or a migration/recovery SourceStore owned by startup"
            )
        self.source_store.write_object(obj, content=content)
        self.index_store.upsert_index(obj, content=self._index_content(obj, content))

    import_object = seed_object

    def commit_operation(self, operation: ContextOperation) -> CommitResult:
        self._require_ready()
        if self.committer is None:
            raise RuntimeError("ContextDB.commit_operation requires OperationCommitter")
        return self.committer.commit(operation.user_id, [operation])

    def commit_operations(self, operations: list[ContextOperation]) -> list[CommitResult]:
        self._require_ready()
        if self.committer is None:
            raise RuntimeError("ContextDB.commit_operations requires OperationCommitter")
        results: list[CommitResult] = []
        operations_by_user: dict[str, list[ContextOperation]] = defaultdict(list)
        for operation in operations:
            operations_by_user[operation.user_id].append(operation)
        for user_id, user_operations in operations_by_user.items():
            results.append(self.committer.commit(user_id, user_operations))
        return results

    def add_relation(self, relation: ContextRelation) -> None:
        self._require_ready()
        canonical_endpoint = self._canonical_uri(relation.source_uri) or self._canonical_uri(relation.target_uri)
        if not canonical_endpoint:
            for uri in (relation.source_uri, relation.target_uri):
                try:
                    endpoint = self.source_store.read_object(uri)
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    continue
                if self._canonical_object(endpoint):
                    canonical_endpoint = True
                    break
        if canonical_endpoint:
            raise PermissionError("canonical relations can only be published from an immutable transaction receipt")
        self.relation_store.add_relation(relation)

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
        hits = self.index_store.search(query, filters=filters, limit=limit)
        visible: list[IndexHit] = []
        for hit in hits:
            if self._canonical_uri(hit.uri) or str(dict(hit.metadata or {}).get("canonical_kind") or "") in {
                "slot",
                "claim",
                "pending_proposal",
            }:
                continue
            try:
                obj = self.source_store.read_object(hit.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            if self._canonical_object(obj):
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
        canonical_target = self._canonical_uri(uri)
        if not canonical_target:
            try:
                canonical_target = self._canonical_object(self.source_store.read_object(uri))
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                canonical_target = False
        if canonical_target:
            read_committed_canonical(self.source_store, uri, self.relation_store)
            relations = list(list_committed_relations(self.source_store, uri, self.relation_store))
            if tenant_id is not None:
                relations = [
                    relation
                    for relation in relations
                    if str(relation.metadata.get("tenant_id") or "default") == tenant_id
                ]
            if owner_user_id is not None:
                relations = [
                    relation
                    for relation in relations
                    if str(relation.metadata.get("owner_user_id") or "") == owner_user_id
                ]
            return relations
        return self.relation_store.relations_of(uri, tenant_id=tenant_id, owner_user_id=owner_user_id)

    def commit_session(self, archive: SessionArchive, *, async_commit: bool = True) -> SessionCommitResult:
        self._require_ready()
        if self.session_commit_service is None:
            raise RuntimeError("ContextDB requires SessionCommitService to commit sessions")
        # 兼容原有公开接口：True 会立刻跑异步提交阶段，False 只归档并留下后台任务。
        if async_commit:
            self.session_commit_service.sync_archive(archive, enqueue_commit_job=False)
            return self.session_commit_service.async_commit(archive)
        return self.session_commit_service.sync_archive(archive, enqueue_commit_job=True)

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict:
        self._require_ready()
        authoritative, projection_worker = self._canonical_preflight()
        try:
            consistency = IndexConsistencyService(
                self.source_store,
                self.index_store,
                self.relation_store,
            )
            rebuilt = 0
            if owner_user_id is None:
                result = consistency.rebuild_for_canonical_reprojection()
            else:
                for obj in self.source_store.list_objects():
                    if obj.owner_user_id != owner_user_id or self._canonical_object(obj):
                        continue
                    try:
                        content = self.source_store.read_content(obj.uri)
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        content = obj.title
                    self.index_store.upsert_index(obj, content=content)
                    rebuilt += 1
                result = consistency.verify()
            payload = self._consistency_payload(result)
            payload["canonical_authoritative"] = authoritative
            if self.canonical_projector is not None:
                payload["canonical_projection"] = self.canonical_projector.rebuild()
            projection, projection_error = self._verify_canonical_projection(projection_worker)
            if projection_error:
                raise RuntimeError(projection_error)
            payload["canonical_projection_validation"] = projection
            if owner_user_id is not None:
                payload["rebuilt_count"] = rebuilt
            self._require_ready()
            return payload
        except Exception as exc:
            self._mark_not_ready(exc, artifact="canonical_rebuild_publication")
            raise

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict:
        self._require_ready()
        authoritative, projection_worker = self._canonical_preflight()
        result = IndexConsistencyService(self.source_store, self.index_store, self.relation_store).verify()
        payload = self._consistency_payload(result)
        projection, projection_error = self._verify_canonical_projection(projection_worker)
        payload["canonical_authoritative"] = authoritative
        payload["canonical_projection_validation"] = projection
        payload["canonical_projection_error"] = projection_error
        if projection_error:
            payload["consistent"] = False
        if owner_user_id is None:
            return payload
        source_uris = {
            obj.uri
            for obj in self.source_store.list_objects()
            if obj.owner_user_id == owner_user_id and not self._canonical_object(obj)
        }
        indexed_uris = set(self.index_store.indexed_uris())
        payload["source_count"] = len(source_uris)
        payload["indexed_count"] = len(source_uris & indexed_uris)
        payload["missing_index"] = [uri for uri in payload["missing_index"] if uri in source_uris]
        payload["dangling_index"] = [
            uri for uri in payload["dangling_index"] if uri.startswith(f"memoryos://user/{owner_user_id}/")
        ]
        payload["broken_relations"] = [
            relation
            for relation in payload["broken_relations"]
            if relation.get("source_uri") in source_uris or relation.get("target_uri") in source_uris
        ]
        payload["consistent"] = not projection_error and not (
            payload["missing_index"]
            or payload["dangling_index"]
            or payload["deleted_or_archived_in_default_search"]
            or payload["broken_relations"]
        )
        return payload

    def _consistency_payload(self, result) -> dict:
        return {
            "source_count": result.source_count,
            "indexed_count": result.index_count,
            "missing_index": result.missing_in_index,
            "dangling_index": result.orphan_index,
            "deleted_or_archived_in_default_search": result.deleted_or_archived_in_default_search,
            "broken_relations": result.broken_relations,
            "consistent": result.consistent,
        }

    def _index_content(self, obj: ContextObject, content: str | bytes) -> str:
        if isinstance(content, bytes):
            return obj.title
        return content or obj.metadata.get("summary", obj.title)

    @staticmethod
    def _canonical_object(obj: ContextObject) -> bool:
        return is_canonical_memory_object(obj)

    @staticmethod
    def _canonical_uri(uri: str) -> bool:
        return is_canonical_memory_uri(uri)
