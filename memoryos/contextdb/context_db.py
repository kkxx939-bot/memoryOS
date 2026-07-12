"""上下文数据库里的上下文数据库入口。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult
from memoryos.contextdb.store.index_consistency import IndexConsistencyService
from memoryos.contextdb.store.source_store import IndexHit, IndexStore, QueueStore, RelationStore, SourceStore
from memoryos.memory.canonical.projection_state import ProjectionRecordStore
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
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.session_commit_service = session_commit_service
        self.committer = committer
        self.projection_store = projection_store

    def read_object(self, uri: str) -> ContextObject:
        return self.source_store.read_object(uri)

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """写入 object。"""
        self.seed_object(obj, content=content)

    def seed_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """处理 seed object 这一步。"""
        self.source_store.write_object(obj, content=content)
        self.index_store.upsert_index(obj, content=self._index_content(obj, content))

    import_object = seed_object

    def commit_operation(self, operation: ContextOperation) -> CommitResult:
        if self.committer is None:
            raise RuntimeError("ContextDB.commit_operation requires OperationCommitter")
        return self.committer.commit(operation.user_id, [operation])

    def commit_operations(self, operations: list[ContextOperation]) -> list[CommitResult]:
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
        return self.index_store.search(query, filters=filters, limit=limit)

    def relations_of(
        self,
        uri: str,
        *,
        owner_user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[ContextRelation]:
        return self.relation_store.relations_of(uri, tenant_id=tenant_id, owner_user_id=owner_user_id)

    def commit_session(self, archive: SessionArchive, *, async_commit: bool = True) -> SessionCommitResult:
        if self.session_commit_service is None:
            raise RuntimeError("ContextDB requires SessionCommitService to commit sessions")
        # 兼容原有公开接口：True 会立刻跑异步提交阶段，False 只归档并留下后台任务。
        if async_commit:
            self.session_commit_service.sync_archive(archive, enqueue_commit_job=False)
            return self.session_commit_service.async_commit(archive)
        return self.session_commit_service.sync_archive(archive, enqueue_commit_job=True)

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict:
        if owner_user_id is None:
            result = IndexConsistencyService(self.source_store, self.index_store, self.relation_store).rebuild()
            return self._consistency_payload(result)
        rebuilt = 0
        for obj in self.source_store.list_objects():
            if obj.owner_user_id != owner_user_id:
                continue
            try:
                content = self.source_store.read_content(obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = obj.title
            self.index_store.upsert_index(obj, content=content)
            rebuilt += 1
        payload = self.verify_consistency(owner_user_id=owner_user_id)
        payload["rebuilt_count"] = rebuilt
        return payload

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict:
        result = IndexConsistencyService(self.source_store, self.index_store, self.relation_store).verify()
        payload = self._consistency_payload(result)
        if owner_user_id is None:
            return payload
        source_uris = {obj.uri for obj in self.source_store.list_objects() if obj.owner_user_id == owner_user_id}
        indexed_uris = set(self.index_store.indexed_uris())
        payload["source_count"] = len(source_uris)
        payload["indexed_count"] = len(source_uris & indexed_uris)
        payload["missing_index"] = [uri for uri in payload["missing_index"] if uri in source_uris]
        payload["dangling_index"] = [
            uri
            for uri in payload["dangling_index"]
            if uri.startswith(f"memoryos://user/{owner_user_id}/")
        ]
        payload["broken_relations"] = [
            relation
            for relation in payload["broken_relations"]
            if relation.get("source_uri") in source_uris or relation.get("target_uri") in source_uris
        ]
        payload["consistent"] = not (
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
