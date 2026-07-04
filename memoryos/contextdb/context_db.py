from __future__ import annotations

from collections import defaultdict

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult
from memoryos.contextdb.store.index_consistency import IndexConsistencyService
from memoryos.contextdb.store.source_store import IndexHit, IndexStore, QueueStore, RelationStore, SourceStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation

CommitResult = ContextDiff


class ContextDB:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore,
        queue_store: QueueStore | None = None,
        session_commit_service=None,
        committer: OperationCommitter | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.session_commit_service = session_commit_service
        self.committer = committer

    def read_object(self, uri: str) -> ContextObject:
        return self.source_store.read_object(uri)

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """Seed/import helper only; not for production long-term writes.

        Production updates must use ContextOperation through commit_operation(),
        commit_operations(), or SessionCommit/OperationCommitter.
        """
        self.seed_object(obj, content=content)

    def seed_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """Write SourceStore and IndexStore directly for tests, imports, and seed data."""
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
    ) -> list[IndexHit]:
        filters = {}
        if owner_user_id is not None:
            filters["owner_user_id"] = owner_user_id
        if context_type is not None:
            filters["context_type"] = context_type.value
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
        result = self.session_commit_service.sync_archive(archive)
        if async_commit:
            return self.session_commit_service.async_commit(archive)
        return result

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
