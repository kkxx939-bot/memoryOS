from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import IndexHit, LockToken, QueueJob


class FileSystemSourceStore:
    """Filesystem source of truth for ContextObject metadata and L2 content."""

    def __init__(self, root: str | Path, tenant_id: str = "default") -> None:
        self.root = Path(root).expanduser().resolve()
        self.tenant_id = tenant_id

    def read_object(self, uri: str) -> ContextObject:
        path = self._object_dir(uri) / ".meta.json"
        return ContextObject.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        directory = self._object_dir(obj.uri)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_atomic(directory / ".meta.json", json.dumps(obj.to_dict(), ensure_ascii=False, indent=2))
        relations = {"uri": obj.uri, "relations": [relation.to_dict() for relation in obj.relations]}
        self._write_atomic(directory / ".relations.json", json.dumps(relations, ensure_ascii=False, indent=2))
        if content:
            self.write_content(obj.layers.l2_uri or obj.uri, content)

    def read_content(self, uri: str) -> str:
        path = self._content_path(uri)
        return path.read_text(encoding="utf-8")

    def write_content(self, uri: str, content: str | bytes) -> None:
        path = self._content_path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(content)
            os.replace(tmp, path)
        else:
            self._write_atomic(path, content)

    def soft_delete(self, uri: str, reason: str) -> None:
        obj = self.read_object(uri)
        obj.lifecycle_state = LifecycleState.DELETED
        obj.metadata = {**obj.metadata, "delete_reason": reason}
        self.write_object(obj)

    def list_objects(self) -> list[ContextObject]:
        if not self.root.exists():
            return []
        objects = []
        for path in sorted(self.root.glob("**/.meta.json")):
            objects.append(ContextObject.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return objects

    def _object_dir(self, uri: str) -> Path:
        return ContextURI.parse(uri).to_source_path(self.root, tenant_id=self.tenant_id)

    def _content_path(self, uri: str) -> Path:
        parsed = ContextURI.parse(uri)
        path = parsed.to_source_path(self.root, tenant_id=self.tenant_id)
        if path.suffix:
            return path
        return path / "content.md"

    def _write_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)


class InMemoryIndexStore:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[ContextObject, str]] = {}

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        self.rows[obj.uri] = (obj, content)

    def delete_index(self, uri: str) -> None:
        self.rows.pop(uri, None)

    def indexed_uris(self) -> list[str]:
        return list(self.rows)

    def clear(self) -> None:
        self.rows.clear()

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        filters = filters or {}
        terms = [term.lower() for term in str(query).split() if term.strip()]
        hits = []
        for obj, content in self.rows.values():
            if filters.get("owner_user_id") and obj.owner_user_id != filters["owner_user_id"]:
                continue
            if filters.get("context_type") and obj.context_type.value != filters["context_type"]:
                continue
            text = " ".join([obj.title, content, json.dumps(obj.metadata, ensure_ascii=False)]).lower()
            score = sum(1.0 for term in terms if term in text)
            if not terms:
                score = 0.1
            if score > 0:
                hits.append(IndexHit(uri=obj.uri, score=score, context_type=obj.context_type.value, title=obj.title))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


class InMemoryRelationStore:
    def __init__(self) -> None:
        self.relations: list[ContextRelation] = []

    def add_relation(self, relation: ContextRelation) -> None:
        if relation not in self.relations:
            self.relations.append(relation)

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[ContextRelation]:
        rows = [relation for relation in self.relations if relation.source_uri == uri or relation.target_uri == uri]
        if tenant_id is not None:
            rows = [relation for relation in rows if relation.metadata.get("tenant_id", "default") == tenant_id]
        if owner_user_id is not None:
            rows = [
                relation
                for relation in rows
                if relation.metadata.get("owner_user_id") in {None, "", owner_user_id}
                or relation.target_uri.startswith(("memoryos://resources/", "memoryos://skills/"))
            ]
        return rows

    def delete_relation(self, source_uri: str, relation_type: str, target_uri: str) -> None:
        self.relations = [
            relation
            for relation in self.relations
            if not (
                relation.source_uri == source_uri
                and relation.relation_type == relation_type
                and relation.target_uri == target_uri
            )
        ]


class InMemoryQueueStore:
    def __init__(self) -> None:
        self.jobs: dict[str, QueueJob] = {}

    def enqueue(self, job: QueueJob) -> None:
        self.jobs[job.job_id] = job

    def lease(self, queue_name: str, limit: int = 10) -> list[QueueJob]:
        leased = []
        for job in self.jobs.values():
            if job.queue_name == queue_name and job.status == "pending":
                leased.append(QueueJob(**{**job.__dict__, "status": "leased"}))
            if len(leased) >= limit:
                break
        for job in leased:
            self.jobs[job.job_id] = job
        return leased

    def ack(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id] = QueueJob(**{**self.jobs[job_id].__dict__, "status": "done"})

    def fail(self, job_id: str, error: str) -> None:
        if job_id in self.jobs:
            job = self.jobs[job_id]
            self.jobs[job_id] = QueueJob(**{**job.__dict__, "status": "failed", "retry_count": job.retry_count + 1, "last_error": error})


class InMemoryLockStore:
    def __init__(self) -> None:
        self.locks: dict[str, str] = {}

    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken:
        if lock_key in self.locks:
            raise TimeoutError(f"Lock already held: {lock_key}")
        token = uuid.uuid4().hex
        self.locks[lock_key] = token
        return LockToken(lock_key=lock_key, token=token)

    def release(self, token: LockToken) -> None:
        if self.locks.get(token.lock_key) == token.token:
            self.locks.pop(token.lock_key, None)
