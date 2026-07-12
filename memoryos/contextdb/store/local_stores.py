"""上下文数据库里的本地存储集合。"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import IndexHit, LockLostError, LockToken, QueueJob
from memoryos.contextdb.store.sqlite_index_store import lexical_match_count, lexical_relevance, lexical_terms


class FileSystemSourceStore:
    """负责 FileSystemSourceStore 的持久化读写。"""

    def __init__(self, root: str | Path, tenant_id: str = "default") -> None:
        if (
            not isinstance(tenant_id, str)
            or not tenant_id.strip()
            or tenant_id in {".", ".."}
            or "/" in tenant_id
            or "\\" in tenant_id
        ):
            raise ValueError("tenant_id must be one safe non-empty path segment")
        self.root = Path(root).expanduser().resolve()
        self.tenant_id = tenant_id

    def read_object(self, uri: str) -> ContextObject:
        path = self._object_dir(uri) / ".meta.json"
        obj = ContextObject.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if ContextURI.parse(uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
            raise FileNotFoundError(uri)
        return obj

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        if ContextURI.parse(obj.uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
            raise PermissionError("ContextObject tenant does not match SourceStore tenant")
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

    def delete_object(self, uri: str) -> None:
        directory = self._object_dir(uri)
        if directory.exists():
            shutil.rmtree(directory)

    def list_objects(self) -> list[ContextObject]:
        if not self.root.exists():
            return []
        objects = []
        paths = [
            *self.root.glob(f"tenants/{self.tenant_id}/**/.meta.json"),
            *self.root.glob("resources/**/.meta.json"),
            *self.root.glob("skills/**/.meta.json"),
        ]
        for path in sorted(set(paths)):
            obj = ContextObject.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if ContextURI.parse(obj.uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
                continue
            objects.append(obj)
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
        hits = []
        for obj, content in self.rows.values():
            if "allowed_uris" in filters and obj.uri not in set(filters.get("allowed_uris", []) or []):
                continue
            if filters.get("lifecycle_state") is None and obj.lifecycle_state in {
                LifecycleState.DELETED,
                LifecycleState.ARCHIVED,
                LifecycleState.OBSOLETE,
            }:
                continue
            if filters.get("lifecycle_state") and obj.lifecycle_state.value != filters["lifecycle_state"]:
                continue
            if filters.get("owner_user_id") and obj.owner_user_id != filters["owner_user_id"]:
                continue
            if filters.get("tenant_id") and str(obj.tenant_id or "default") != str(filters["tenant_id"]):
                continue
            if filters.get("context_type") and obj.context_type.value != filters["context_type"]:
                continue
            metadata = dict(obj.metadata or {})
            try:
                from memoryos.memory.canonical.scope import scope_keys_from_payloads

                raw_scope = metadata.get("scope", {}) or {}
                if not isinstance(raw_scope, dict):
                    continue
                raw_applicability = raw_scope.get("applicability", {}) or {}
                if not isinstance(raw_applicability, dict):
                    continue
                actual_scope_keys = set(scope_keys_from_payloads(raw_applicability.get("all_of", [])))
            except (KeyError, TypeError, ValueError):
                continue
            admission = dict(metadata.get("admission", {}) or {})
            if filters.get("admission_status") is None and admission.get("decision") in {
                "pending",
                "restricted",
                "archive_only",
                "reject",
            }:
                continue
            if filters.get("project_id"):
                scope = raw_scope
                fields = dict(metadata.get("fields", {}) or {})
                applicability = raw_applicability
                workspace = next(
                    (
                        str(item.get("id"))
                        for item in applicability.get("all_of", []) or []
                        if isinstance(item, dict) and item.get("kind") == "workspace"
                    ),
                    "",
                )
                project_id = str(
                    scope.get("project_id") or fields.get("project_id") or metadata.get("project_id") or workspace
                )
                memory_type = str(metadata.get("memory_type") or "")
                if memory_type in {"project_rule", "project_decision", "agent_experience"} and project_id != str(
                    filters["project_id"]
                ):
                    continue
            metadata_matches = True
            for field in ("adapter_id", "admission_status", "claim_state", "slot_id", "memory_type"):
                expected = filters.get(field)
                if expected is None:
                    continue
                values = set(expected) if isinstance(expected, list | tuple | set | frozenset) else {expected}
                actual = {
                    "adapter_id": metadata.get("source_adapter_id")
                    or dict(metadata.get("connect", {}) or {}).get("adapter_id"),
                    "admission_status": dict(metadata.get("admission", {}) or {}).get("decision"),
                    "claim_state": metadata.get("state") or metadata.get("claim_state"),
                    "slot_id": metadata.get("slot_id"),
                    "memory_type": metadata.get("memory_type"),
                }[field]
                if actual not in values:
                    metadata_matches = False
                    break
            if not metadata_matches:
                continue
            required_scopes = set(filters.get("applicability_scope_keys", []) or [])
            if required_scopes and not actual_scope_keys.issubset(required_scopes):
                continue
            text = " ".join([obj.title, content, json.dumps(obj.metadata, ensure_ascii=False)]).casefold()
            lexical_matches = lexical_match_count(query, text)
            lexical = lexical_relevance(query, text)
            identity = (
                1.0
                if any(
                    str(metadata.get(field, "")) == str(query).strip()
                    for field in {"scene_key", "action", "memory_anchor_uri"}
                )
                else 0.0
            )
            base_relevance = max(lexical, identity)
            if base_relevance <= 0:
                continue
            hotness = (obj.hotness + obj.semantic_hotness + obj.behavior_support_hotness) / 3.0
            score = max(float(lexical_matches), identity) + 0.05 * hotness
            hit_metadata = {
                "retrieval_scores": {
                    "lexical": lexical,
                    "vector": 0.0,
                    "identity": identity,
                    "base_relevance": base_relevance,
                    "hotness": hotness,
                    "score": score,
                },
            }
            hits.append(
                IndexHit(
                    uri=obj.uri,
                    score=score,
                    context_type=obj.context_type.value,
                    title=obj.title,
                    metadata=hit_metadata,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _lexical_terms(self, query: str) -> tuple[str, ...]:
        return lexical_terms(query)


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
        self.jobs.setdefault(job.job_id, job)

    def lease(self, queue_name: str, limit: int = 10, lease_seconds: int = 60) -> list[QueueJob]:
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
            self.jobs[job_id] = QueueJob(
                **{**job.__dict__, "status": "failed", "retry_count": job.retry_count + 1, "last_error": error}
            )

    def retry(self, job_id: str, error: str, *, max_retries: int = 3, retryable: bool = True) -> str:
        job = self.jobs[job_id]
        retry_count = job.retry_count + 1
        status = "pending" if retryable and retry_count < max_retries else "dead_letter"
        self.jobs[job_id] = QueueJob(
            **{
                **job.__dict__,
                "status": status,
                "leased_until": None,
                "retry_count": retry_count,
                "last_error": error[:500],
            }
        )
        return status

    def stats(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for job in self.jobs.values():
            result[job.status] = result.get(job.status, 0) + 1
        return result


class InMemoryLockStore:
    def __init__(self) -> None:
        self.locks: dict[str, tuple[str, int, datetime]] = {}
        self.fences: dict[str, int] = {}
        self._guard = threading.RLock()

    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken:
        with self._guard:
            now = datetime.now(timezone.utc)
            existing = self.locks.get(lock_key)
            if existing is not None and existing[2] > now:
                raise TimeoutError(f"Lock already held: {lock_key}")
            fence = self.fences.get(lock_key, 0) + 1
            self.fences[lock_key] = fence
            token = uuid.uuid4().hex
            self.locks[lock_key] = (
                token,
                fence,
                now + timedelta(seconds=max(1, ttl_seconds)),
            )
            return LockToken(lock_key=lock_key, token=token, fence=fence)

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> LockToken:
        with self._guard:
            self._assert_owned_unlocked(token)
            self.locks[token.lock_key] = (
                token.token,
                token.fence,
                datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds)),
            )
        return token

    def assert_owned(self, token: LockToken) -> None:
        with self._guard:
            self._assert_owned_unlocked(token)

    @contextmanager
    def fenced(self, tokens: Sequence[LockToken], ttl_seconds: int = 30) -> Iterator[None]:
        with self._guard:
            for token in tokens:
                self._assert_owned_unlocked(token)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
            for token in tokens:
                self.locks[token.lock_key] = (token.token, token.fence, expires_at)
            yield
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
            for token in tokens:
                self._assert_identity_unlocked(token)
                self.locks[token.lock_key] = (token.token, token.fence, expires_at)

    def release(self, token: LockToken) -> None:
        with self._guard:
            current = self.locks.get(token.lock_key)
            if current is not None and current[:2] == (token.token, token.fence):
                self.locks.pop(token.lock_key, None)

    def _assert_owned_unlocked(self, token: LockToken) -> None:
        self._assert_identity_unlocked(token)
        current = self.locks[token.lock_key]
        if current[2] <= datetime.now(timezone.utc):
            raise LockLostError(f"Lock lease lost: {token.lock_key}")

    def _assert_identity_unlocked(self, token: LockToken) -> None:
        current = self.locks.get(token.lock_key)
        if current is None or current[:2] != (token.token, token.fence):
            raise LockLostError(f"Lock lease lost: {token.lock_key}")
