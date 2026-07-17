"""Session archive, health, and commit orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memoryos.application.context.query_service import ContextQueryService
from memoryos.application.service import ApplicationRuntime, ApplicationService
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.retrieval.query_plan import CanonicalResolutionMode, RetrievalOptions, RetrievalQueryIntent
from memoryos.contextdb.session.errors import EvidenceArchiveIntegrityError
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.security.context_projection import ContextProjectionSanitizer
from memoryos.security.trusted_context import (
    COMMIT_SESSION,
    READ_CONTEXT,
    TrustedRequestContext,
    sanitize_ingress_messages,
    sanitize_ingress_tool_results,
    sanitize_session_provenance,
    sanitize_session_scope,
)


def _stable_session_commit_task_id(payload: dict[str, Any]) -> str:
    return f"session_commit_{stable_hash(payload, length=32)}"


class SessionApplicationService(ApplicationService):
    def __init__(self, runtime: ApplicationRuntime, context_queries: ContextQueryService) -> None:
        super().__init__(runtime)
        self._context_queries = context_queries

    def archive_read(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        if caller is not None:
            caller.require(READ_CONTEXT)
            if ContextURI.parse(archive_uri).user_id != caller.user_id:
                raise FileNotFoundError(archive_uri)
        if not self.session_archive_store.archive_exists(archive_uri, tenant_id=tenant_id):
            raise FileNotFoundError(archive_uri)
        archive = self.session_archive_store.read_archive(archive_uri, tenant_id=tenant_id)
        if caller is not None:
            if archive.user_id != caller.user_id:
                raise FileNotFoundError(archive_uri)
            self._require_exact_workspace(dict(archive.metadata or {}), caller, archive_uri)
        return {"archive": archive.manifest(), "messages": archive.messages, "tool_results": archive.tool_results}

    def archive_search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 20,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
        project_id: str = "",
        search_context: Any | None = None,
        archive_read: Any | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("archive search limit must be between 1 and 100")
        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        if caller is not None:
            caller.require(READ_CONTEXT)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            project_id = caller.bind_read_workspace(project_id)
        expanded_limit = min(200, max(limit, limit * 5))
        search = search_context or self._context_queries.search_context
        contexts = search(
            query,
            options=RetrievalOptions(
                context_types=(ContextType.SESSION,),
                tenant_id=tenant_id,
                owner_user_id=user_id,
                workspace_ids=((project_id,) if project_id else ()),
                query_intent=RetrievalQueryIntent.OPEN_RECALL,
                canonical_resolution_mode=CanonicalResolutionMode.DISABLED,
                candidate_limit=max(100, expanded_limit),
                final_limit=expanded_limit,
                metadata_filters={"minimum_lexical_relevance": 1.0},
            ),
            user_id=user_id,
            project_id=project_id,
            tenant_id=tenant_id,
            caller=caller,
        )
        results: list[dict[str, Any]] = []
        seen_archives: set[str] = set()
        for item in contexts:
            metadata = dict(item.get("metadata", {}) or {})
            archive_uri = str(metadata.get("archive_uri") or item.get("source_uri") or "")
            if not archive_uri or archive_uri in seen_archives:
                continue
            seen_archives.add(archive_uri)
            # Compatibility reads are exact and candidate-bounded: they verify
            # the immutable archive evidence without restoring the former
            # recursive directory scan.
            try:
                reader = archive_read or self.archive_read
                archive_payload = reader(archive_uri, tenant_id=tenant_id, caller=caller)
            except EvidenceArchiveIntegrityError as exc:
                raise EvidenceArchiveIntegrityError(f"archive commit head evidence is invalid: {exc}") from exc
            archive_manifest = dict(archive_payload.get("archive", {}) or {})
            # The unified Catalog already performed lexical/semantic matching
            # over its sanitized projection.  Do not restore the legacy second
            # Python substring pass over full immutable archive contents.
            catalog_preview = str(item.get("content") or item.get("text") or "")
            safe_preview = (
                ContextProjectionSanitizer()
                .sanitize(
                    title=str(item.get("title") or ""),
                    l0_text="",
                    l1_text=catalog_preview,
                    metadata={},
                    source_kind=str(metadata.get("source_kind") or "session"),
                )
                .l1_text
            )
            session_id = str(
                metadata.get("session_id")
                or archive_manifest.get("session_id")
                or archive_uri.rstrip("/").rsplit("/", 1)[-1]
            )
            preview = safe_preview[:500]
            results.append(
                {
                    **dict(item),
                    "archive_uri": archive_uri,
                    "session_id": session_id,
                    "preview": preview,
                }
            )
            if len(results) >= limit:
                break
        return results


    def health(self) -> dict[str, Any]:
        artifact_root = Path(self.root) if self.tenant_id == "default" else Path(self.root) / "tenants" / self.tenant_id
        heartbeat = artifact_root / "system" / "worker-health.json"
        worker_health: dict[str, Any] = {}
        if heartbeat.exists():
            try:
                payload = json.loads(heartbeat.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    worker_health = {
                        key: payload.get(key)
                        for key in (
                            "status",
                            "updated_at",
                            "processed",
                            "succeeded",
                            "failed",
                            "retried",
                            "dead_letter",
                            "quarantine",
                            "last_error",
                        )
                    }
            except (OSError, UnicodeError, json.JSONDecodeError):
                worker_health = {"status": "failed", "last_error": "InvalidWorkerHealth"}
        queue_stats: dict[str, int] = getattr(self.queue_store, "stats", lambda: {})()
        runtime = self.readiness.snapshot()
        runtime_ready = bool(runtime.get("ready"))
        worker_status = str(worker_health.get("status") or "stopped")

        def failure_count(payload: dict[str, Any], key: str) -> int:
            """Treat malformed health counters as evidence of degradation."""

            try:
                value = int(payload.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 1
            return value if value >= 0 else 1

        derived_unhealthy = bool(
            worker_status in {"degraded", "failed"}
            or failure_count(worker_health, "dead_letter") > 0
            or failure_count(worker_health, "quarantine") > 0
            or failure_count(queue_stats, "dead_letter") > 0
            or failure_count(queue_stats, "quarantine") > 0
        )
        overall_status = "not_ready" if not runtime_ready else "degraded" if derived_unhealthy else "ready"
        operational_state = "ready" if runtime_ready else "not_ready"

        def optional_state(configured: object) -> str:
            if configured is None:
                return "disabled"
            return operational_state

        return {
            "status": overall_status,
            "runtime": runtime,
            "source_store": operational_state,
            "index_store": operational_state,
            "queue_store": operational_state,
            "worker": worker_status,
            "worker_health": worker_health,
            "memory_extractor": optional_state(self.session_commit_service.memory_planner.extractor),
            "embedding": optional_state(self.embedding_provider),
            "vector_store": optional_state(self.vector_store),
            "reranker": optional_state(self.reranker),
            "http_server": operational_state if self.mode == "server" else "disabled",
            "queue": queue_stats,
            "degraded_features": [
                name
                for name, value in (
                    ("embedding", self.embedding_provider),
                    ("vector_store", self.vector_store),
                    ("reranker", self.reranker),
                )
                if value is None
            ],
        }

    def commit_agent_session(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]] | None = None,
        used_contexts: list[dict[str, Any]] | None = None,
        used_skills: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        connect_metadata: dict[str, Any] | None = None,
        async_commit: bool = True,
        project_id: str = "",
        session_key: str = "",
        scope: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> Any:
        """归档并提交一次 Agent 会话。"""

        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        if caller is not None:
            caller.require(COMMIT_SESSION)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
        metadata = self._parse_connect_metadata(connect_metadata)
        stable_session_id = session_key or session_id
        archive_uri = f"memoryos://user/{user_id}/sessions/history/{stable_session_id}"
        normalized_metadata = metadata.to_dict()
        normalized_project_id = project_id or self._project_id_from_metadata(connect_metadata)
        if caller is not None:
            normalized_project_id = caller.bind_write_workspace(normalized_project_id)
        if caller is None:
            normalized_scope = {
                **dict(scope or {}),
                "user_id": user_id,
                "project_id": normalized_project_id,
                "session_key": stable_session_id,
                "tenant_id": tenant_id,
            }
            normalized_provenance = {"native_session_id": session_id, **dict(provenance or {})}
            normalized_messages = messages or []
            normalized_tool_results = tool_results or []
        else:
            normalized_scope = sanitize_session_scope(
                scope,
                caller,
                project_id=normalized_project_id,
                session_key=stable_session_id,
            )
            normalized_scope["tenant_id"] = tenant_id
            normalized_provenance = sanitize_session_provenance(
                provenance,
                caller,
                native_session_id=session_id,
            )
            normalized_messages = sanitize_ingress_messages(messages, caller)
            normalized_tool_results = sanitize_ingress_tool_results(tool_results, caller)
        task_id = _stable_session_commit_task_id(
            {
                "user_id": user_id,
                "session_id": session_id,
                "archive_uri": archive_uri,
                "messages": normalized_messages,
                "used_contexts": used_contexts or [],
                "used_skills": used_skills or [],
                "tool_results": normalized_tool_results,
                "metadata": {
                    "connect": normalized_metadata,
                    "scope": normalized_scope,
                    "provenance": normalized_provenance,
                },
            }
        )
        archive = SessionArchive(
            user_id=user_id,
            session_id=session_id,
            archive_uri=archive_uri,
            messages=normalized_messages,
            used_contexts=used_contexts or [],
            used_skills=used_skills or [],
            tool_results=normalized_tool_results,
            metadata={
                "connect": normalized_metadata,
                "scope": normalized_scope,
                "provenance": normalized_provenance,
                "project_id": normalized_scope.get("project_id", ""),
                "tenant_id": normalized_scope.get("tenant_id", "default"),
            },
            task_id=task_id,
        )
        archive_tenant = str(normalized_scope.get("tenant_id") or "default")
        archive_store = getattr(self, "session_archive_store", None)
        if archive_store is not None and archive_store.archive_exists(archive_uri, tenant_id=archive_tenant):
            existing = archive_store.read_archive(archive_uri, tenant_id=archive_tenant)
            if existing.task_id == task_id:
                archive = existing
        return self.session_commit_service.commit_session(archive, async_commit=async_commit)



__all__ = ["SessionApplicationService"]
