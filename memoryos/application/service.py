"""Shared, explicit dependency boundary for application services."""

from __future__ import annotations

from typing import Any, Protocol


class ApplicationRuntime(Protocol):
    """Narrow runtime view consumed by orchestration services.

    The protocol is implemented by the in-process SDK facade today and can be
    implemented by another composition root without importing an API module
    into the application layer.
    """

    root: str
    mode: str
    tenant_id: str
    last_recall_trace_id: str
    source_store: Any
    index_store: Any
    relation_store: Any
    queue_store: Any
    lock_store: Any
    vector_store: Any
    embedding_provider: Any
    hybrid_search: Any
    reranker: Any
    committer: Any
    session_archive_store: Any
    session_commit_service: Any
    context_db: Any
    engine: Any
    executor: Any
    memory_document_store: Any
    memory_document_control_store: Any
    memory_document_revision_store: Any
    memory_document_planner: Any
    memory_document_committer: Any
    memory_document_consolidation_store: Any
    memory_document_consolidator: Any
    memory_document_projector: Any
    memory_document_scanner: Any
    memory_document_edit_worker: Any
    memory_document_scan_worker: Any
    memory_document_eraser: Any
    memory_command_service: Any
    memory_review_service: Any
    memory_projection_worker: Any
    readiness: Any

    def _require_ready(self) -> None: ...

    def _effective_tenant(self, caller: Any, explicit_tenant_id: str | None) -> str: ...

    def _process_memory_projections_or_raise(self) -> dict[str, list[str]]: ...

    def _require_exact_workspace(self, metadata: dict[str, Any], caller: Any, target: str) -> None: ...

    def _workspace_matches(self, metadata: dict[str, Any], project_id: str, caller: Any) -> bool: ...

    def _parse_connect_metadata(self, payload: dict[str, Any] | None) -> Any: ...

    def _require_predict_metadata(self, payload: dict[str, Any] | None) -> Any: ...

    def _require_process_observation_metadata(self, payload: dict[str, Any] | None) -> Any: ...

    def _connect_filters_from_metadata(self, connect_metadata: dict[str, Any] | None) -> dict[str, str]: ...

    def _parse_context_type(self, context_type: object) -> Any: ...

    def _context_assembler(self) -> Any: ...

    def _retrieval_orchestrator(self) -> Any: ...

    def _project_id_from_metadata(self, connect_metadata: dict[str, Any] | None) -> str: ...

    def _require_exact_read_visibility(self, uri: str, obj: Any, caller: Any) -> None: ...


class ApplicationService:
    """Explicit adapter from a composition-root runtime to service dependencies."""

    def __init__(self, runtime: ApplicationRuntime) -> None:
        self._runtime = runtime

    @property
    def root(self) -> str:
        return self._runtime.root

    @property
    def mode(self) -> str:
        return self._runtime.mode

    @property
    def tenant_id(self) -> str:
        return self._runtime.tenant_id

    @property
    def source_store(self) -> Any:
        return self._runtime.source_store

    @property
    def index_store(self) -> Any:
        return self._runtime.index_store

    @property
    def relation_store(self) -> Any:
        return self._runtime.relation_store

    @property
    def queue_store(self) -> Any:
        return self._runtime.queue_store

    @property
    def lock_store(self) -> Any:
        return self._runtime.lock_store

    @property
    def vector_store(self) -> Any:
        return self._runtime.vector_store

    @property
    def embedding_provider(self) -> Any:
        return self._runtime.embedding_provider

    @property
    def hybrid_search(self) -> Any:
        return self._runtime.hybrid_search

    @property
    def reranker(self) -> Any:
        return self._runtime.reranker

    @property
    def committer(self) -> Any:
        return self._runtime.committer

    @property
    def session_archive_store(self) -> Any:
        return self._runtime.session_archive_store

    @property
    def session_commit_service(self) -> Any:
        service = getattr(self._runtime, "session_commit_service", None)
        if service is not None:
            return service
        return self._runtime.context_db

    @property
    def context_db(self) -> Any:
        return self._runtime.context_db

    @property
    def engine(self) -> Any:
        return self._runtime.engine

    @property
    def executor(self) -> Any:
        return self._runtime.executor

    @property
    def memory_document_store(self) -> Any:
        return self._runtime.memory_document_store

    @property
    def memory_document_control_store(self) -> Any:
        return self._runtime.memory_document_control_store

    @property
    def memory_document_revision_store(self) -> Any:
        return self._runtime.memory_document_revision_store

    @property
    def memory_document_planner(self) -> Any:
        return self._runtime.memory_document_planner

    @property
    def memory_document_committer(self) -> Any:
        return self._runtime.memory_document_committer

    @property
    def memory_document_consolidation_store(self) -> Any:
        return self._runtime.memory_document_consolidation_store

    @property
    def memory_document_consolidator(self) -> Any:
        return self._runtime.memory_document_consolidator

    @property
    def memory_document_projector(self) -> Any:
        return self._runtime.memory_document_projector

    @property
    def memory_document_scanner(self) -> Any:
        return self._runtime.memory_document_scanner

    @property
    def memory_document_edit_worker(self) -> Any:
        return self._runtime.memory_document_edit_worker

    @property
    def memory_document_scan_worker(self) -> Any:
        return self._runtime.memory_document_scan_worker

    @property
    def memory_document_eraser(self) -> Any:
        return self._runtime.memory_document_eraser

    @property
    def memory_command_service(self) -> Any:
        return self._runtime.memory_command_service

    @property
    def memory_review_service(self) -> Any:
        return self._runtime.memory_review_service

    @property
    def memory_projection_worker(self) -> Any:
        return self._runtime.memory_projection_worker

    @property
    def readiness(self) -> Any:
        return self._runtime.readiness

    def _require_ready(self) -> None:
        self._runtime._require_ready()

    def _effective_tenant(self, caller: Any, explicit_tenant_id: str | None) -> str:
        return self._runtime._effective_tenant(caller, explicit_tenant_id)

    def _process_memory_projections_or_raise(self) -> dict[str, list[str]]:
        return self._runtime._process_memory_projections_or_raise()

    def _require_exact_workspace(self, metadata: dict[str, Any], caller: Any, target: str) -> None:
        self._runtime._require_exact_workspace(metadata, caller, target)

    def _workspace_matches(self, metadata: dict[str, Any], project_id: str, caller: Any) -> bool:
        return self._runtime._workspace_matches(metadata, project_id, caller)

    def _parse_connect_metadata(self, payload: dict[str, Any] | None) -> Any:
        return self._runtime._parse_connect_metadata(payload)

    def _require_predict_metadata(self, payload: dict[str, Any] | None) -> Any:
        return self._runtime._require_predict_metadata(payload)

    def _require_process_observation_metadata(self, payload: dict[str, Any] | None) -> Any:
        return self._runtime._require_process_observation_metadata(payload)

    def _connect_filters_from_metadata(self, connect_metadata: dict[str, Any] | None) -> dict[str, str]:
        return self._runtime._connect_filters_from_metadata(connect_metadata)

    def _parse_context_type(self, context_type: object) -> Any:
        return self._runtime._parse_context_type(context_type)

    def _context_assembler(self) -> Any:
        return self._runtime._context_assembler()

    def _retrieval_orchestrator(self) -> Any:
        return self._runtime._retrieval_orchestrator()

    def _project_id_from_metadata(self, connect_metadata: dict[str, Any] | None) -> str:
        return self._runtime._project_id_from_metadata(connect_metadata)

    def _require_exact_read_visibility(self, uri: str, obj: Any, caller: Any) -> None:
        self._runtime._require_exact_read_visibility(uri, obj, caller)

    @staticmethod
    def _uri_items(uris: list[str]) -> list[dict[str, str]]:
        return [{"uri": uri} for uri in dict.fromkeys(str(uri) for uri in uris if uri)]

    @staticmethod
    def _merge_uri_items(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                uri = str(item.get("uri", ""))
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                merged.append(dict(item))
        return merged


__all__ = ["ApplicationRuntime", "ApplicationService"]
