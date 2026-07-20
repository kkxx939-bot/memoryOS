"""候选内容回源、校验与 L2 加载。"""

from __future__ import annotations

import hmac
import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from foundation.ids import stable_hash
from infrastructure.context.layers.memory_document_overlay import MemoryDocumentContextOverlay
from infrastructure.context.retrieval.fusion import RetrievalCandidate
from infrastructure.context.retrieval.query_plan import RetrievalQueryPlan
from infrastructure.context.selection import ContextSelector
from infrastructure.store.contracts.queue import QueueJob, QueueStore
from infrastructure.store.contracts.session_archive import SessionArchiveStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from memory.ports.document_store import DocumentConflictError, DocumentNotFoundError, DocumentUnsafeError
from sanitization.context_projection import (
    ContextProjectionSanitizationError,
    ContextProjectionSanitizer,
)


class ContextHydrator:
    """在固定读取预算内把召回候选还原为可返回内容。"""

    def __init__(
        self,
        *,
        source_store: SourceStore | None,
        session_archive_store: SessionArchiveStore | None,
        queue_store: QueueStore | None,
        selector: ContextSelector,
        sanitizer: ContextProjectionSanitizer,
        document_overlay: MemoryDocumentContextOverlay | None,
    ) -> None:
        self.source_store = source_store
        self.session_archive_store = session_archive_store
        self.queue_store = queue_store
        self.selector = selector
        self.sanitizer = sanitizer
        self.document_overlay = document_overlay

    def hydrate(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
        source_read_budget: int,
    ) -> tuple[
        tuple[RetrievalCandidate, ...],
        int,
        tuple[str, ...],
        tuple[dict[str, Any], ...],
        int,
    ]:
        result: list[RetrievalCandidate] = []
        dropped: list[dict[str, Any]] = []
        degraded_modes: list[str] = []
        reads = 0
        memory_validated = 0
        l2_resource_keys = frozenset(self.selector.l2_hydration_record_keys(candidates, plan=plan))
        memory_l2_remaining = self.selector.policy.max_l2_items
        for item in candidates:
            if item.document_id:
                if reads >= source_read_budget:
                    degraded_modes.append("memory_source_read_bound")
                    dropped.append(self._drop(item, "memory_source_read_bound"))
                    continue
                reads += 1
                hydrated, drop = self._memory_document(
                    item,
                    plan=plan,
                    include_l2=memory_l2_remaining > 0,
                )
                if drop is not None:
                    dropped.append(drop)
                    self._schedule_document_rescan(item, plan=plan)
                    continue
                memory_validated += 1
                if hydrated.text:
                    memory_l2_remaining -= 1
                result.append(hydrated)
                continue
            if item.record_key in l2_resource_keys:
                hydrated, used_reads, mode = self._resource_l2(
                    item,
                    plan=plan,
                    source_read_budget=max(0, source_read_budget - reads),
                )
                reads += used_reads
                if mode:
                    degraded_modes.append(mode)
                result.append(hydrated)
                continue
            if item.context_type == ContextType.SESSION.value:
                if item.record_kind in {"session_root", "semantic_segment", "session_l1"} and item.l2_uri:
                    if reads < source_read_budget:
                        reads += 1
                        full = self._session_l2(item, plan=plan)
                        if full is not None:
                            result.append(
                                replace(
                                    item,
                                    text=full,
                                    metadata={
                                        **dict(item.metadata),
                                        "source_validation_status": "archive_digest_verified",
                                    },
                                )
                            )
                            continue
                result.append(item)
                continue
            if item.l0_text or item.l1_text:
                result.append(item)
                continue
            if reads >= source_read_budget:
                degraded_modes.append("source_read_bound")
                result.append(self._mark_degraded(item, "source_read_bound"))
                continue
            ordinary_hydrated, used_reads = self._ordinary(
                item,
                plan=plan,
                remaining=source_read_budget - reads,
            )
            reads += used_reads
            if ordinary_hydrated is not None:
                result.append(ordinary_hydrated)
        return (
            tuple(result),
            reads,
            tuple(dict.fromkeys(degraded_modes)),
            tuple(dropped),
            memory_validated,
        )

    def _memory_document(
        self,
        item: RetrievalCandidate,
        *,
        plan: RetrievalQueryPlan,
        include_l2: bool,
    ) -> tuple[RetrievalCandidate, dict[str, Any] | None]:
        relative_path = str(item.metadata.get("relative_path") or "")
        if self.document_overlay is None or not relative_path or not item.source_digest:
            return item, self._drop(item, "memory_source_unavailable")
        try:
            view = self.document_overlay.read(
                tenant_id=str(plan.tenant_id or "default"),
                owner_user_id=str(plan.owner_user_id or item.owner_user_id),
                document_uri=item.source_uri or item.uri,
                relative_path=relative_path,
                expected_source_digest=item.source_digest,
            )
        except (DocumentConflictError, DocumentNotFoundError, DocumentUnsafeError, PermissionError, ValueError):
            return item, self._drop(item, "stale_memory_document_projection")
        try:
            safe = self.sanitizer.sanitize(
                title=item.title,
                l0_text=item.l0_text,
                l1_text=view.markdown if include_l2 else item.l1_text,
                metadata=dict(item.metadata),
                source_kind="memory_document",
            )
        except ContextProjectionSanitizationError:
            return item, self._drop(item, "memory_document_sanitization_failed")
        return (
            replace(
                item,
                text=safe.l1_text if include_l2 else "",
                l0_text=safe.l0_text,
                l1_text=item.l1_text or safe.l1_text,
                metadata={
                    **safe.metadata,
                    "relative_path": relative_path,
                    "source_validation_status": "live_digest_verified",
                },
            ),
            None,
        )

    def _ordinary(
        self,
        item: RetrievalCandidate,
        *,
        plan: RetrievalQueryPlan,
        remaining: int,
    ) -> tuple[RetrievalCandidate | None, int]:
        if self.source_store is None or remaining < 1:
            return item, 0
        reads = 1
        try:
            obj = self.source_store.read_object(item.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, RuntimeError, ValueError):
            return None, reads
        if (
            obj.lifecycle_state != LifecycleState.ACTIVE
            or str(obj.tenant_id or "default") != str(plan.tenant_id or "default")
            or (plan.owner_user_id and str(obj.owner_user_id or "") not in {"", plan.owner_user_id})
        ):
            return None, reads
        content = obj.title
        if remaining >= 2:
            reads += 1
            try:
                content = self.source_store.read_content(obj.layers.l2_uri or obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                pass
        safe = self.sanitizer.sanitize(
            title=obj.title,
            l0_text=obj.title,
            l1_text=content,
            metadata=obj.metadata,
            source_kind=str(obj.metadata.get("source_kind") or "context"),
        )
        return (
            replace(
                item,
                title=safe.title,
                l0_text=safe.l0_text,
                l1_text=safe.l1_text,
                metadata={
                    **dict(item.metadata),
                    **safe.metadata,
                    "source_validation_status": "source_read",
                },
            ),
            reads,
        )

    def _resource_l2(
        self,
        item: RetrievalCandidate,
        *,
        plan: RetrievalQueryPlan,
        source_read_budget: int,
    ) -> tuple[RetrievalCandidate, int, str]:
        if source_read_budget < 2:
            mode = "l2_source_read_bound"
            return self._mark_degraded(item, mode), 0, mode
        if self.source_store is None:
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), 0, mode
        reads = 1
        try:
            obj = self.source_store.read_object(item.uri)
            source_record = CatalogRecord.from_context_object(obj)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, RuntimeError, ValueError):
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), reads, mode
        if (
            obj.uri != item.uri
            or obj.context_type != ContextType.RESOURCE
            or obj.lifecycle_state != LifecycleState.ACTIVE
            or source_record.record_kind != CatalogRecordKind.CONTEXT.value
            or str(obj.tenant_id or "default") != str(plan.tenant_id or "default")
            or source_record.owner_user_id != item.owner_user_id
            or source_record.workspace_id != item.workspace_id
            or source_record.source_uri != (item.source_uri or item.uri)
        ):
            mode = "l2_source_authority_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        source_l2_uri = str(obj.layers.l2_uri or source_record.source_uri or obj.uri)
        if source_l2_uri != str(item.l2_uri or item.source_uri or item.uri):
            mode = "l2_source_authority_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        reads += 1
        try:
            content = self.source_store.read_content(source_l2_uri)
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            PermissionError,
            RuntimeError,
            UnicodeError,
            ValueError,
        ):
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), reads, mode
        actual_digest = self.sanitizer.digest(content or obj.to_dict())
        if not item.source_digest or not hmac.compare_digest(item.source_digest, actual_digest):
            mode = "l2_source_revision_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        try:
            safe = self.sanitizer.sanitize(
                title=item.title,
                l0_text=item.l0_text,
                l1_text=content,
                metadata={},
                source_kind=ContextType.RESOURCE.value,
            )
        except ContextProjectionSanitizationError:
            mode = "l2_sanitization_failed"
            return self._mark_degraded(item, mode), reads, mode
        return (
            replace(
                item,
                text=safe.l1_text,
                metadata={
                    **dict(item.metadata),
                    "source_validation_status": "source_digest_verified",
                },
            ),
            reads,
            "",
        )

    def _session_l2(self, item: RetrievalCandidate, *, plan: RetrievalQueryPlan) -> str | None:
        if self.session_archive_store is None:
            return None
        archive_uri = str(item.metadata.get("archive_uri") or "")
        if not archive_uri:
            return None
        try:
            archive = self.session_archive_store.read_archive(
                archive_uri,
                tenant_id=str(plan.tenant_id or "default"),
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError):
            return None
        if plan.owner_user_id and archive.user_id != plan.owner_user_id:
            return None
        manifest_digest = str(item.manifest_digest or item.metadata.get("manifest_digest") or "")
        if manifest_digest and archive.manifest_digest != manifest_digest:
            return None
        payload = {
            "messages": archive.messages,
            "tool_results": archive.tool_results,
            "observations": archive.observations,
            "action_results": archive.action_results,
            "used_contexts": archive.used_contexts,
            "used_skills": archive.used_skills,
        }
        safe = self.sanitizer.sanitize(
            title=item.title,
            l0_text=item.l0_text,
            l1_text=json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
            metadata={},
            source_kind="session",
        )
        return safe.l1_text

    def _schedule_document_rescan(self, item: RetrievalCandidate, *, plan: RetrievalQueryPlan) -> None:
        if self.queue_store is None:
            return
        self.queue_store.enqueue(
            QueueJob(
                job_id=f"memory_rescan_{stable_hash((plan.tenant_id, item.owner_user_id, item.document_id, item.source_digest), 32)}",
                queue_name="memory_document_scan",
                action="rescan",
                target_uri=item.source_uri or item.uri,
                payload={
                    "tenant_id": str(plan.tenant_id or "default"),
                    "owner_user_id": str(plan.owner_user_id or item.owner_user_id),
                    "document_id": item.document_id,
                    "observed_source_digest": item.source_digest,
                },
            )
        )

    @staticmethod
    def _drop(item: RetrievalCandidate, reason: str) -> dict[str, Any]:
        return {
            "record_key": item.record_key,
            "uri": item.uri,
            "source_uri": item.source_uri or item.uri,
            "drop_reason": reason,
            "source_validation_status": "stale" if reason.startswith("stale") else "unavailable",
        }

    @staticmethod
    def _mark_degraded(item: RetrievalCandidate, mode: str) -> RetrievalCandidate:
        existing = str(item.metadata.get("degraded_mode") or "")
        modes = tuple(dict.fromkeys(part for part in (*existing.split(","), mode) if part))
        return replace(item, metadata={**dict(item.metadata), "degraded_mode": ",".join(modes)})


__all__ = ["ContextHydrator"]
