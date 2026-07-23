"""消费 Markdown Memory 变更事件并协调可重建投影。"""

from __future__ import annotations

from infrastructure.context.projection.memory_document import MemoryDocumentProjector
from infrastructure.context.retrieval.embedding import EmbeddingProvider
from infrastructure.store.contracts.index import MemoryDocumentProjectionStore
from infrastructure.store.contracts.queue import QueueJob, QueueStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.vector import VectorStore
from infrastructure.store.memory.control_store import (
    DocumentDeletionStatus,
    MemoryDocumentControlStore,
)
from infrastructure.store.memory.erasure_store import MemoryDocumentEraseStore
from memory.core.model import DocumentEditKind, ManagedDocument, PresentPath
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import DocumentConflictError, MemoryDocumentStore
from memory.ports.erase import DocumentErasedError
from memory.worker.projection.catalog import ProjectionCatalogMixin
from memory.worker.projection.erasure import ProjectionErasureMixin
from memory.worker.projection.event import parse_projection_job, projection_deletion_digest
from memory.worker.projection.model import MemoryProjectionRun, coerce_persisted_int
from memory.worker.projection.publication import ProjectionPublicationMixin


class MemoryDocumentProjectionWorker(
    ProjectionPublicationMixin,
    ProjectionErasureMixin,
    ProjectionCatalogMixin,
):
    """协调队列、事实源校验、重建和各投影子职责。"""

    queue_name = "memory_projection"

    def __init__(
        self,
        document_store: MemoryDocumentStore,
        control_store: MemoryDocumentControlStore,
        catalog_store: MemoryDocumentProjectionStore,
        queue_store: QueueStore,
        *,
        projector: MemoryDocumentProjector | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        relation_store: RelationStore | None = None,
        erasure_store: MemoryDocumentEraseStore | None = None,
        lease_owner: str = "memory-document-projector",
    ) -> None:
        self.document_store = document_store
        self.control_store = control_store
        self.catalog_store = catalog_store
        self.queue_store = queue_store
        self.projector = projector or MemoryDocumentProjector()
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.relation_store = relation_store
        self.erasure_store = erasure_store or MemoryDocumentEraseStore(control_store.root)
        self.lease_owner = lease_owner

    def process_pending(self, *, limit: int = 10, lease_seconds: int = 60) -> MemoryProjectionRun:
        jobs = self.queue_store.lease(
            self.queue_name,
            lease_owner=self.lease_owner,
            limit=limit,
            lease_seconds=lease_seconds,
        )
        processed: list[str] = []
        stale: list[str] = []
        failed: list[str] = []
        for job in jobs:
            try:
                outcome = self.process_job(job)
                self.queue_store.ack(job)
                (stale if outcome == "stale" else processed).append(job.job_id)
            except (DocumentErasedError, DocumentConflictError, ValueError, RuntimeError, OSError) as exc:
                failed.append(job.job_id)
                self.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=5,
                    retryable=not isinstance(exc, DocumentErasedError | ValueError),
                )
        return MemoryProjectionRun(tuple(processed), tuple(stale), tuple(failed))

    def drain_until_quiescent(
        self,
        *,
        max_rounds: int = 1_000,
        batch_size: int = 100,
        lease_seconds: int = 300,
    ) -> dict[str, int]:
        """在启动或修复期间有界清空本 Worker 的投影队列。"""

        processed = stale = 0
        for _ in range(max_rounds):
            stats = self.queue_store.stats(queue_name=self.queue_name)
            if not int(stats.get("pending", 0) or 0):
                break
            run = self.process_pending(limit=batch_size, lease_seconds=lease_seconds)
            processed += len(run.processed)
            stale += len(run.stale)
        stats = self.queue_store.stats(queue_name=self.queue_name)
        if any(int(stats.get(name, 0) or 0) for name in ("pending", "leased", "dead_letter", "quarantine")):
            raise RuntimeError("memory projection queue is not quiescent after startup recovery")
        return {"processed": processed, "stale": stale, **{key: int(value) for key, value in stats.items()}}

    def process_job(self, job: QueueJob) -> str:
        payload = parse_projection_job(job)
        tenant = str(payload["tenant_id"])
        owner = str(payload["owner_user_id"])
        document_id = str(payload["document_id"])
        generation = coerce_persisted_int(payload["projection_generation"])
        edit_kind = DocumentEditKind(str(payload["edit_kind"]))
        barrier = self.control_store.load_publication_barrier(tenant, owner, document_id)
        if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
            with self.erasure_store.document_lock(tenant, owner, document_id):
                durable = self._fence_durable_erasure(tenant, owner, document_id)
                if durable is None:
                    self._mirror_barrier(barrier)
            return "stale"
        self.erasure_store.assert_projection_allowed(
            tenant,
            owner,
            document_id,
            projection_generation=generation,
        )
        serving_state = self._projection_state(tenant, owner, document_id)
        control = self.control_store.load_control(tenant, owner, document_id)
        if control is None:
            raise RuntimeError("projection event has no durable document control record")
        if generation < control.projection_generation:
            return "stale"
        if generation > control.projection_generation:
            raise RuntimeError("projection event is newer than its durable control record")
        if str(payload["event_id"]) != control.last_event_id:
            raise ValueError("projection event identity differs from durable control")
        expected_after = str(payload["after_raw_digest"])
        if barrier is None and serving_state and str(serving_state.get("deletion_status") or ""):
            raise RuntimeError("serving deletion state has no protected control-store barrier")
        if barrier is not None and generation <= barrier.deletion_generation:
            if (
                edit_kind is DocumentEditKind.DELETE
                and generation == barrier.deletion_generation
                and control.status == "deleted"
                and not expected_after
                and projection_deletion_digest(payload) == barrier.deletion_event_digest
            ):
                self._mirror_barrier(barrier)
                return "processed"
            return "stale"
        if control.status == "present":
            if expected_after != control.raw_sha256:
                raise ValueError("projection event digest differs from durable control")
            live_state = self.document_store.read_state(
                tenant,
                owner,
                control.relative_path,
            )
            if not isinstance(live_state, PresentPath) or live_state.raw_sha256 != control.raw_sha256:
                # 投影事件已经不再对应精确事实。缺失或变化的文件只能由稳定性
                # Scanner 协调；可重建任务在此标记为过期，避免启动流程伪造
                # 删除权限或陷入重试循环。
                return "stale"
            restored_generation = 0
            if barrier is not None:
                if (
                    barrier.status is not DocumentDeletionStatus.SOFT_FORGOTTEN
                    or control.restored_from_deletion_generation != barrier.deletion_generation
                    or control.projection_generation <= barrier.deletion_generation
                ):
                    raise RuntimeError("live control is not authorized past its deletion barrier")
                restored_generation = barrier.deletion_generation
                if serving_state is None or str(serving_state.get("deletion_status") or ""):
                    self._mirror_barrier(barrier)
            self._publish_live(
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=document_id,
                relative_path=control.relative_path,
                source_digest=control.raw_sha256,
                document_revision=control.logical_revision,
                projection_generation=generation,
                restored_from_deletion_generation=restored_generation,
            )
        else:
            if expected_after:
                raise ValueError("deleted projection event cannot claim live source bytes")
            if barrier is None:
                raise RuntimeError("deleted projection event has no protected publication barrier")
            self._mirror_barrier(barrier)
        return "processed"

    def rebuild_owner(self, tenant_id: str, owner_user_id: str) -> dict[str, int]:
        """以 live 事实源有界协调一个用户目录，不信任 watcher 提示。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        scan = self.document_store.full_scan(tenant, owner)
        if not scan.complete or scan.errors or scan.unsafe_paths:
            raise RuntimeError("memory document rebuild requires one complete safe full scan")
        projected = 0
        skipped = 0
        deleted = 0
        pending_missing = 0
        live_document_ids = {item.document_id for item in scan.managed}
        barriers = {barrier.document_id: barrier for barrier in self.control_store.publication_barriers(tenant, owner)}
        for document_id, durable_barrier in barriers.items():
            if document_id not in live_document_ids:
                with self.erasure_store.document_lock(tenant, owner, document_id):
                    durable = self._fence_durable_erasure(tenant, owner, document_id)
                    if durable is None:
                        self._mirror_barrier(durable_barrier)
                control = self.control_store.load_control(tenant, owner, document_id)
                restored_after_barrier = bool(
                    durable_barrier.status is DocumentDeletionStatus.SOFT_FORGOTTEN
                    and control is not None
                    and control.status == "present"
                    and control.restored_from_deletion_generation == durable_barrier.deletion_generation
                    and control.projection_generation > durable_barrier.deletion_generation
                )
                deleted += int(not restored_after_barrier)
        for registration in scan.registrations:
            if not isinstance(registration, ManagedDocument):
                raise RuntimeError("memory document rebuild found an unmanaged or duplicate identity")
            with self.erasure_store.document_lock(tenant, owner, registration.document_id):
                erasure_barrier = self._fence_durable_erasure(
                    tenant,
                    owner,
                    registration.document_id,
                )
            if erasure_barrier is not None:
                barriers[registration.document_id] = erasure_barrier
                skipped += 1
                continue
            control = self.control_store.load_control(tenant, owner, registration.document_id)
            barrier = barriers.get(registration.document_id)
            state = self._projection_state(tenant, owner, registration.document_id)
            if barrier is None and state is not None and str(state.get("deletion_status") or ""):
                # 并发 Scanner 或 Eraser 可能在本次重建取得 owner 快照之后发布
                # 受保护屏障；判定 Serving tombstone 游离前必须重读耐久权限。
                barrier = self.control_store.load_publication_barrier(
                    tenant,
                    owner,
                    registration.document_id,
                )
                if barrier is not None:
                    barriers[registration.document_id] = barrier
            restored_from_deletion_generation = 0
            if barrier is not None:
                if not self._control_authorizes_restored_registration(
                    control,
                    barrier,
                    registration,
                ):
                    self._mirror_barrier(barrier)
                    skipped += 1
                    continue
                restored_from_deletion_generation = barrier.deletion_generation
                if state is None or str(state.get("deletion_status") or ""):
                    self._mirror_barrier(barrier)
                    state = self._projection_state(tenant, owner, registration.document_id)
            elif state is not None and str(state.get("deletion_status") or ""):
                raise RuntimeError("serving deletion state has no protected control-store barrier")
            deletion_status = str((state or {}).get("deletion_status") or "")
            if deletion_status and not restored_from_deletion_generation:
                skipped += 1
                continue
            current_generation = coerce_persisted_int((state or {}).get("projection_generation") or 0)
            current_digest = str((state or {}).get("source_digest") or "")
            current_path = str((state or {}).get("relative_path") or "")
            if current_digest == registration.raw_sha256 and current_path == registration.relative_path:
                skipped += 1
                continue
            generation = current_generation + 1
            if (
                control is not None
                and control.status == "present"
                and control.relative_path == registration.relative_path
                and control.raw_sha256 == registration.raw_sha256
            ):
                generation = max(generation, control.projection_generation)
            revision = int(control.logical_revision if control is not None else generation)
            self._publish_live(
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=registration.document_id,
                relative_path=registration.relative_path,
                source_digest=registration.raw_sha256,
                document_revision=max(1, revision),
                projection_generation=generation,
                expected_previous_generation=current_generation,
                restored_from_deletion_generation=restored_from_deletion_generation,
            )
            projected += 1
        # 链接目标的排序可能晚于来源文档，因此必须先发布全部 live 文档记录，
        # 再统一重建文档链接。
        records_by_id = {record.document_id: record for record in self._owner_document_records(tenant, owner)}
        for registration in scan.managed:
            record = records_by_id.get(registration.document_id)
            if record is None:
                continue
            self._refresh_live_document_links(tenant, owner, registration, record)
        for record in self._owner_document_records(tenant, owner):
            if record.document_id in live_document_ids:
                continue
            state = self._projection_state(tenant, owner, record.document_id) or {}
            if str(state.get("deletion_status") or ""):
                continue
            # 单次全量扫描缺失不构成删除权限；只有稳定性 Scanner/Committer
            # 可以持久化删除事件和允许 Serving tombstone 的发布屏障。
            pending_missing += 1
        result = {
            "projected": projected,
            "skipped": skipped,
            "deleted": deleted,
            "documents": len(scan.managed),
        }
        if pending_missing:
            result["pending_missing"] = pending_missing
        return result

    def verify_owner(self, tenant_id: str, owner_user_id: str) -> dict[str, int]:
        """证明每个 live 注册记录都有完全匹配的 Serving generation。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        scan = self.document_store.full_scan(tenant, owner)
        if not scan.complete or scan.errors or scan.unsafe_paths:
            raise RuntimeError("memory document verification requires one complete safe full scan")
        projected = {record.document_id: record for record in self._owner_document_records(tenant, owner)}
        live = {item.document_id: item for item in scan.managed}
        for document_id, registration in live.items():
            state = self._projection_state(tenant, owner, document_id)
            if state is None or str(state.get("deletion_status") or ""):
                raise RuntimeError("live Markdown is blocked or missing from serving projection state")
            record = projected.get(document_id)
            if (
                record is None
                or record.source_digest != registration.raw_sha256
                or str(record.metadata.get("relative_path") or "") != registration.relative_path
                or record.projection_generation != coerce_persisted_int(state.get("projection_generation") or 0)
            ):
                raise RuntimeError("live Markdown and serving Catalog projection disagree")
        stale = set(projected) - set(live)
        # 缺失的 live bytes 会在事实源回读阶段被排除，但稳定窗口内的暂时缺失
        # 不能让启动进入 NOT_READY，也不能在重建链路中合成删除权限。
        result = {
            "verified": len(live),
            "projected": len(projected),
        }
        if stale:
            result["pending_missing"] = len(stale)
            result["degraded"] = len(stale)
        return result
