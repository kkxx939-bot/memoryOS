"""Session 投影前沿、归档重建和提交组恢复。"""

from __future__ import annotations

from infrastructure.store.session.commit_group import (
    CommitGroupStatus,
)
from pre.session import SessionArchive
from runtime.session.commit_model import (
    SessionCommitResult,
)
from runtime.session.commit_types import _SessionCommitState


class _SessionCommitRecovery(_SessionCommitState):
    def recover_session_projection_frontier(self, *, batch_size: int = 256) -> dict[str, int]:
        """重放普通 Session 投影及其精确提交任务。"""

        if self.session_projector is None or not self.projection_journal.enabled:
            return {"projected": 0, "abandoned": 0, "failed": 0}
        self._require_runtime_ready()
        tenant_id = str(self.archive_store.tenant_id)
        maximum = max(1, min(int(batch_size), 1_000))
        counts = {"projected": 0, "abandoned": 0, "failed": 0}
        after = ""
        while True:
            entries = self.projection_journal.pending(
                tenant_id=tenant_id,
                after_archive_uri=after,
                limit=maximum,
            )
            if not entries:
                break
            for entry in entries:
                after = entry.archive_uri
                if not self.archive_store.archive_exists(entry.archive_uri, tenant_id=tenant_id):
                    self.projection_journal.mark(entry, status="ABANDONED", error="archive_missing")
                    counts["abandoned"] += 1
                    continue
                try:
                    archive = self.archive_store.read_archive(
                        entry.archive_uri,
                        tenant_id=tenant_id,
                        manifest_digest=entry.manifest_digest or None,
                    )
                    if archive.user_id != entry.owner_user_id or archive.session_id != entry.session_id:
                        raise RuntimeError("Session projection journal identity is detached")
                    self._project_session_archive(archive)
                    self._enqueue_session_commit(archive, tenant_id=tenant_id)
                    self.projection_journal.mark(entry, status="PROJECTED")
                    counts["projected"] += 1
                except Exception as exc:
                    self.projection_journal.mark(entry, status="FAILED", error=type(exc).__name__)
                    counts["failed"] += 1
                    raise RuntimeError("Session projection journal recovery failed") from exc
            if len(entries) < maximum:
                break
        return counts

    def rebuild_session_archives(
        self,
        *,
        batch_size: int = 256,
        max_archives: int = 10_000,
    ) -> dict[str, int]:
        """从不可变归档头重建派生的 Session Catalog 记录。

        启动过程会在 Runtime 处于 RECOVERING 时调用，因此这里有意绕过普通
        READY 门禁。枚举和总重放数量都有上限；达到上限后仍有未处理任务时，
        必须失败关闭，不能发布只完成部分重建的 Runtime。
        """

        if self.session_projector is None:
            return {
                "projected_archives": 0,
                "projected_records": 0,
                "async_output_archives": 0,
            }
        maximum = max(1, min(int(batch_size), 1_000))
        total_bound = int(max_archives)
        if total_bound <= 0 or total_bound > 100_000:
            raise ValueError("Session archive rebuild bound must be between 1 and 100000")
        tenant_id = str(self.archive_store.tenant_id)
        cursor = ""
        counts = {
            "projected_archives": 0,
            "projected_records": 0,
            "async_output_archives": 0,
        }
        while counts["projected_archives"] < total_bound:
            requested = min(maximum, total_bound - counts["projected_archives"])
            archives = self.archive_store.list_archives(
                tenant_id=tenant_id,
                after_archive_uri=cursor,
                limit=requested,
            )
            if not archives:
                break
            for archive in archives:
                cursor = archive.archive_uri
                try:
                    projection, _status = self._project_session_archive(
                        archive,
                        respect_applied_tombstones=True,
                    )
                    self._record_projection(
                        archive,
                        tenant_id=tenant_id,
                        status="PROJECTED",
                    )
                except Exception as exc:
                    self._record_projection(
                        archive,
                        tenant_id=tenant_id,
                        status="FAILED",
                        error=type(exc).__name__,
                    )
                    raise RuntimeError("Session archive Catalog rebuild failed") from exc
                counts["projected_archives"] += 1
                counts["projected_records"] += int(getattr(projection, "projected", 0) or 0)
                counts["async_output_archives"] += int(self.archive_store.async_outputs_done_for_task(archive))
            if len(archives) < requested:
                break
        if counts["projected_archives"] >= total_bound and self.archive_store.list_archives(
            tenant_id=tenant_id,
            after_archive_uri=cursor,
            limit=1,
        ):
            raise RuntimeError("Session archive rebuild exceeded its total bound")
        return counts

    def resume_startup_commit_group(
        self,
        archive: SessionArchive,
        *,
        group_id: str,
    ) -> SessionCommitResult:
        """精确校验归档身份后，重放一个耐久提交组。"""

        expected_group = f"commit_group_{archive.task_id}"
        if group_id != expected_group:
            raise RuntimeError("startup archive is detached from its commit-group identity")
        tenant_id = self._bind_archive_tenant(archive)
        group = self.commit_group_store.load(group_id)
        if group is None:
            raise RuntimeError("startup commit group does not exist")
        identity = (
            archive.task_id,
            archive.archive_uri,
            archive.user_id,
            tenant_id,
            archive.archive_digest,
            archive.manifest_digest,
        )
        durable = (
            group.task_id,
            group.archive_uri,
            group.user_id,
            group.tenant_id,
            group.archive_digest,
            group.manifest_digest,
        )
        if identity != durable:
            raise RuntimeError("startup archive is detached from its durable commit group")
        with self._startup_recovery_scope(group_id):
            return self.async_commit(archive)

    def resumable_commit_groups(self, *, limit: int = 256) -> tuple[CommitGroupStatus, ...]:
        """发现未完成提交组，以及缺少异步输出的已完成提交组。

        三个消费者全部耐久完成后、异步输出头发布前，进程仍可能退出。此时提交组
        在消费者存储中已经终结，``CommitGroupStore.pending()`` 无法发现它；恢复逻辑
        因而必须检查每个已完成提交组指向的精确不可变归档，再判断是否已经没有工作。

        此方法有意不要求 READY，因为启动恢复会在 Runtime 仍在证明耐久状态时调用。
        """

        maximum = max(1, min(int(limit), 1_000))
        actionable: list[CommitGroupStatus] = []
        leased: list[CommitGroupStatus] = []
        for group in self.commit_group_store.all():
            if not group.terminal:
                target = leased if any(item.status == "running" for item in group.consumers.values()) else actionable
                target.append(group)
            elif group.complete:
                archive = self.archive_store.read_archive_at_manifest(
                    group.archive_uri,
                    group.manifest_digest,
                    tenant_id=group.tenant_id,
                )
                identity = (
                    archive.task_id,
                    archive.archive_uri,
                    archive.user_id,
                    self._tenant_id(archive),
                    archive.archive_digest,
                    archive.manifest_digest,
                )
                durable = (
                    group.task_id,
                    group.archive_uri,
                    group.user_id,
                    group.tenant_id,
                    group.archive_digest,
                    group.manifest_digest,
                )
                if identity != durable:
                    raise RuntimeError("commit-group discovery found a detached SessionArchive")
                if not self.archive_store.async_outputs_done_for_task(archive):
                    actionable.append(group)
        # 长时间运行的消费者不能阻塞后续可立即恢复消费者工作或输出头发布的提交组。
        return tuple((*actionable, *leased)[:maximum])
