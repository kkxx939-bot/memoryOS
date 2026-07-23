"""恢复中断的 Session Commit Group。"""

from __future__ import annotations

from typing import Any

from memory.commit.session_commit import SessionCommitService


def recover_session_commit_groups(service: SessionCommitService) -> dict[str, Any]:
    """回收租约并只恢复没有存活消费者的提交组。"""

    abandoned = service.commit_group_store.recover_abandoned_leases()
    expired = service.commit_group_store.recover_expired_consumers()
    resumed = 0
    for group in service.resumable_commit_groups(limit=1_000):
        if any(consumer.status == "running" for consumer in group.consumers.values()):
            raise RuntimeError(f"Session commit group has a live lease: {group.group_id}")
        archive = service.archive_store.read_archive_at_manifest(
            group.archive_uri,
            group.manifest_digest,
            tenant_id=group.tenant_id,
        )
        result = service.resume_startup_commit_group(archive, group_id=group.group_id)
        if not result.done:
            raise RuntimeError(f"Session commit group remains incomplete: {group.group_id}")
        resumed += 1
    return {"abandoned_leases": abandoned, "expired_consumers": expired, "resumed": resumed}


__all__ = ["recover_session_commit_groups"]
