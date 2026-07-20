"""发布扫描器确认的外部 Markdown 变化，并收口收养后的初始化状态。"""

from __future__ import annotations

import hashlib

from infrastructure.store.memory.bootstrap import MemoryDocumentBootstrapper
from infrastructure.store.memory.control_store import MemoryDocumentControlStore
from infrastructure.store.memory.scanner import ExternalDocumentChange
from memory.commit.document_commit import DocumentCommitResult, MemoryDocumentCommitter
from memory.core.structure.frontmatter import matches_adopted_source
from memory.ports.document_store import MemoryDocumentStore


def publish_external_change(
    change: ExternalDocumentChange,
    *,
    committer: MemoryDocumentCommitter,
    control_store: MemoryDocumentControlStore,
    document_store: MemoryDocumentStore,
    bootstrapper: MemoryDocumentBootstrapper,
) -> DocumentCommitResult | None:
    """发布扫描事实，并在运行时 READY 前完成 adopt-first 初始化。"""

    result = committer.record_external_change(change)
    if change.change_kind.value != "create":
        return result
    receipt = control_store.load_adoption_receipt_for_document(
        change.tenant_id,
        change.owner_user_id,
        change.document_id,
    )
    if receipt is None:
        return result
    raw = document_store.read_raw(
        change.tenant_id,
        change.owner_user_id,
        document_id=change.document_id,
    )
    exact_adoption = (
        change.new_relative_path == receipt.relative_path
        and hashlib.sha256(raw).hexdigest() == change.after_raw_digest
        and matches_adopted_source(raw, receipt.document_id, receipt.expected_raw_sha256)
    )
    if exact_adoption:
        bootstrapper.ensure_adopted_user(
            change.tenant_id,
            change.owner_user_id,
            receipt.relative_path,
            document_id=receipt.document_id,
            adopted_raw_sha256=change.after_raw_digest,
        )
    else:
        # 已完成标记让后续编辑或重命名回到普通路径；缺失时必须在启动阶段失败关闭。
        bootstrapper.ensure_user(change.tenant_id, change.owner_user_id)
    return result


__all__ = ["publish_external_change"]
