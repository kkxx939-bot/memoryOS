"""会话归档的租户绑定与文件系统目录布局。"""

from __future__ import annotations

from pathlib import Path

from infrastructure.store.model.context.context_uri import ContextURI
from memory.commit.evidence.errors import EvidenceArchiveIntegrityError
from pre.session import SessionArchive


class SessionArchiveLayout:
    """集中维护 archive_uri、租户和磁盘路径之间的唯一映射。"""

    def __init__(self, root: Path, tenant_id: str) -> None:
        self.root = root
        self.tenant_id = tenant_id

    def directory(self, archive_uri: str, *, tenant_id: str | None = None) -> Path:
        """把会话归档 URI 转换成当前存储根目录下的唯一源路径。"""

        return ContextURI.parse(archive_uri).to_source_path(
            self.root,
            tenant_id=tenant_id or self.tenant_id,
        )

    def archive_tenant(self, archive: SessionArchive) -> str:
        """验证归档声明的租户与当前存储绑定一致。"""

        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        direct = str(metadata.get("tenant_id") or "")
        scoped = str(scope.get("tenant_id") or "")
        if direct and scoped and direct != scoped:
            raise EvidenceArchiveIntegrityError("session archive metadata has conflicting tenants")
        claimed = direct or scoped
        if claimed and claimed != self.tenant_id:
            raise EvidenceArchiveIntegrityError("session archive tenant does not match the bound archive store")
        return self.tenant_id

    def materialize_archive_tenant(self, archive: SessionArchive, tenant_id: str) -> None:
        """在写盘前把已验证租户写入归档元数据。"""

        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        claimed = tuple(
            str(value) for value in (metadata.get("tenant_id"), scope.get("tenant_id")) if value not in (None, "")
        )
        if any(value != tenant_id for value in claimed):
            raise EvidenceArchiveIntegrityError("session archive metadata tenant mismatch")
        metadata["tenant_id"] = tenant_id
        if "scope" in metadata:
            scope["tenant_id"] = tenant_id
            metadata["scope"] = scope
        archive.metadata = metadata

    @staticmethod
    def manifest_uri(archive_uri: str, manifest_digest: str) -> str:
        """生成指向特定不可变 manifest 的归档 URI。"""

        return f"{archive_uri}#manifest={manifest_digest}"


__all__ = ["SessionArchiveLayout"]
