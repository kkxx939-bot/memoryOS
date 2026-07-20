"""用户可编辑 Markdown 记忆文档的文件系统存储编排器。

本类只维护文档身份注册、CAS 前置条件和增删改流程。目录扫描由
``memory_document_scan`` 负责，安全文件访问与原子写入由
``MemoryDocumentFileIO`` 负责，避免业务语义与系统调用细节互相耦合。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

from infrastructure.store.filesystem.memory_document_io import (
    MemoryDocumentFileIO,
    StoreFaultHook,
)
from infrastructure.store.filesystem.memory_document_scan import scan_memory_documents
from memory.core.model import (
    ABSENT,
    AbsentPath,
    ManagedDocument,
    MemoryDocument,
    PresentPath,
    RawPathState,
    ScanGeneration,
    UnsafePath,
)
from memory.core.structure.frontmatter import (
    FrontMatterError,
    MissingFrontMatter,
    adopt_raw_document,
    new_document_id,
    parse_front_matter,
    validate_document_id,
)
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentUnsafeError,
)


class FileSystemMemoryDocumentStore:
    """以原始字节摘要作为 CAS 条件持久化 Markdown 记忆文档。"""

    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int = 2 * 1024 * 1024,
        max_front_matter_bytes: int = 32 * 1024,
        max_front_matter_depth: int = 12,
        max_scan_files: int = 10_000,
    ) -> None:
        candidate = Path(root).expanduser().absolute()
        if any(path.is_symlink() for path in (candidate, *candidate.parents)):
            raise DocumentUnsafeError("memory document runtime root cannot traverse a symbolic link")
        self.root = candidate.resolve(strict=False)
        if max_file_bytes <= 0 or max_front_matter_bytes <= 0 or max_front_matter_bytes >= max_file_bytes:
            raise ValueError("invalid memory document byte limits")
        if max_front_matter_depth <= 0 or max_scan_files <= 0:
            raise ValueError("invalid memory document scan limits")
        self.max_file_bytes = max_file_bytes
        self.max_front_matter_bytes = max_front_matter_bytes
        self.max_front_matter_depth = max_front_matter_depth
        self.max_scan_files = max_scan_files
        self._locations: dict[tuple[str, str, str], str] = {}
        self._path_ids: dict[tuple[str, str, str], str] = {}
        self._files = MemoryDocumentFileIO(self.root, max_file_bytes)

    def probe_write_capabilities(
        self,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> None:
        """在启动时验证当前挂载点满足安全持久化所需的系统调用语义。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = (
            MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
            if owner_user_id is not None
            else None
        )
        self._files.probe_write_capabilities(tenant, owner)

    def read_state(self, tenant_id: str, owner_user_id: str, relative_path: str) -> RawPathState:
        """读取路径的精确原始状态，供后续 CAS 写入比较。"""

        tenant, owner, relative = self._bound_identity(tenant_id, owner_user_id, relative_path)
        root_descriptor = self._files.open_user_root(tenant, owner, create=False)
        if root_descriptor is None:
            return ABSENT
        try:
            parent_descriptor, filename = self._files.open_parent(root_descriptor, relative, create=False)
            if parent_descriptor is None:
                return ABSENT
            try:
                raw = self._files.read_regular(parent_descriptor, filename)
            except FileNotFoundError:
                return ABSENT
            except DocumentUnsafeError as exc:
                return UnsafePath(relative, str(exc))
            finally:
                os.close(parent_descriptor)
            return PresentPath(relative, hashlib.sha256(raw).hexdigest(), len(raw))
        except (PermissionError, OSError) as exc:
            return UnsafePath(relative, self._files.safe_os_reason(exc))
        finally:
            os.close(root_descriptor)

    def read_raw(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        document_id: str = "",
        relative_path: str = "",
    ) -> bytes:
        """按 document_id 或相对路径读取文档原始字节。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        if bool(document_id) == bool(relative_path):
            raise ValueError("read_raw requires exactly one of document_id or relative_path")
        if document_id:
            key = (tenant, owner, str(document_id))
            relative_path = self._locations.get(key, "")
            if not relative_path:
                raise DocumentNotFoundError("document ID is not registered; a startup/full scan is required")
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        root_descriptor = self._files.open_user_root(tenant, owner, create=False)
        if root_descriptor is None:
            raise DocumentNotFoundError("memory document root does not exist")
        try:
            parent_descriptor, filename = self._files.open_parent(root_descriptor, relative, create=False)
            if parent_descriptor is None:
                raise DocumentNotFoundError("memory document does not exist")
            try:
                return self._files.read_regular(parent_descriptor, filename)
            except FileNotFoundError as exc:
                raise DocumentNotFoundError("memory document does not exist") from exc
            finally:
                os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)

    def full_scan(self, tenant_id: str, owner_user_id: str) -> ScanGeneration:
        """执行有界扫描，并只用完整扫描结果刷新进程内身份注册表。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        previous_path_ids = {
            relative: document_id
            for (registered_tenant, registered_owner, relative), document_id in self._path_ids.items()
            if (registered_tenant, registered_owner) == (tenant, owner)
        }
        generation = scan_memory_documents(
            files=self._files,
            tenant=tenant,
            owner=owner,
            previous_path_ids=previous_path_ids,
            max_front_matter_bytes=self.max_front_matter_bytes,
            max_front_matter_depth=self.max_front_matter_depth,
            max_scan_files=self.max_scan_files,
        )
        # 只有真实存在且扫描完整的根目录，才有权替换已有注册表快照。
        if generation.complete and generation.root_identity:
            for key in tuple(self._locations):
                if key[:2] == (tenant, owner):
                    self._locations.pop(key, None)
            for key in tuple(self._path_ids):
                if key[:2] == (tenant, owner):
                    self._path_ids.pop(key, None)
            # 路径仍存在但处于 unsafe/quarantined 状态时保留旧 ID 基线；
            # 完整观察到路径消失后清除基线，使未来 create 获得新身份。
            present_paths = {item.relative_path for item in generation.registrations}
            present_paths.update(item.relative_path for item in generation.unsafe_paths)
            for relative, document_id in previous_path_ids.items():
                if relative in present_paths:
                    self._path_ids[(tenant, owner, relative)] = document_id
        for item in generation.registrations:
            if isinstance(item, ManagedDocument):
                self._locations[(tenant, owner, item.document_id)] = item.relative_path
                self._path_ids[(tenant, owner, item.relative_path)] = item.document_id
        return generation

    def seed_registration(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        relative_path: str,
    ) -> None:
        """在首次进程扫描前注入已经持久化的路径与 ID 基线。"""

        tenant, owner, relative = self._bound_identity(tenant_id, owner_user_id, relative_path)
        identifier = validate_document_id(document_id)
        path_key = (tenant, owner, relative)
        existing_id = self._path_ids.get(path_key)
        if existing_id and existing_id != identifier:
            raise DocumentConflictError("durable path registration conflicts with the process baseline")
        identity_key = (tenant, owner, identifier)
        existing_path = self._locations.get(identity_key)
        if existing_path and existing_path != relative:
            raise DocumentConflictError("durable document identity conflicts with the process baseline")
        self._path_ids[path_key] = identifier
        self._locations[identity_key] = relative

    def create(
        self,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        after_bytes: bytes,
        *,
        expected: RawPathState = ABSENT,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
    ) -> MemoryDocument:
        """仅当目标仍为 ABSENT 时创建新文档。"""

        tenant, owner, relative = self._bound_identity(tenant_id, owner_user_id, relative_path)
        if not isinstance(expected, AbsentPath) or self.read_state(tenant, owner, relative) != expected:
            raise DocumentConflictError("create expected ABSENT but the live raw state differs")
        document = self._document_from_raw(tenant, owner, relative, after_bytes)
        existing = self._locations.get((tenant, owner, document.document_id))
        if existing and existing != relative:
            raise DocumentConflictError("document_id is already registered at another path")
        root_descriptor = self._files.open_user_root(tenant, owner, create=True)
        assert root_descriptor is not None
        try:
            parent_descriptor, filename = self._files.open_parent(root_descriptor, relative, create=True)
            assert parent_descriptor is not None
            try:
                self._files.reject_collision(parent_descriptor, relative, exclude_names=(filename,))
                self._files.atomic_create(
                    parent_descriptor,
                    filename,
                    after_bytes,
                    operation_id=operation_id,
                    fault_hook=fault_hook,
                )
                self._files.reject_collision(parent_descriptor, relative, exclude_names=(filename,))
            finally:
                os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)
        self._register(document)
        return document

    def replace(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        after_bytes: bytes,
        *,
        expected_state: RawPathState,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
    ) -> MemoryDocument:
        """在精确原始状态未变化时替换文档内容。"""

        tenant, owner, relative = self._registered_location(tenant_id, owner_user_id, document_id)
        self._require_expected(tenant, owner, relative, expected_state)
        document = self._document_from_raw(tenant, owner, relative, after_bytes)
        if document.document_id != document_id:
            raise DocumentConflictError("system update cannot change document_id")
        root_descriptor = self._files.open_user_root(tenant, owner, create=False)
        assert root_descriptor is not None
        try:
            parent_descriptor, filename = self._files.open_parent(root_descriptor, relative, create=False)
            assert parent_descriptor is not None
            try:
                self._require_expected(tenant, owner, relative, expected_state)
                self._files.atomic_replace(
                    parent_descriptor,
                    filename,
                    after_bytes,
                    operation_id=operation_id,
                    fault_hook=fault_hook,
                    pre_install=lambda: self._require_expected(tenant, owner, relative, expected_state),
                )
            finally:
                os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)
        self._register(document)
        return document

    def delete(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        *,
        expected_state: RawPathState,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
    ) -> RawPathState:
        """在 CAS 状态匹配时删除文档，并同步清除身份注册。"""

        del operation_id
        tenant, owner, relative = self._registered_location(tenant_id, owner_user_id, document_id)
        self._require_expected(tenant, owner, relative, expected_state)
        root_descriptor = self._files.open_user_root(tenant, owner, create=False)
        assert root_descriptor is not None
        try:
            parent_descriptor, filename = self._files.open_parent(root_descriptor, relative, create=False)
            assert parent_descriptor is not None
            try:
                self._require_expected(tenant, owner, relative, expected_state)
                os.unlink(filename, dir_fd=parent_descriptor)
                self._files.notify_fault(fault_hook, "atomic_installed")
                os.fsync(parent_descriptor)
                self._files.notify_fault(fault_hook, "parent_fsynced")
            finally:
                os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)
        self._locations.pop((tenant, owner, document_id), None)
        self._path_ids.pop((tenant, owner, relative), None)
        return ABSENT

    def rename(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        new_relative_path: str,
        *,
        expected_old: RawPathState,
        expected_new: RawPathState = ABSENT,
        after_bytes: bytes | None = None,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
    ) -> MemoryDocument:
        """以无覆盖语义移动文档，可同时安装新的精确字节。"""

        tenant, owner, old_relative = self._registered_location(tenant_id, owner_user_id, document_id)
        new_relative = MemoryDocumentPathPolicy.normalize_relative_path(new_relative_path)
        if old_relative == new_relative:
            raise ValueError("rename target must differ from the current path")
        self._require_expected(tenant, owner, old_relative, expected_old)
        if not isinstance(expected_new, AbsentPath) or self.read_state(tenant, owner, new_relative) != expected_new:
            raise DocumentConflictError("rename target is not ABSENT")
        raw = (
            bytes(after_bytes) if after_bytes is not None else self.read_raw(tenant, owner, relative_path=old_relative)
        )
        document = self._document_from_raw(tenant, owner, new_relative, raw)
        if document.document_id != document_id:
            raise DocumentConflictError("rename source document_id no longer matches")
        root_descriptor = self._files.open_user_root(tenant, owner, create=False)
        assert root_descriptor is not None
        try:
            old_parent, old_name = self._files.open_parent(root_descriptor, old_relative, create=False)
            new_parent, new_name = self._files.open_parent(root_descriptor, new_relative, create=True)
            assert old_parent is not None and new_parent is not None
            try:
                self._require_expected(tenant, owner, old_relative, expected_old)
                if self.read_state(tenant, owner, new_relative) != ABSENT:
                    raise DocumentConflictError("rename target changed after planning")
                same_parent = self._files.same_directory(old_parent, new_parent)
                excluded = (new_name, old_name) if same_parent else (new_name,)
                self._files.reject_collision(new_parent, new_relative, exclude_names=excluded)
                if after_bytes is None:
                    self._files.rename_noreplace(old_parent, old_name, new_parent, new_name)
                else:

                    def temp_stage_only(stage: str) -> None:
                        if stage == "temp_file_fsynced":
                            self._files.notify_fault(fault_hook, stage)

                    self._files.atomic_create(
                        new_parent,
                        new_name,
                        raw,
                        operation_id=operation_id,
                        fault_hook=temp_stage_only,
                    )
                    self._files.reject_collision(
                        new_parent,
                        new_relative,
                        exclude_names=(new_name, old_name) if same_parent else (new_name,),
                    )
                    self._files.notify_fault(fault_hook, "rename_target_installed")
                    self._require_expected(tenant, owner, old_relative, expected_old)
                    os.unlink(old_name, dir_fd=old_parent)
                self._files.notify_fault(fault_hook, "atomic_installed")
                os.fsync(old_parent)
                if not same_parent:
                    os.fsync(new_parent)
                self._files.notify_fault(fault_hook, "parent_fsynced")
                self._files.reject_collision(new_parent, new_relative, exclude_names=(new_name,))
            finally:
                os.close(old_parent)
                os.close(new_parent)
        finally:
            os.close(root_descriptor)
        self._locations[(tenant, owner, document_id)] = new_relative
        self._path_ids.pop((tenant, owner, old_relative), None)
        self._path_ids[(tenant, owner, new_relative)] = document_id
        return document

    def adopt(
        self,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        *,
        expected_raw_sha256: str,
        assigned_document_id: str | None = None,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
    ) -> MemoryDocument:
        """为尚未托管的用户文件写入 document_id，并纳入注册表。"""

        tenant, owner, relative = self._bound_identity(tenant_id, owner_user_id, relative_path)
        current = self.read_state(tenant, owner, relative)
        if not isinstance(current, PresentPath) or current.raw_sha256 != expected_raw_sha256:
            raise DocumentConflictError("adopt expected raw digest does not match")
        raw = self.read_raw(tenant, owner, relative_path=relative)
        try:
            parsed = parse_front_matter(
                raw,
                max_header_bytes=self.max_front_matter_bytes,
                max_depth=self.max_front_matter_depth,
                require_document_id=False,
            )
            if "document_id" in parsed.values:
                raise DocumentConflictError("managed or invalid-ID files cannot be adopted")
        except MissingFrontMatter:
            pass
        except FrontMatterError as exc:
            raise DocumentUnsafeError(str(exc)) from exc
        document_id = (
            validate_document_id(assigned_document_id) if assigned_document_id is not None else new_document_id()
        )
        after = adopt_raw_document(
            raw,
            document_id,
            max_header_bytes=self.max_front_matter_bytes,
            max_depth=self.max_front_matter_depth,
        )
        root_descriptor = self._files.open_user_root(tenant, owner, create=False)
        assert root_descriptor is not None
        try:
            parent_descriptor, filename = self._files.open_parent(root_descriptor, relative, create=False)
            assert parent_descriptor is not None
            try:
                self._require_expected(tenant, owner, relative, current)
                self._files.atomic_replace(
                    parent_descriptor,
                    filename,
                    after,
                    operation_id=operation_id,
                    fault_hook=fault_hook,
                    pre_install=lambda: self._require_expected(tenant, owner, relative, current),
                )
            finally:
                os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)
        document = self._document_from_raw(tenant, owner, relative, after)
        self._register(document)
        return document

    def cleanup_operation_temps(
        self,
        tenant_id: str,
        owner_user_id: str,
        expected_raw_sha256_by_path: Mapping[str, str],
        operation_id: str,
    ) -> int:
        """委托底层文件层清理可证明属于指定操作的临时文件。"""

        return self._files.cleanup_operation_temps(
            tenant_id,
            owner_user_id,
            expected_raw_sha256_by_path,
            operation_id,
        )

    def _bound_identity(self, tenant_id: str, owner_user_id: str, relative_path: str) -> tuple[str, str, str]:
        return (
            MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id"),
            MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id"),
            MemoryDocumentPathPolicy.normalize_relative_path(relative_path),
        )

    def _registered_location(self, tenant_id: str, owner_user_id: str, document_id: str) -> tuple[str, str, str]:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        relative = self._locations.get((tenant, owner, str(document_id)), "")
        if not relative:
            raise DocumentNotFoundError("document ID is not registered for this tenant and owner")
        return tenant, owner, relative

    def _document_from_raw(self, tenant: str, owner: str, relative: str, raw: bytes) -> MemoryDocument:
        if len(raw) > self.max_file_bytes:
            raise DocumentUnsafeError("memory document exceeds the configured byte limit")
        try:
            parsed = parse_front_matter(
                raw,
                max_header_bytes=self.max_front_matter_bytes,
                max_depth=self.max_front_matter_depth,
            )
        except FrontMatterError as exc:
            raise DocumentUnsafeError(str(exc)) from exc
        return MemoryDocument(
            tenant_id=tenant,
            owner_user_id=owner,
            document_id=parsed.document_id,
            relative_path=relative,
            document_kind=MemoryDocumentPathPolicy.kind_for(relative),
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
            raw_bytes=bytes(raw),
            body=parsed.body,
            front_matter=parsed.values,
        )

    def _register(self, document: MemoryDocument) -> None:
        self._locations[(document.tenant_id, document.owner_user_id, document.document_id)] = document.relative_path
        self._path_ids[(document.tenant_id, document.owner_user_id, document.relative_path)] = document.document_id

    def _require_expected(self, tenant: str, owner: str, relative: str, expected: RawPathState) -> None:
        if isinstance(expected, UnsafePath):
            raise DocumentUnsafeError("UNSAFE state can never authorize a write")
        if self.read_state(tenant, owner, relative) != expected:
            raise DocumentConflictError("live raw state no longer matches the exact expected state")


__all__ = ["FileSystemMemoryDocumentStore"]
