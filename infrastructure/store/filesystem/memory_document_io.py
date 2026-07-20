"""Markdown 记忆文档的安全文件访问与原子写入原语。

该模块只处理文件描述符、目录边界、链接防护、fsync 和无覆盖重命名；
文档身份、front matter 和注册表语义不在这里决定。
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import (
    DocumentConflictError,
    DocumentUnsafeError,
)

StoreFaultHook = Callable[[str], None]


class MemoryDocumentFileIO:
    """封装记忆文档存储所需的低层安全文件系统操作。"""

    def __init__(self, root: Path, max_file_bytes: int) -> None:
        self.root = root
        self.max_file_bytes = max_file_bytes
        self._probed_scopes: set[tuple[str, str]] = set()

    def probe_write_capabilities(self, tenant: str, owner: str | None) -> None:
        """验证挂载点支持本存储依赖的持久化与无覆盖写入原语。"""

        probe_scope = (tenant, owner or "__control__")
        if probe_scope in self._probed_scopes:
            if owner is not None:
                # 能力探测可以缓存，但每次 CREATE 预检仍需重新确认真实 owner 根目录。
                owner_descriptor = self.open_user_root(tenant, owner, create=True)
                assert owner_descriptor is not None
                os.close(owner_descriptor)
            return
        root_descriptor = self.open_directory_chain((), create=True)
        assert root_descriptor is not None
        scope_descriptor: int | None = None
        parent_descriptor: int | None = None
        probe_descriptor: int | None = None
        probe_name = f".filesystem-probe-{uuid.uuid4().hex}"
        probe_succeeded = False
        cleanup_error: OSError | None = None
        try:
            if owner is None:
                scope_descriptor = self.open_directory_chain(
                    () if tenant == "default" else ("tenants", tenant),
                    create=True,
                )
                assert scope_descriptor is not None
                parent_descriptor, _placeholder = self.open_parent(
                    scope_descriptor,
                    "system/memory-documents/.probe-placeholder",
                    create=True,
                )
            else:
                scope_descriptor = self.open_user_root(tenant, owner, create=True)
                assert scope_descriptor is not None
                parent_descriptor = os.dup(scope_descriptor)
            assert parent_descriptor is not None
            if os.fstat(parent_descriptor).st_dev != os.fstat(root_descriptor).st_dev:
                raise DocumentUnsafeError("memory document paths cross filesystems")
            os.mkdir(probe_name, 0o700, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            probe_descriptor = os.open(probe_name, flags, dir_fd=parent_descriptor)
            metadata = os.fstat(probe_descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_nlink < 1:
                raise DocumentUnsafeError("filesystem capability probe is not a directory")

            self.atomic_create(probe_descriptor, "source", b"probe-v1")
            self.atomic_replace(probe_descriptor, "source", b"probe-v2")
            self.atomic_create(probe_descriptor, "occupied", b"occupied")
            try:
                self.rename_noreplace(probe_descriptor, "source", probe_descriptor, "occupied")
            except DocumentConflictError:
                pass
            else:
                raise DocumentUnsafeError("filesystem rename unexpectedly overwrote an existing file")
            if self.read_regular(probe_descriptor, "source") != b"probe-v2":
                raise DocumentUnsafeError("filesystem no-replace probe changed its source")
            os.unlink("occupied", dir_fd=probe_descriptor)
            self.rename_noreplace(probe_descriptor, "source", probe_descriptor, "renamed")
            os.fsync(probe_descriptor)
            if self.read_regular(probe_descriptor, "renamed") != b"probe-v2":
                raise DocumentUnsafeError("filesystem rename probe did not preserve exact bytes")
            os.unlink("renamed", dir_fd=probe_descriptor)
            os.fsync(probe_descriptor)
            probe_succeeded = True
        except (OSError, DocumentConflictError) as exc:
            raise DocumentUnsafeError("memory document filesystem capability probe failed") from exc
        finally:
            if probe_descriptor is not None:
                for entry in ("source", "occupied", "renamed"):
                    try:
                        os.unlink(entry, dir_fd=probe_descriptor)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        cleanup_error = cleanup_error or exc
                try:
                    os.close(probe_descriptor)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
            if parent_descriptor is not None:
                try:
                    os.rmdir(probe_name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
                finally:
                    try:
                        os.close(parent_descriptor)
                    except OSError as exc:
                        cleanup_error = cleanup_error or exc
            if scope_descriptor is not None:
                try:
                    os.close(scope_descriptor)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
            try:
                os.close(root_descriptor)
            except OSError as exc:
                cleanup_error = cleanup_error or exc
            if probe_succeeded and cleanup_error is not None:
                raise DocumentUnsafeError(
                    "memory document filesystem capability probe cleanup failed"
                ) from cleanup_error
        self._probed_scopes.add(probe_scope)

    def open_user_root(self, tenant: str, owner: str, *, create: bool) -> int | None:
        """从受信任片段打开一个租户用户的记忆根目录。"""

        return self.open_directory_chain(
            ("tenants", tenant, "users", owner, "memory"),
            create=create,
        )

    def open_directory_chain(self, segments: tuple[str, ...], *, create: bool) -> int | None:
        """逐段打开目录，拒绝符号链接和非目录节点。"""

        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.root, flags)
        except OSError as exc:
            raise DocumentUnsafeError("runtime root is not a safe directory") from exc
        try:
            for segment in segments:
                try:
                    child = os.open(segment, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        os.close(descriptor)
                        return None
                    os.mkdir(segment, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    child = os.open(segment, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise DocumentUnsafeError("memory root traverses a symlink or non-directory") from exc
                metadata = os.fstat(child)
                if not stat.S_ISDIR(metadata.st_mode) or metadata.st_nlink < 1:
                    os.close(child)
                    raise DocumentUnsafeError("memory root contains a non-directory")
                try:
                    os.fchmod(child, 0o700)
                except OSError:
                    pass
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    @staticmethod
    def open_parent(root_descriptor: int, relative: str, *, create: bool) -> tuple[int | None, str]:
        """相对已打开根目录安全遍历到目标父目录。"""

        parts = relative.split("/")
        descriptor = os.dup(root_descriptor)
        root_device = os.fstat(root_descriptor).st_dev
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            for segment in parts[:-1]:
                try:
                    child = os.open(segment, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        os.close(descriptor)
                        return None, parts[-1]
                    os.mkdir(segment, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    child = os.open(segment, flags, dir_fd=descriptor)
                metadata = os.fstat(child)
                if not stat.S_ISDIR(metadata.st_mode) or metadata.st_dev != root_device:
                    os.close(child)
                    raise DocumentUnsafeError("memory document path crosses a filesystem boundary")
                os.close(descriptor)
                descriptor = child
            return descriptor, parts[-1]
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    def read_regular(self, parent_descriptor: int, filename: str) -> bytes:
        """读取一个无符号链接、无硬链接且大小受限的普通文件。"""

        descriptor = os.open(filename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise DocumentUnsafeError("memory path is not a regular file")
            if metadata.st_nlink > 1:
                raise DocumentUnsafeError("hard-linked memory documents are forbidden")
            return self._read_bounded(descriptor, metadata.st_size, "memory document")
        finally:
            os.close(descriptor)

    def read_path(self, path: Path) -> bytes:
        """通过绝对路径读取受限普通文件，主要用于恢复临时文件。"""

        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1:
                raise DocumentUnsafeError("memory path is not one unlinked regular file")
            return self._read_bounded(descriptor, metadata.st_size, "memory document")
        finally:
            os.close(descriptor)

    def cleanup_operation_temps(
        self,
        tenant_id: str,
        owner_user_id: str,
        expected_raw_sha256_by_path: Mapping[str, str],
        operation_id: str,
    ) -> int:
        """只删除名称、operation_id 和预期摘要全部匹配的确定性临时文件。"""

        if not operation_id:
            return 0
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        root_descriptor = self.open_user_root(tenant, owner, create=False)
        if root_descriptor is None:
            return 0
        removed = 0
        try:
            for candidate, expected_digest in dict(expected_raw_sha256_by_path).items():
                if len(expected_digest) != 64 or any(
                    character not in "0123456789abcdef" for character in expected_digest
                ):
                    raise ValueError("document operation temp requires an exact SHA-256 digest")
                relative = MemoryDocumentPathPolicy.normalize_relative_path(candidate)
                parent_descriptor, filename = self.open_parent(root_descriptor, relative, create=False)
                if parent_descriptor is None:
                    continue
                temporary = self.temporary_name(filename, operation_id)
                try:
                    try:
                        metadata = os.stat(temporary, dir_fd=parent_descriptor, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if not stat.S_ISREG(metadata.st_mode):
                        raise DocumentUnsafeError("document operation temp is not a regular file")
                    raw = self._read_cleanup_temp(parent_descriptor, temporary, target=filename)
                    if hashlib.sha256(raw).hexdigest() != expected_digest:
                        raise DocumentConflictError("document operation temp differs from its durable prepared digest")
                    os.unlink(temporary, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                    removed += 1
                finally:
                    os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)
        return removed

    def atomic_create(
        self,
        parent_descriptor: int,
        filename: str,
        raw: bytes,
        *,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
    ) -> None:
        """以 create-only 语义发布新文件，绝不覆盖已有用户文件。"""

        temporary = self._write_temp(parent_descriptor, filename, raw, operation_id=operation_id)
        self.notify_fault(fault_hook, "temp_file_fsynced")
        preserve_temp = False
        try:
            try:
                os.link(
                    temporary,
                    filename,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise DocumentConflictError("create target appeared after CAS validation") from exc
            try:
                self.notify_fault(fault_hook, "atomic_installed")
            except BaseException:
                preserve_temp = True
                raise
            os.unlink(temporary, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            self.notify_fault(fault_hook, "parent_fsynced")
        finally:
            if not preserve_temp:
                try:
                    os.unlink(temporary, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass

    def atomic_replace(
        self,
        parent_descriptor: int,
        filename: str,
        raw: bytes,
        *,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
        pre_install: Callable[[], None] | None = None,
    ) -> None:
        """写入并 fsync 临时文件，最后用一次原子替换安装。"""

        temporary = self._write_temp(parent_descriptor, filename, raw, operation_id=operation_id)
        self.notify_fault(fault_hook, "temp_file_fsynced")
        try:
            if pre_install is not None:
                # 临时文件准备期间用户可能修改目标；安装前必须再次验证 CAS 前置条件。
                pre_install()
            os.replace(
                temporary,
                filename,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            self.notify_fault(fault_hook, "atomic_installed")
            os.fsync(parent_descriptor)
            self.notify_fault(fault_hook, "parent_fsynced")
        finally:
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass

    def _read_cleanup_temp(self, parent_descriptor: int, temporary: str, *, target: str) -> bytes:
        descriptor = os.open(
            temporary,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink not in {1, 2}:
                raise DocumentUnsafeError("document operation temp has an unsafe link count")
            if metadata.st_nlink == 2:
                try:
                    target_metadata = os.stat(target, dir_fd=parent_descriptor, follow_symlinks=False)
                except FileNotFoundError as exc:
                    raise DocumentUnsafeError("linked document operation temp is detached from its target") from exc
                if (
                    not stat.S_ISREG(target_metadata.st_mode)
                    or target_metadata.st_dev != metadata.st_dev
                    or target_metadata.st_ino != metadata.st_ino
                ):
                    raise DocumentUnsafeError("linked document operation temp is detached from its exact target")
            return self._read_bounded(descriptor, metadata.st_size, "document operation temp")
        finally:
            os.close(descriptor)

    def _write_temp(
        self,
        parent_descriptor: int,
        filename: str,
        raw: bytes,
        *,
        operation_id: str = "",
    ) -> str:
        temporary = self.temporary_name(filename, operation_id)
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as exc:
            existing = self._read_temp(parent_descriptor, temporary)
            if existing != raw:
                raise DocumentConflictError("document operation temp does not match its deterministic bytes") from exc
            return temporary
        try:
            view = memoryview(raw)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("memory document write made no progress")
                view = view[count:]
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
            raise
        else:
            os.close(descriptor)
        return temporary

    def _read_temp(self, parent_descriptor: int, temporary: str) -> bytes:
        descriptor = os.open(
            temporary,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise DocumentUnsafeError("document operation temp is not one regular file")
            return self._read_bounded(descriptor, metadata.st_size, "document operation temp")
        finally:
            os.close(descriptor)

    def _read_bounded(self, descriptor: int, size: int, label: str) -> bytes:
        if size > self.max_file_bytes:
            raise DocumentUnsafeError(f"{label} exceeds the configured byte limit")
        chunks: list[bytes] = []
        remaining = self.max_file_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > self.max_file_bytes:
            raise DocumentUnsafeError(f"{label} exceeds the configured byte limit")
        return raw

    @staticmethod
    def temporary_name(filename: str, operation_id: str) -> str:
        token = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:32] if operation_id else uuid.uuid4().hex
        return f".{filename}.memoryos-{token}.tmp"

    @staticmethod
    def notify_fault(fault_hook: StoreFaultHook | None, stage: str) -> None:
        if fault_hook is not None:
            fault_hook(stage)

    @staticmethod
    def same_directory(left: int, right: int) -> bool:
        left_metadata = os.fstat(left)
        right_metadata = os.fstat(right)
        return (left_metadata.st_dev, left_metadata.st_ino) == (
            right_metadata.st_dev,
            right_metadata.st_ino,
        )

    @staticmethod
    def reject_collision(
        parent_descriptor: int,
        relative_path: str,
        *,
        exclude_names: Iterable[str] = (),
    ) -> None:
        """拒绝 Unicode NFC 或大小写折叠后与目标相同的同级路径。"""

        excluded = frozenset(exclude_names)
        parent, _separator, _filename = relative_path.rpartition("/")
        target_key = unicodedata.normalize("NFC", relative_path).casefold()
        for name in os.listdir(parent_descriptor):
            if name in excluded:
                continue
            sibling = f"{parent}/{name}" if parent else name
            if unicodedata.normalize("NFC", sibling).casefold() == target_key:
                raise DocumentConflictError("Unicode/casefold path collision")

    @staticmethod
    def rename_noreplace(source_parent: int, source: str, target_parent: int, target: str) -> None:
        """优先使用平台无覆盖重命名，并提供不会覆盖目标的安全回退。"""

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            result = renameat2(source_parent, source.encode(), target_parent, target.encode(), 1)
            if result == 0:
                return
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise DocumentConflictError("rename target appeared after CAS validation")
            if error not in {errno.ENOSYS, errno.EINVAL}:
                raise OSError(error, os.strerror(error))
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is not None:
            result = renameatx_np(source_parent, source.encode(), target_parent, target.encode(), 0x00000004)
            if result == 0:
                return
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise DocumentConflictError("rename target appeared after CAS validation")
            if error not in {errno.ENOTSUP, errno.EINVAL, errno.ENOSYS}:
                raise OSError(error, os.strerror(error))
        # 回退方案用 create-only 硬链接安装目标，不会覆盖用户文件。崩溃若留下重复
        # document_id，后续扫描会隔离该状态，等待前滚修复。
        try:
            os.link(
                source,
                target,
                src_dir_fd=source_parent,
                dst_dir_fd=target_parent,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise DocumentConflictError("rename target appeared after CAS validation") from exc
        os.fsync(target_parent)
        os.unlink(source, dir_fd=source_parent)

    @staticmethod
    def safe_os_reason(exc: BaseException) -> str:
        if isinstance(exc, DocumentUnsafeError):
            return str(exc)
        if isinstance(exc, PermissionError):
            return "permission denied while reading memory tree"
        if isinstance(exc, OSError):
            return f"filesystem error errno={exc.errno}"
        return type(exc).__name__


__all__ = ["MemoryDocumentFileIO", "StoreFaultHook"]
