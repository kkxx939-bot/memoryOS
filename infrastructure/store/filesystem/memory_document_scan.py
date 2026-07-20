"""Markdown 记忆目录的有界安全扫描器。

扫描器只产生一次不可变的 ``ScanGeneration`` 观察结果，不直接修改进程内
document_id 注册表。注册表如何采用完整扫描结果由上层存储编排器决定。
"""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from datetime import datetime, timezone

from infrastructure.store.filesystem.memory_document_io import MemoryDocumentFileIO
from foundation.ids import stable_hash
from memory.core.structure.frontmatter import (
    FrontMatterError,
    MissingDocumentId,
    MissingFrontMatter,
    parse_front_matter,
)
from memory.core.model import (
    ManagedDocument,
    QuarantinedDocument,
    ScanGeneration,
    UnmanagedDocument,
    UnsafePath,
)
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import DocumentUnsafeError


def scan_memory_documents(
    *,
    files: MemoryDocumentFileIO,
    tenant: str,
    owner: str,
    previous_path_ids: dict[str, str],
    max_front_matter_bytes: int,
    max_front_matter_depth: int,
    max_scan_files: int,
) -> ScanGeneration:
    """扫描固定目录布局，隔离不安全文件并返回完整性标记。"""

    observed_at = datetime.now(timezone.utc).isoformat()
    generation_id = f"scan_{uuid.uuid4().hex}"
    try:
        root_descriptor = files.open_user_root(tenant, owner, create=False)
    except DocumentUnsafeError as exc:
        return ScanGeneration(
            generation_id=generation_id,
            tenant_id=tenant,
            owner_user_id=owner,
            root_identity="",
            observed_at=observed_at,
            complete=False,
            errors=(str(exc),),
        )
    if root_descriptor is None:
        # 根目录不存在只是一次观察事实，不构成清空历史身份注册的授权。
        return ScanGeneration(
            generation_id=generation_id,
            tenant_id=tenant,
            owner_user_id=owner,
            root_identity="",
            observed_at=observed_at,
            complete=True,
        )

    registrations: list[ManagedDocument | UnmanagedDocument | QuarantinedDocument] = []
    unsafe_paths: list[UnsafePath] = []
    errors: list[str] = []
    seen_ids: dict[str, list[int]] = {}
    collision_indexes: dict[str, list[int]] = {}
    file_count = 0
    visited_entry_count = 0
    controlled_directories = frozenset(
        {
            "knowledge",
            "knowledge/entities",
            "knowledge/topics",
            "knowledge/episodes",
            "experiences",
        }
    )

    def append_registration(
        item: ManagedDocument | UnmanagedDocument | QuarantinedDocument,
        *,
        collision_key: str = "",
        document_id: str = "",
    ) -> None:
        index = len(registrations)
        registrations.append(item)
        if collision_key:
            collision_indexes.setdefault(collision_key, []).append(index)
        if document_id:
            seen_ids.setdefault(document_id, []).append(index)

    def register_raw(relative: str, raw: bytes) -> None:
        digest = hashlib.sha256(raw).hexdigest()
        try:
            normalized = MemoryDocumentPathPolicy.normalize_relative_path(relative)
        except ValueError as exc:
            append_registration(QuarantinedDocument(relative, str(exc), raw_sha256=digest, size=len(raw)))
            return
        collision_key = MemoryDocumentPathPolicy.collision_key(normalized)
        try:
            parsed = parse_front_matter(
                raw,
                max_header_bytes=max_front_matter_bytes,
                max_depth=max_front_matter_depth,
                require_document_id=False,
            )
            try:
                document_id = parsed.document_id
            except MissingDocumentId as exc:
                if "document_id" in parsed.values:
                    append_registration(
                        QuarantinedDocument(
                            normalized,
                            str(exc),
                            raw_sha256=digest,
                            size=len(raw),
                        ),
                        collision_key=collision_key,
                    )
                else:
                    append_registration(
                        UnmanagedDocument(normalized, digest, len(raw), str(exc)),
                        collision_key=collision_key,
                    )
                return
        except MissingFrontMatter as exc:
            append_registration(
                UnmanagedDocument(normalized, digest, len(raw), str(exc)),
                collision_key=collision_key,
            )
            return
        except FrontMatterError as exc:
            append_registration(
                QuarantinedDocument(normalized, str(exc), raw_sha256=digest, size=len(raw)),
                collision_key=collision_key,
            )
            return
        previous_id = previous_path_ids.get(normalized)
        if previous_id and previous_id != document_id:
            append_registration(
                QuarantinedDocument(
                    normalized,
                    "document_id changed for an already observed path",
                    raw_sha256=digest,
                    size=len(raw),
                ),
                collision_key=collision_key,
                document_id=document_id,
            )
            return
        append_registration(
            ManagedDocument(normalized, document_id, digest, len(raw)),
            collision_key=collision_key,
            document_id=document_id,
        )

    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_nlink < 1:
            raise DocumentUnsafeError("memory root is not a safe directory")
        root_identity = stable_hash((root_metadata.st_dev, root_metadata.st_ino, tenant, owner), 32)
        stop_scan = False
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

        def scan_directory(descriptor: int, prefix: str) -> None:
            nonlocal file_count, stop_scan, visited_entry_count
            try:
                names: list[str] = []
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        visited_entry_count += 1
                        # 固定布局最多包含五个受控目录；其他所有条目都计入上限，
                        # 防止符号链接或未知目录树把有界扫描扩成无限枚举。
                        if visited_entry_count > max_scan_files + len(controlled_directories):
                            errors.append("memory scan entry limit exceeded")
                            stop_scan = True
                            return
                        names.append(entry.name)
            except OSError as exc:
                errors.append(files.safe_os_reason(exc))
                return
            for filename in sorted(names):
                if stop_scan:
                    return
                relative = f"{prefix}/{filename}" if prefix else filename
                try:
                    metadata = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
                except OSError as exc:
                    errors.append(files.safe_os_reason(exc))
                    continue
                if stat.S_ISDIR(metadata.st_mode):
                    if relative not in controlled_directories or metadata.st_dev != root_metadata.st_dev:
                        unsafe_paths.append(
                            UnsafePath(relative, "directory entry is unsafe or outside the controlled layout")
                        )
                        continue
                    try:
                        child = os.open(filename, directory_flags, dir_fd=descriptor)
                    except OSError as exc:
                        unsafe_paths.append(UnsafePath(relative, files.safe_os_reason(exc)))
                        continue
                    try:
                        opened = os.fstat(child)
                        if (
                            not stat.S_ISDIR(opened.st_mode)
                            or opened.st_dev != metadata.st_dev
                            or opened.st_ino != metadata.st_ino
                            or opened.st_dev != root_metadata.st_dev
                        ):
                            errors.append("memory directory changed during full scan")
                            continue
                        scan_directory(child, relative)
                    finally:
                        os.close(child)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    unsafe_paths.append(UnsafePath(relative, "path is a symlink or non-regular file"))
                    continue
                file_count += 1
                if file_count > max_scan_files:
                    errors.append("memory scan file limit exceeded")
                    stop_scan = True
                    return
                if metadata.st_nlink > 1:
                    unsafe_paths.append(UnsafePath(relative, "hard-linked memory documents are forbidden"))
                    continue
                try:
                    raw = files.read_regular(descriptor, filename)
                except (OSError, DocumentUnsafeError) as exc:
                    unsafe_paths.append(UnsafePath(relative, files.safe_os_reason(exc)))
                    continue
                register_raw(relative, raw)

        scan_directory(root_descriptor, "")
        try:
            rebound = files.open_user_root(tenant, owner, create=False)
            if rebound is None:
                errors.append("memory root disappeared during full scan")
            else:
                try:
                    rebound_metadata = os.fstat(rebound)
                    if (
                        rebound_metadata.st_dev != root_metadata.st_dev
                        or rebound_metadata.st_ino != root_metadata.st_ino
                    ):
                        errors.append("memory root changed during full scan")
                finally:
                    os.close(rebound)
        except DocumentUnsafeError as exc:
            errors.append(str(exc))
    except (OSError, DocumentUnsafeError) as exc:
        root_identity = ""
        errors.append(files.safe_os_reason(exc))
    finally:
        os.close(root_descriptor)

    # 同一个 document_id 或折叠后相同的路径都不能被静默选中，必须整体隔离。
    for document_id, indexes in seen_ids.items():
        if len(indexes) <= 1:
            continue
        for index in indexes:
            item = registrations[index]
            registrations[index] = QuarantinedDocument(
                item.relative_path,
                f"duplicate document_id: {document_id}",
                raw_sha256=item.raw_sha256,
                size=item.size,
            )
    for indexes in collision_indexes.values():
        paths = {registrations[index].relative_path for index in indexes}
        if len(paths) <= 1:
            continue
        for index in indexes:
            item = registrations[index]
            registrations[index] = QuarantinedDocument(
                item.relative_path,
                "Unicode/casefold path collision",
                raw_sha256=getattr(item, "raw_sha256", ""),
                size=getattr(item, "size", 0),
            )

    return ScanGeneration(
        generation_id=generation_id,
        tenant_id=tenant,
        owner_user_id=owner,
        root_identity=root_identity,
        observed_at=observed_at,
        complete=not errors and file_count <= max_scan_files,
        registrations=tuple(registrations),
        unsafe_paths=tuple(unsafe_paths),
        errors=tuple(errors),
    )


__all__ = ["scan_memory_documents"]
