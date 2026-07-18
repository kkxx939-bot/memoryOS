"""Durable exact-byte filesystem source for user-editable Markdown memory."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from memoryos.core.ids import stable_hash
from memoryos.memory.documents.frontmatter import (
    FrontMatterError,
    MissingDocumentId,
    MissingFrontMatter,
    adopt_raw_document,
    new_document_id,
    parse_front_matter,
    validate_document_id,
)
from memoryos.memory.documents.model import (
    ABSENT,
    AbsentPath,
    ManagedDocument,
    MemoryDocument,
    PresentPath,
    QuarantinedDocument,
    RawPathState,
    ScanGeneration,
    UnmanagedDocument,
    UnsafePath,
)
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.store import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentUnsafeError,
)

StoreFaultHook = Callable[[str], None]


class FileSystemMemoryDocumentStore:
    """A separate source store with raw-digest CAS and no hidden bundle format."""

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
        self._probed_scopes: set[tuple[str, str]] = set()

    def probe_write_capabilities(
        self,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> None:
        """Fail closed unless the bound filesystem supports our write primitives.

        The probe deliberately makes no claim that a remote/FUSE filesystem has
        local-disk crash semantics.  It only verifies, at runtime startup, that
        this mounted root accepts durable file and directory fsync plus
        create-only and no-replace rename operations.  All artifacts live in a
        private, uniquely named bound source/control directory and are removed
        before return.
        """

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = (
            MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
            if owner_user_id is not None
            else None
        )
        probe_scope = (tenant, owner or "__control__")
        if probe_scope in self._probed_scopes:
            if owner is not None:
                # Capability results may be cached, but CREATE preflight still
                # needs a real current owner-root inode. Re-open/create it on
                # every call so a removed root can never fall back to the
                # synthetic ``full_scan(... absent)`` identity.
                owner_descriptor = self._open_user_root(tenant, owner, create=True)
                assert owner_descriptor is not None
                os.close(owner_descriptor)
            return
        root_descriptor = self._open_directory_chain((), create=True)
        assert root_descriptor is not None
        scope_descriptor: int | None = None
        parent_descriptor: int | None = None
        probe_descriptor: int | None = None
        probe_name = f".filesystem-probe-{uuid.uuid4().hex}"
        probe_succeeded = False
        cleanup_error: OSError | None = None
        try:
            if owner is None:
                scope_descriptor = self._open_directory_chain(
                    () if tenant == "default" else ("tenants", tenant),
                    create=True,
                )
                assert scope_descriptor is not None
                parent_descriptor, _placeholder = self._open_parent(
                    scope_descriptor,
                    "system/memory-documents/.probe-placeholder",
                    create=True,
                )
            else:
                scope_descriptor = self._open_user_root(tenant, owner, create=True)
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

            self._atomic_create(probe_descriptor, "source", b"probe-v1")
            self._atomic_replace(probe_descriptor, "source", b"probe-v2")
            self._atomic_create(probe_descriptor, "occupied", b"occupied")
            try:
                self._rename_noreplace(probe_descriptor, "source", probe_descriptor, "occupied")
            except DocumentConflictError:
                pass
            else:
                raise DocumentUnsafeError("filesystem rename unexpectedly overwrote an existing file")
            if self._read_regular(probe_descriptor, "source") != b"probe-v2":
                raise DocumentUnsafeError("filesystem no-replace probe changed its source")
            os.unlink("occupied", dir_fd=probe_descriptor)
            self._rename_noreplace(probe_descriptor, "source", probe_descriptor, "renamed")
            os.fsync(probe_descriptor)
            if self._read_regular(probe_descriptor, "renamed") != b"probe-v2":
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

    def read_state(self, tenant_id: str, owner_user_id: str, relative_path: str) -> RawPathState:
        tenant, owner, relative = self._bound_identity(tenant_id, owner_user_id, relative_path)
        root_descriptor = self._open_user_root(tenant, owner, create=False)
        if root_descriptor is None:
            return ABSENT
        try:
            parent_descriptor, filename = self._open_parent(root_descriptor, relative, create=False)
            if parent_descriptor is None:
                return ABSENT
            try:
                raw = self._read_regular(parent_descriptor, filename)
            except FileNotFoundError:
                return ABSENT
            except DocumentUnsafeError as exc:
                return UnsafePath(relative, str(exc))
            finally:
                os.close(parent_descriptor)
            return PresentPath(relative, hashlib.sha256(raw).hexdigest(), len(raw))
        except (PermissionError, OSError) as exc:
            return UnsafePath(relative, self._safe_os_reason(exc))
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
        root_descriptor = self._open_user_root(tenant, owner, create=False)
        if root_descriptor is None:
            raise DocumentNotFoundError("memory document root does not exist")
        try:
            parent_descriptor, filename = self._open_parent(root_descriptor, relative, create=False)
            if parent_descriptor is None:
                raise DocumentNotFoundError("memory document does not exist")
            try:
                return self._read_regular(parent_descriptor, filename)
            except FileNotFoundError as exc:
                raise DocumentNotFoundError("memory document does not exist") from exc
            finally:
                os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)

    def full_scan(self, tenant_id: str, owner_user_id: str) -> ScanGeneration:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        observed_at = datetime.now(timezone.utc).isoformat()
        generation_id = f"scan_{uuid.uuid4().hex}"
        try:
            root_descriptor = self._open_user_root(tenant, owner, create=False)
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
            # Absence is a scan fact, not an inode identity. It must never
            # establish durable authority for a later delete decision.
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
                    max_header_bytes=self.max_front_matter_bytes,
                    max_depth=self.max_front_matter_depth,
                    require_document_id=False,
                )
                try:
                    document_id = parsed.document_id
                except MissingDocumentId as exc:
                    if "document_id" in parsed.values:
                        append_registration(
                            QuarantinedDocument(normalized, str(exc), raw_sha256=digest, size=len(raw)),
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
            previous_id = self._path_ids.get((tenant, owner, normalized))
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
                            # The controlled layout has at most these five
                            # directory entries. Count every other filesystem
                            # entry against the configured file bound so a
                            # tree of symlinks/unknown directories cannot make
                            # a supposedly bounded scan enumerate forever.
                            if visited_entry_count > self.max_scan_files + len(controlled_directories):
                                errors.append("memory scan entry limit exceeded")
                                stop_scan = True
                                return
                            names.append(entry.name)
                except OSError as exc:
                    errors.append(self._safe_os_reason(exc))
                    return
                for filename in sorted(names):
                    if stop_scan:
                        return
                    relative = f"{prefix}/{filename}" if prefix else filename
                    try:
                        metadata = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
                    except OSError as exc:
                        errors.append(self._safe_os_reason(exc))
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
                            unsafe_paths.append(UnsafePath(relative, self._safe_os_reason(exc)))
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
                    if file_count > self.max_scan_files:
                        errors.append("memory scan file limit exceeded")
                        stop_scan = True
                        return
                    if metadata.st_nlink > 1:
                        unsafe_paths.append(UnsafePath(relative, "hard-linked memory documents are forbidden"))
                        continue
                    try:
                        raw = self._read_regular(descriptor, filename)
                    except (OSError, DocumentUnsafeError) as exc:
                        unsafe_paths.append(UnsafePath(relative, self._safe_os_reason(exc)))
                        continue
                    register_raw(relative, raw)

            scan_directory(root_descriptor, "")
            try:
                rebound = self._open_user_root(tenant, owner, create=False)
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
            errors.append(self._safe_os_reason(exc))
        finally:
            os.close(root_descriptor)

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

        complete = not errors and file_count <= self.max_scan_files
        if complete:
            for key in tuple(self._locations):
                if key[:2] == (tenant, owner):
                    self._locations.pop(key, None)
            previous_path_ids = {
                key: document_id
                for key, document_id in self._path_ids.items()
                if key[:2] == (tenant, owner)
            }
            for key in previous_path_ids:
                self._path_ids.pop(key, None)
            # Keep the old ID baseline only while that exact path is still
            # present but unsafe/quarantined. A complete observation of
            # absence clears it, so a later create is a new identity.
            present_paths = {item.relative_path for item in registrations}
            present_paths.update(item.relative_path for item in unsafe_paths)
            for key, document_id in previous_path_ids.items():
                if key[2] in present_paths:
                    self._path_ids[key] = document_id
        for item in registrations:
            if isinstance(item, ManagedDocument):
                self._locations[(tenant, owner, item.document_id)] = item.relative_path
                self._path_ids[(tenant, owner, item.relative_path)] = item.document_id
        return ScanGeneration(
            generation_id=generation_id,
            tenant_id=tenant,
            owner_user_id=owner,
            root_identity=root_identity,
            observed_at=observed_at,
            complete=complete,
            registrations=tuple(registrations),
            unsafe_paths=tuple(unsafe_paths),
            errors=tuple(errors),
        )

    def seed_registration(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        relative_path: str,
    ) -> None:
        """Seed a durable path/ID baseline before the first process scan."""

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
        tenant, owner, relative = self._bound_identity(tenant_id, owner_user_id, relative_path)
        if not isinstance(expected, AbsentPath) or self.read_state(tenant, owner, relative) != expected:
            raise DocumentConflictError("create expected ABSENT but the live raw state differs")
        document = self._document_from_raw(tenant, owner, relative, after_bytes)
        existing = self._locations.get((tenant, owner, document.document_id))
        if existing and existing != relative:
            raise DocumentConflictError("document_id is already registered at another path")
        root_descriptor = self._open_user_root(tenant, owner, create=True)
        assert root_descriptor is not None
        try:
            parent_descriptor, filename = self._open_parent(root_descriptor, relative, create=True)
            assert parent_descriptor is not None
            try:
                self._reject_collision(parent_descriptor, relative, exclude_names=(filename,))
                self._atomic_create(
                    parent_descriptor,
                    filename,
                    after_bytes,
                    operation_id=operation_id,
                    fault_hook=fault_hook,
                )
                self._reject_collision(parent_descriptor, relative, exclude_names=(filename,))
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
        tenant, owner, relative = self._registered_location(tenant_id, owner_user_id, document_id)
        self._require_expected(tenant, owner, relative, expected_state)
        document = self._document_from_raw(tenant, owner, relative, after_bytes)
        if document.document_id != document_id:
            raise DocumentConflictError("system update cannot change document_id")
        root_descriptor = self._open_user_root(tenant, owner, create=False)
        assert root_descriptor is not None
        try:
            parent_descriptor, filename = self._open_parent(root_descriptor, relative, create=False)
            assert parent_descriptor is not None
            try:
                self._require_expected(tenant, owner, relative, expected_state)
                self._atomic_replace(
                    parent_descriptor,
                    filename,
                    after_bytes,
                    operation_id=operation_id,
                    fault_hook=fault_hook,
                    pre_install=lambda: self._require_expected(
                        tenant,
                        owner,
                        relative,
                        expected_state,
                    ),
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
        tenant, owner, relative = self._registered_location(tenant_id, owner_user_id, document_id)
        self._require_expected(tenant, owner, relative, expected_state)
        root_descriptor = self._open_user_root(tenant, owner, create=False)
        assert root_descriptor is not None
        try:
            parent_descriptor, filename = self._open_parent(root_descriptor, relative, create=False)
            assert parent_descriptor is not None
            try:
                self._require_expected(tenant, owner, relative, expected_state)
                os.unlink(filename, dir_fd=parent_descriptor)
                self._notify_store_fault(fault_hook, "atomic_installed")
                os.fsync(parent_descriptor)
                self._notify_store_fault(fault_hook, "parent_fsynced")
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
        tenant, owner, old_relative = self._registered_location(tenant_id, owner_user_id, document_id)
        new_relative = MemoryDocumentPathPolicy.normalize_relative_path(new_relative_path)
        if old_relative == new_relative:
            raise ValueError("rename target must differ from the current path")
        self._require_expected(tenant, owner, old_relative, expected_old)
        if not isinstance(expected_new, AbsentPath) or self.read_state(tenant, owner, new_relative) != expected_new:
            raise DocumentConflictError("rename target is not ABSENT")
        raw = (
            bytes(after_bytes)
            if after_bytes is not None
            else self.read_raw(tenant, owner, relative_path=old_relative)
        )
        document = self._document_from_raw(tenant, owner, new_relative, raw)
        if document.document_id != document_id:
            raise DocumentConflictError("rename source document_id no longer matches")
        root_descriptor = self._open_user_root(tenant, owner, create=False)
        assert root_descriptor is not None
        try:
            old_parent, old_name = self._open_parent(root_descriptor, old_relative, create=False)
            new_parent, new_name = self._open_parent(root_descriptor, new_relative, create=True)
            assert old_parent is not None and new_parent is not None
            try:
                self._require_expected(tenant, owner, old_relative, expected_old)
                if self.read_state(tenant, owner, new_relative) != ABSENT:
                    raise DocumentConflictError("rename target changed after planning")
                same_parent = self._same_directory(old_parent, new_parent)
                excluded = (new_name, old_name) if same_parent else (new_name,)
                self._reject_collision(new_parent, new_relative, exclude_names=excluded)
                if after_bytes is None:
                    self._rename_noreplace(old_parent, old_name, new_parent, new_name)
                else:
                    def temp_stage_only(stage: str) -> None:
                        if stage == "temp_file_fsynced":
                            self._notify_store_fault(fault_hook, stage)

                    self._atomic_create(
                        new_parent,
                        new_name,
                        raw,
                        operation_id=operation_id,
                        fault_hook=temp_stage_only,
                    )
                    self._reject_collision(
                        new_parent,
                        new_relative,
                        exclude_names=(new_name, old_name) if same_parent else (new_name,),
                    )
                    self._notify_store_fault(fault_hook, "rename_target_installed")
                    self._require_expected(tenant, owner, old_relative, expected_old)
                    os.unlink(old_name, dir_fd=old_parent)
                self._notify_store_fault(fault_hook, "atomic_installed")
                os.fsync(old_parent)
                if not same_parent:
                    os.fsync(new_parent)
                self._notify_store_fault(fault_hook, "parent_fsynced")
                self._reject_collision(new_parent, new_relative, exclude_names=(new_name,))
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
            validate_document_id(assigned_document_id)
            if assigned_document_id is not None
            else new_document_id()
        )
        after = adopt_raw_document(
            raw,
            document_id,
            max_header_bytes=self.max_front_matter_bytes,
            max_depth=self.max_front_matter_depth,
        )
        root_descriptor = self._open_user_root(tenant, owner, create=False)
        assert root_descriptor is not None
        try:
            parent_descriptor, filename = self._open_parent(root_descriptor, relative, create=False)
            assert parent_descriptor is not None
            try:
                self._require_expected(tenant, owner, relative, current)
                self._atomic_replace(
                    parent_descriptor,
                    filename,
                    after,
                    operation_id=operation_id,
                    fault_hook=fault_hook,
                    pre_install=lambda: self._require_expected(
                        tenant,
                        owner,
                        relative,
                        current,
                    ),
                )
            finally:
                os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)
        document = self._document_from_raw(tenant, owner, relative, after)
        self._register(document)
        return document

    def _bound_identity(self, tenant_id: str, owner_user_id: str, relative_path: str) -> tuple[str, str, str]:
        return (
            MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id"),
            MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id"),
            MemoryDocumentPathPolicy.normalize_relative_path(relative_path),
        )

    def _registered_location(self, tenant_id: str, owner_user_id: str, document_id: str) -> tuple[str, str, str]:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        key = (tenant, owner, str(document_id))
        relative = self._locations.get(key, "")
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
        live = self.read_state(tenant, owner, relative)
        if live != expected:
            raise DocumentConflictError("live raw state no longer matches the exact expected state")

    def _open_user_root(self, tenant: str, owner: str, *, create: bool) -> int | None:
        return self._open_directory_chain(
            ("tenants", tenant, "users", owner, "memory"),
            create=create,
        )

    def _open_directory_chain(self, segments: tuple[str, ...], *, create: bool) -> int | None:
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
    def _open_parent(root_descriptor: int, relative: str, *, create: bool) -> tuple[int | None, str]:
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

    def _read_regular(self, parent_descriptor: int, filename: str) -> bytes:
        descriptor = os.open(filename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise DocumentUnsafeError("memory path is not a regular file")
            if metadata.st_nlink > 1:
                raise DocumentUnsafeError("hard-linked memory documents are forbidden")
            if metadata.st_size > self.max_file_bytes:
                raise DocumentUnsafeError("memory document exceeds the configured byte limit")
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
                raise DocumentUnsafeError("memory document exceeds the configured byte limit")
            return raw
        finally:
            os.close(descriptor)

    def _read_path(self, path: Path) -> bytes:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1:
                raise DocumentUnsafeError("memory path is not one unlinked regular file")
            if metadata.st_size > self.max_file_bytes:
                raise DocumentUnsafeError("memory document exceeds the configured byte limit")
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
                raise DocumentUnsafeError("memory document exceeds the configured byte limit")
            return raw
        finally:
            os.close(descriptor)

    def cleanup_operation_temps(
        self,
        tenant_id: str,
        owner_user_id: str,
        expected_raw_sha256_by_path: Mapping[str, str],
        operation_id: str,
    ) -> int:
        """Durably remove only exact deterministic temps owned by one operation.

        The filename alone is not deletion authority because the user can edit
        this directory directly.  Recovery must also prove the temp's exact
        prepared digest; a colliding or externally modified file is preserved.
        """

        if not operation_id:
            return 0
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        root_descriptor = self._open_user_root(tenant, owner, create=False)
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
                parent_descriptor, filename = self._open_parent(root_descriptor, relative, create=False)
                if parent_descriptor is None:
                    continue
                temporary = self._temporary_name(filename, operation_id)
                try:
                    try:
                        metadata = os.stat(temporary, dir_fd=parent_descriptor, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if not stat.S_ISREG(metadata.st_mode):
                        raise DocumentUnsafeError("document operation temp is not a regular file")
                    raw = self._read_cleanup_temp(
                        parent_descriptor,
                        temporary,
                        target=filename,
                    )
                    if hashlib.sha256(raw).hexdigest() != expected_digest:
                        raise DocumentConflictError(
                            "document operation temp differs from its durable prepared digest"
                        )
                    os.unlink(temporary, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                    removed += 1
                finally:
                    os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)
        return removed

    def _read_cleanup_temp(
        self,
        parent_descriptor: int,
        temporary: str,
        *,
        target: str,
    ) -> bytes:
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
                    target_metadata = os.stat(
                        target,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError as exc:
                    raise DocumentUnsafeError(
                        "linked document operation temp is detached from its target"
                    ) from exc
                if (
                    not stat.S_ISREG(target_metadata.st_mode)
                    or target_metadata.st_dev != metadata.st_dev
                    or target_metadata.st_ino != metadata.st_ino
                ):
                    raise DocumentUnsafeError(
                        "linked document operation temp is detached from its exact target"
                    )
            if metadata.st_size > self.max_file_bytes:
                raise DocumentUnsafeError("document operation temp exceeds the configured byte limit")
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
                raise DocumentUnsafeError("document operation temp exceeds the configured byte limit")
            return raw
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
        temporary = self._temporary_name(filename, operation_id)
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
                raise DocumentConflictError(
                    "document operation temp does not match its deterministic bytes"
                ) from exc
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

    def _atomic_create(
        self,
        parent_descriptor: int,
        filename: str,
        raw: bytes,
        *,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
    ) -> None:
        temporary = self._write_temp(
            parent_descriptor,
            filename,
            raw,
            operation_id=operation_id,
        )
        self._notify_store_fault(fault_hook, "temp_file_fsynced")
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
                self._notify_store_fault(fault_hook, "atomic_installed")
            except BaseException:
                preserve_temp = True
                raise
            os.unlink(temporary, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            self._notify_store_fault(fault_hook, "parent_fsynced")
        finally:
            if not preserve_temp:
                try:
                    os.unlink(temporary, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass

    def _atomic_replace(
        self,
        parent_descriptor: int,
        filename: str,
        raw: bytes,
        *,
        operation_id: str = "",
        fault_hook: StoreFaultHook | None = None,
        pre_install: Callable[[], None] | None = None,
    ) -> None:
        temporary = self._write_temp(
            parent_descriptor,
            filename,
            raw,
            operation_id=operation_id,
        )
        self._notify_store_fault(fault_hook, "temp_file_fsynced")
        try:
            if pre_install is not None:
                # Temp preparation may take long enough for an uncooperative
                # editor to replace the live file. Revalidate immediately
                # before the single atomic install primitive.
                pre_install()
            os.replace(
                temporary,
                filename,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            self._notify_store_fault(fault_hook, "atomic_installed")
            os.fsync(parent_descriptor)
            self._notify_store_fault(fault_hook, "parent_fsynced")
        finally:
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass

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
                raise DocumentUnsafeError("document operation temp exceeds the configured byte limit")
            return raw
        finally:
            os.close(descriptor)

    @staticmethod
    def _temporary_name(filename: str, operation_id: str) -> str:
        token = (
            hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:32]
            if operation_id
            else uuid.uuid4().hex
        )
        return f".{filename}.memoryos-{token}.tmp"

    @staticmethod
    def _notify_store_fault(fault_hook: StoreFaultHook | None, stage: str) -> None:
        if fault_hook is not None:
            fault_hook(stage)

    @staticmethod
    def _same_directory(left: int, right: int) -> bool:
        left_metadata = os.fstat(left)
        right_metadata = os.fstat(right)
        return (left_metadata.st_dev, left_metadata.st_ino) == (
            right_metadata.st_dev,
            right_metadata.st_ino,
        )

    @staticmethod
    def _reject_collision(
        parent_descriptor: int,
        relative_path: str,
        *,
        exclude_names: Iterable[str] = (),
    ) -> None:
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
    def _rename_noreplace(source_parent: int, source: str, target_parent: int, target: str) -> None:
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
        # Portable fail-safe fallback: link is create-only, so it never
        # overwrites a user file. A crash between link and unlink leaves a
        # duplicate ID which the scanner quarantines for roll-forward repair.
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
    def _safe_os_reason(exc: BaseException) -> str:
        if isinstance(exc, DocumentUnsafeError):
            return str(exc)
        if isinstance(exc, PermissionError):
            return "permission denied while reading memory tree"
        if isinstance(exc, OSError):
            return f"filesystem error errno={exc.errno}"
        return type(exc).__name__


__all__ = ["FileSystemMemoryDocumentStore"]
