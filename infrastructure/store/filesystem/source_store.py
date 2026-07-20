"""ContextDB 权威 SourceStore 协议的文件系统实现。

本模块负责对象级路由、租户边界和生命周期；需要原子发布的版本化对象由
``SourceBundleStore`` 处理，底层文件持久化由 ``SourceFileIO`` 处理。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from infrastructure.store.contracts.domain import ContextDomainClassifier, NoContextDomainClassifier
from infrastructure.store.filesystem.source_bundle import (
    BundleIntegrityError,
    SourceBundleStore,
    SourceFileIO,
)
from infrastructure.store.locks.process_local import ProcessLocalLockStore
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_uri import ContextURI
from infrastructure.store.model.context.lifecycle import LifecycleState


class FileSystemSourceStore:
    """通过文件系统持久化 ContextDB 源对象，并维护严格的租户边界。"""

    def __init__(
        self,
        root: str | Path,
        tenant_id: str = "default",
        *,
        domain_classifier: ContextDomainClassifier | None = None,
    ) -> None:
        if (
            not isinstance(tenant_id, str)
            or not tenant_id.strip()
            or tenant_id in {".", ".."}
            or "/" in tenant_id
            or "\\" in tenant_id
        ):
            raise ValueError("tenant_id must be one safe non-empty path segment")
        self.root = Path(root).expanduser().resolve()
        self.tenant_id = tenant_id
        self.domain_classifier = domain_classifier or NoContextDomainClassifier()
        self._operation_lock_store = ProcessLocalLockStore()
        self.test_hook: Callable[[str, str, str], None] | None = None
        self._files = SourceFileIO(self.root)
        self._bundles = SourceBundleStore(
            self.root,
            self.tenant_id,
            self._files,
            test_hook=lambda: self.test_hook,
        )

    def operation_lock_store(self) -> ProcessLocalLockStore:
        """返回当前进程内协调 SourceStore 写操作的锁实现。"""

        return self._operation_lock_store

    def read_object(self, uri: str) -> ContextObject:
        """读取源对象，并在返回前校验 bundle 与租户身份。"""

        self._reject_memory_document_uri(uri)
        directory = self._object_dir(uri)
        pointer = directory / ".bundle-current.json"
        if pointer.exists() or pointer.is_symlink():
            obj, _content = self._bundles.read(uri, pointer)
        else:
            path = directory / ".meta.json"
            if path.is_symlink():
                raise BundleIntegrityError(f"object metadata cannot be a symbolic link: {uri}")
            obj = ContextObject.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if ContextURI.parse(uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
            raise FileNotFoundError(uri)
        return obj

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        """根据领域归属选择普通文件或完整版本化 bundle 写入。"""

        self._reject_memory_document_uri(obj.uri)
        if ContextURI.parse(obj.uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
            raise PermissionError("ContextObject tenant does not match SourceStore tenant")
        if self.domain_classifier.owns_object(obj):
            self._bundles.write(obj, content)
            return
        directory = self._object_dir(obj.uri)
        self._files.ensure_private_directory(directory)
        self._files.write_text_atomic(
            directory / ".meta.json",
            json.dumps(obj.to_dict(), ensure_ascii=False, indent=2),
        )
        relations = {"uri": obj.uri, "relations": [relation.to_dict() for relation in obj.relations]}
        self._files.write_text_atomic(
            directory / ".relations.json",
            json.dumps(relations, ensure_ascii=False, indent=2),
        )
        if content:
            self.write_content(obj.layers.l2_uri or obj.uri, content)

    def read_content(self, uri: str) -> str:
        """读取普通正文，或从当前 bundle generation 读取 L2 正文。"""

        self._reject_memory_document_uri(uri)
        bundle = self._bundles.resolve_content_pointer(uri)
        if bundle is not None:
            _obj, content = self._bundles.read(bundle[0], bundle[1])
            return content
        path = self._content_path(uri)
        if path.is_symlink():
            raise BundleIntegrityError(f"object content cannot be a symbolic link: {uri}")
        return path.read_text(encoding="utf-8")

    def write_content(self, uri: str, content: str | bytes) -> None:
        """写入正文；已进入 bundle 的对象必须发布完整的新 generation。"""

        self._reject_memory_document_uri(uri)
        bundle = self._bundles.resolve_content_pointer(uri)
        if bundle is not None:
            obj, _old_content = self._bundles.read(bundle[0], bundle[1])
            self._bundles.write(obj, content, preserve_existing_content=False)
            return
        path = self._content_path(uri)
        if isinstance(content, bytes):
            self._files.write_bytes_atomic(path, content)
        else:
            self._files.write_text_atomic(path, content)

    def soft_delete(self, uri: str, reason: str) -> None:
        """保留对象内容，并把生命周期更新为逻辑删除。"""

        self._reject_memory_document_uri(uri)
        obj = self.read_object(uri)
        obj.lifecycle_state = LifecycleState.DELETED
        obj.metadata = {**obj.metadata, "delete_reason": reason}
        self.write_object(obj)

    def delete_object(self, uri: str) -> None:
        """物理删除一个源对象目录。"""

        self._reject_memory_document_uri(uri)
        directory = self._object_dir(uri)
        if directory.exists():
            shutil.rmtree(directory)

    def list_objects(self) -> list[ContextObject]:
        """枚举当前租户和共享资源中的普通对象与 bundle current 指针。"""

        if not self.root.exists():
            return []
        objects: list[ContextObject] = []
        paths = [
            *self.root.glob(f"tenants/{self.tenant_id}/**/.meta.json"),
            *self.root.glob("resources/**/.meta.json"),
            *self.root.glob("skills/**/.meta.json"),
        ]
        for path in sorted(set(paths)):
            if ".bundle-generations" in path.parts:
                continue
            if path.is_symlink():
                raise BundleIntegrityError(f"object metadata cannot be a symbolic link: {path}")
            obj = ContextObject.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if ContextURI.parse(obj.uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
                continue
            objects.append(obj)
        pointer_paths = [
            *self.root.glob(f"tenants/{self.tenant_id}/**/.bundle-current.json"),
            *self.root.glob("resources/**/.bundle-current.json"),
            *self.root.glob("skills/**/.bundle-current.json"),
        ]
        by_uri = {obj.uri: obj for obj in objects}
        for pointer in sorted(set(pointer_paths)):
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                uri = str(payload.get("uri") or "")
                if not uri:
                    raise BundleIntegrityError("bundle pointer is missing its URI")
                obj, _content = self._bundles.read(uri, pointer)
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise BundleIntegrityError(f"cannot enumerate corrupt object bundle: {pointer}") from exc
            if ContextURI.parse(obj.uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
                continue
            by_uri[obj.uri] = obj
        return list(by_uri.values())

    def _object_dir(self, uri: str) -> Path:
        return ContextURI.parse(uri).to_source_path(self.root, tenant_id=self.tenant_id)

    @staticmethod
    def _reject_memory_document_uri(uri: str) -> None:
        parsed = ContextURI.parse(uri)
        if parsed.authority == "user" and parsed.segments[1:3] == ("memory", "documents"):
            raise PermissionError(
                "Markdown memory documents are not ordinary SourceStore objects; "
                "use MemoryDocumentStore and MemoryDocumentCommitter"
            )

    def _content_path(self, uri: str) -> Path:
        parsed = ContextURI.parse(uri)
        path = parsed.to_source_path(self.root, tenant_id=self.tenant_id)
        return path if path.suffix else path / "content.md"


__all__ = ["BundleIntegrityError", "FileSystemSourceStore"]
