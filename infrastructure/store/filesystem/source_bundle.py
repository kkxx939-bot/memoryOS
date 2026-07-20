"""ContextDB 源对象版本化 bundle 的持久化实现。

本模块只负责一组必须原子发布的对象元数据、关系和正文。普通 SourceStore
对象的路由、租户校验和生命周期操作仍由 ``FileSystemSourceStore`` 负责。
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path

from foundation.integrity import canonical_digest
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_uri import ContextURI


class BundleIntegrityError(RuntimeError):
    """版本化源对象 bundle 不完整、被篡改或包含不安全路径。"""


class SourceFileIO:
    """提供源对象存储共用的私有目录和原子文件写入能力。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write_text_atomic(self, path: Path, content: str) -> None:
        """在同一目录内写临时文件并原子替换目标文件。"""

        if path.is_symlink():
            raise BundleIntegrityError(f"SourceStore path cannot be a symbolic link: {path}")
        self.ensure_private_directory(path.parent)
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise BundleIntegrityError(f"SourceStore path cannot be a symbolic link: {path}")
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            self.fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def write_bytes_atomic(self, path: Path, content: bytes) -> None:
        """以私有权限原子写入字节内容。"""

        if path.is_symlink():
            raise BundleIntegrityError(f"object content cannot be a symbolic link: {path}")
        self.ensure_private_directory(path.parent)
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise BundleIntegrityError(f"object content cannot be a symbolic link: {path}")
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            self.fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def fsync_directory(path: Path) -> None:
        """把目录项变化刷新到稳定存储。"""

        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def ensure_private_directory(self, directory: Path) -> None:
        """创建私有目录，并持久化从目标目录到存储根目录的目录项。"""

        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        durable_chain: list[Path] = []
        current = directory
        while current == self.root or self.root in current.parents:
            durable_chain.append(current)
            try:
                current.chmod(0o700)
            except OSError:
                pass
            if current == self.root:
                break
            current = current.parent
        # 仅 fsync 文件及其直接父目录，无法保证新建的更高层目录项已持久化。
        # 因此这里逐层刷新，避免崩溃后留下指向不存在 generation 的 current 指针。
        for current in durable_chain:
            self.fsync_directory(current)


class SourceBundleStore:
    """发布并校验内容寻址的源对象 generation。"""

    def __init__(
        self,
        root: Path,
        tenant_id: str,
        files: SourceFileIO,
        *,
        test_hook: Callable[[], Callable[[str, str, str], None] | None],
    ) -> None:
        self.root = root
        self.tenant_id = tenant_id
        self.files = files
        self._test_hook = test_hook

    def write(
        self,
        obj: ContextObject,
        content: str | bytes,
        *,
        preserve_existing_content: bool = True,
    ) -> None:
        """写入完整 generation，校验后再切换 current 指针。"""

        directory = self._object_dir(obj.uri)
        pointer = directory / ".bundle-current.json"
        if pointer.is_symlink():
            raise BundleIntegrityError(f"bundle pointer cannot be a symbolic link: {obj.uri}")
        if preserve_existing_content and content in {"", b""} and pointer.exists():
            # 只更新对象元数据时必须保留原有 L2 正文，但仍发布一个完整的新 generation。
            _current_object, current_content = self.read(obj.uri, pointer)
            content = current_content
        generations = directory / ".bundle-generations"
        generation_id = uuid.uuid4().hex
        generation = generations / generation_id
        self.files.ensure_private_directory(generation)
        encoded_content = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        object_payload = obj.to_dict()
        relations_payload = {
            "uri": obj.uri,
            "relations": [relation.to_dict() for relation in obj.relations],
        }
        meta_path = generation / ".meta.json"
        relations_path = generation / ".relations.json"
        content_path = generation / "content.md"
        manifest_path = generation / "manifest.json"
        self.files.write_text_atomic(
            meta_path,
            json.dumps(object_payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        self._notify("after_meta", obj.uri, generation_id)
        self.files.write_text_atomic(
            relations_path,
            json.dumps(relations_payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        self._notify("after_relations", obj.uri, generation_id)
        self.files.write_text_atomic(content_path, encoded_content)
        self._notify("after_content", obj.uri, generation_id)
        manifest_core = {
            "schema_version": "source_object_bundle_v1",
            "uri": obj.uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "generation_id": generation_id,
            "object_digest": canonical_digest(object_payload),
            "relations_digest": canonical_digest(relations_payload),
            "content_digest": canonical_digest(encoded_content),
            "files": {
                "metadata": ".meta.json",
                "relations": ".relations.json",
                "content": "content.md",
            },
        }
        manifest = {**manifest_core, "manifest_digest": canonical_digest(manifest_core)}
        self.files.write_text_atomic(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        )
        self._notify("after_bundle_manifest", obj.uri, generation_id)
        self.validate_generation(obj.uri, generation, manifest)
        # generation 内部文件已分别落盘；发布指针前还要持久化 generation 目录项。
        self.files.fsync_directory(generations)
        self._notify("before_bundle_publish", obj.uri, generation_id)
        self._notify("before_current_pointer", obj.uri, generation_id)
        pointer_core = {
            "schema_version": "source_object_bundle_current_v1",
            "uri": obj.uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "generation_id": generation_id,
            "manifest_digest": manifest["manifest_digest"],
        }
        self.files.write_text_atomic(
            pointer,
            json.dumps(
                {**pointer_core, "pointer_digest": canonical_digest(pointer_core)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )
        self._notify("after_current_pointer", obj.uri, generation_id)
        self._notify("after_bundle_publish", obj.uri, generation_id)

    def read(self, uri: str, pointer_path: Path) -> tuple[ContextObject, str]:
        """沿 current 指针读取 generation，并验证所有身份和摘要。"""

        if pointer_path.is_symlink():
            raise BundleIntegrityError(f"bundle pointer cannot be a symbolic link: {uri}")
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BundleIntegrityError(f"bundle pointer is unreadable: {uri}") from exc
        pointer_core = {key: value for key, value in pointer.items() if key != "pointer_digest"}
        generation_id = str(pointer.get("generation_id") or "")
        if (
            pointer.get("schema_version") != "source_object_bundle_current_v1"
            or pointer.get("uri") != uri
            or pointer.get("tenant_id") != self.tenant_id
            or not generation_id
            or any(character not in "0123456789abcdef" for character in generation_id.casefold())
            or pointer.get("pointer_digest") != canonical_digest(pointer_core)
        ):
            raise BundleIntegrityError(f"bundle pointer integrity check failed: {uri}")
        generation = pointer_path.parent / ".bundle-generations" / generation_id
        manifest_path = generation / "manifest.json"
        if generation.is_symlink() or manifest_path.is_symlink():
            raise BundleIntegrityError(f"bundle generation path cannot be a symbolic link: {uri}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BundleIntegrityError(f"bundle manifest is unreadable: {uri}") from exc
        if pointer.get("manifest_digest") != manifest.get("manifest_digest"):
            raise BundleIntegrityError(f"bundle pointer references a different manifest: {uri}")
        return self.validate_generation(uri, generation, manifest)

    def validate_generation(
        self,
        uri: str,
        generation: Path,
        manifest: dict,
    ) -> tuple[ContextObject, str]:
        """校验 generation 的路径、manifest、组件摘要和关系一致性。"""

        component_paths = (
            generation / ".meta.json",
            generation / ".relations.json",
            generation / "content.md",
        )
        if generation.is_symlink() or any(path.is_symlink() for path in component_paths):
            raise BundleIntegrityError(f"bundle generation component cannot be a symbolic link: {uri}")
        core = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        if (
            manifest.get("schema_version") != "source_object_bundle_v1"
            or manifest.get("uri") != uri
            or manifest.get("tenant_id") != self.tenant_id
            or manifest.get("generation_id") != generation.name
            or manifest.get("manifest_digest") != canonical_digest(core)
        ):
            raise BundleIntegrityError(f"bundle manifest integrity check failed: {uri}")
        try:
            object_payload = json.loads((generation / ".meta.json").read_text(encoding="utf-8"))
            relations_payload = json.loads((generation / ".relations.json").read_text(encoding="utf-8"))
            content = (generation / "content.md").read_text(encoding="utf-8")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BundleIntegrityError(f"bundle generation is incomplete: {uri}") from exc
        if (
            manifest.get("object_digest") != canonical_digest(object_payload)
            or manifest.get("relations_digest") != canonical_digest(relations_payload)
            or manifest.get("content_digest") != canonical_digest(content)
            or relations_payload.get("uri") != uri
            or object_payload.get("uri") != uri
            or canonical_digest(relations_payload.get("relations", []))
            != canonical_digest(object_payload.get("relations", []))
        ):
            raise BundleIntegrityError(f"bundle generation digest mismatch: {uri}")
        return ContextObject.from_dict(object_payload), content

    def resolve_content_pointer(self, uri: str) -> tuple[str, Path] | None:
        """把对象自身或其 L2 子 URI 解析到同一个 bundle 指针。"""

        candidates = [uri]
        leaf = uri.rsplit("/", 1)[-1].casefold()
        if leaf in {"content.md", "l2.json", "l2.md"} and "/" in uri:
            # L2 是原子对象 bundle 的一部分；L0/L1 是独立派生物，不走此解析路径。
            candidates.insert(0, uri.rsplit("/", 1)[0])
        for candidate in candidates:
            try:
                pointer = self._object_dir(candidate) / ".bundle-current.json"
            except ValueError:
                continue
            if pointer.exists() or pointer.is_symlink():
                return candidate, pointer
        return None

    def _notify(self, stage: str, uri: str, generation_id: str) -> None:
        hook = self._test_hook()
        if callable(hook):
            hook(stage, uri, generation_id)

    def _object_dir(self, uri: str) -> Path:
        return ContextURI.parse(uri).to_source_path(self.root, tenant_id=self.tenant_id)


__all__ = ["BundleIntegrityError", "SourceBundleStore", "SourceFileIO"]
