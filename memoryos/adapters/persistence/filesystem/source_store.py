"""Filesystem implementation of the authoritative SourceStore protocol."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

from memoryos.adapters.persistence.in_memory.lock_store import InMemoryLockStore
from memoryos.contextdb.extensions import ContextDomainClassifier, NoDomainOverlay
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.integrity import canonical_digest


class BundleIntegrityError(RuntimeError):
    """A versioned SourceStore object bundle is incomplete or corrupt."""


class FileSystemSourceStore:
    """负责 FileSystemSourceStore 的持久化读写。"""

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
        self.domain_classifier = domain_classifier or NoDomainOverlay()
        self._operation_lock_store = InMemoryLockStore()
        self.test_hook: Callable[[str, str, str], None] | None = None

    def operation_lock_store(self) -> InMemoryLockStore:
        """Return the adapter-owned fallback used by direct in-process facades."""

        return self._operation_lock_store

    def read_object(self, uri: str) -> ContextObject:
        directory = self._object_dir(uri)
        pointer = directory / ".bundle-current.json"
        if pointer.exists() or pointer.is_symlink():
            obj, _content = self._read_bundle(uri, pointer)
        else:
            path = directory / ".meta.json"
            if path.is_symlink():
                raise BundleIntegrityError(f"object metadata cannot be a symbolic link: {uri}")
            obj = ContextObject.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if ContextURI.parse(uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
            raise FileNotFoundError(uri)
        return obj

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None:
        if ContextURI.parse(obj.uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
            raise PermissionError("ContextObject tenant does not match SourceStore tenant")
        if self._requires_versioned_bundle(obj):
            self._write_bundle(obj, content)
            return
        directory = self._object_dir(obj.uri)
        self._ensure_private_directory(directory)
        self._write_atomic(directory / ".meta.json", json.dumps(obj.to_dict(), ensure_ascii=False, indent=2))
        relations = {"uri": obj.uri, "relations": [relation.to_dict() for relation in obj.relations]}
        self._write_atomic(directory / ".relations.json", json.dumps(relations, ensure_ascii=False, indent=2))
        if content:
            self.write_content(obj.layers.l2_uri or obj.uri, content)

    def read_content(self, uri: str) -> str:
        bundle = self._bundle_for_content_uri(uri)
        if bundle is not None:
            _obj, content = self._read_bundle(bundle[0], bundle[1])
            return content
        path = self._content_path(uri)
        if path.is_symlink():
            raise BundleIntegrityError(f"object content cannot be a symbolic link: {uri}")
        return path.read_text(encoding="utf-8")

    def write_content(self, uri: str, content: str | bytes) -> None:
        bundle = self._bundle_for_content_uri(uri)
        if bundle is not None:
            obj, _old_content = self._read_bundle(bundle[0], bundle[1])
            self._write_bundle(obj, content, preserve_existing_content=False)
            return
        path = self._content_path(uri)
        self._ensure_private_directory(path.parent)
        if path.is_symlink():
            raise BundleIntegrityError(f"object content cannot be a symbolic link: {uri}")
        if isinstance(content, bytes):
            tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
            try:
                tmp.write_bytes(content)
                os.chmod(tmp, 0o600)
                if path.is_symlink():
                    raise BundleIntegrityError(f"object content cannot be a symbolic link: {uri}")
                os.replace(tmp, path)
                os.chmod(path, 0o600)
            finally:
                tmp.unlink(missing_ok=True)
        else:
            self._write_atomic(path, content)

    def soft_delete(self, uri: str, reason: str) -> None:
        obj = self.read_object(uri)
        obj.lifecycle_state = LifecycleState.DELETED
        obj.metadata = {**obj.metadata, "delete_reason": reason}
        self.write_object(obj)

    def delete_object(self, uri: str) -> None:
        directory = self._object_dir(uri)
        if directory.exists():
            shutil.rmtree(directory)

    def list_objects(self) -> list[ContextObject]:
        if not self.root.exists():
            return []
        objects = []
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
                obj, _content = self._read_bundle(uri, pointer)
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise BundleIntegrityError(f"cannot enumerate corrupt object bundle: {pointer}") from exc
            if ContextURI.parse(obj.uri).authority == "user" and str(obj.tenant_id or "default") != self.tenant_id:
                continue
            by_uri[obj.uri] = obj
        objects = list(by_uri.values())
        return objects

    def _requires_versioned_bundle(self, obj: ContextObject) -> bool:
        return self.domain_classifier.owns_object(obj)

    def _write_bundle(
        self,
        obj: ContextObject,
        content: str | bytes,
        *,
        preserve_existing_content: bool = True,
    ) -> None:
        directory = self._object_dir(obj.uri)
        pointer = directory / ".bundle-current.json"
        if pointer.is_symlink():
            raise BundleIntegrityError(f"bundle pointer cannot be a symbolic link: {obj.uri}")
        if preserve_existing_content and content in {"", b""} and pointer.exists():
            # SourceStore.write_object historically updates metadata without
            # clearing L2. Preserve that semantic while publishing a complete
            # new generation; callers that change L2 use write_content or pass
            # non-empty content explicitly.
            _current_object, current_content = self._read_bundle(obj.uri, pointer)
            content = current_content
        generations = directory / ".bundle-generations"
        generation_id = uuid.uuid4().hex
        generation = generations / generation_id
        self._ensure_private_directory(generation)
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
        self._write_atomic(meta_path, json.dumps(object_payload, ensure_ascii=False, indent=2, sort_keys=True))
        self._notify_bundle("after_meta", obj.uri, generation_id)
        self._write_atomic(
            relations_path,
            json.dumps(relations_payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        self._notify_bundle("after_relations", obj.uri, generation_id)
        self._write_atomic(content_path, encoded_content)
        self._notify_bundle("after_content", obj.uri, generation_id)
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
        self._write_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        self._notify_bundle("after_bundle_manifest", obj.uri, generation_id)
        self._validate_generation(obj.uri, generation, manifest)
        # Component writes fsync the generation itself.  Persist the
        # generation directory entry in its parent before publishing a
        # pointer that can make this bundle visible.
        self._fsync_directory(generations)
        self._notify_bundle("before_bundle_publish", obj.uri, generation_id)
        self._notify_bundle("before_current_pointer", obj.uri, generation_id)
        pointer_core = {
            "schema_version": "source_object_bundle_current_v1",
            "uri": obj.uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "generation_id": generation_id,
            "manifest_digest": manifest["manifest_digest"],
        }
        self._write_atomic(
            directory / ".bundle-current.json",
            json.dumps(
                {**pointer_core, "pointer_digest": canonical_digest(pointer_core)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )
        self._notify_bundle("after_current_pointer", obj.uri, generation_id)
        self._notify_bundle("after_bundle_publish", obj.uri, generation_id)

    def _read_bundle(self, uri: str, pointer_path: Path) -> tuple[ContextObject, str]:
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
        return self._validate_generation(uri, generation, manifest)

    def _validate_generation(
        self,
        uri: str,
        generation: Path,
        manifest: dict,
    ) -> tuple[ContextObject, str]:
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

    def _bundle_for_content_uri(self, uri: str) -> tuple[str, Path] | None:
        candidates = [uri]
        leaf = uri.rsplit("/", 1)[-1].casefold()
        if leaf in {"content.md", "l2.json", "l2.md"} and "/" in uri:
            # Canonical objects may record their durable L2 as a child URI.
            # That content is nevertheless part of the atomic object bundle;
            # L0/L1 projection URIs intentionally remain independent derived
            # artifacts and are therefore not resolved through this path.
            candidates.insert(0, uri.rsplit("/", 1)[0])
        for candidate in candidates:
            try:
                pointer = self._object_dir(candidate) / ".bundle-current.json"
            except ValueError:
                continue
            if pointer.exists() or pointer.is_symlink():
                return candidate, pointer
        return None

    def _notify_bundle(self, stage: str, uri: str, generation_id: str) -> None:
        hook = getattr(self, "test_hook", None)
        if callable(hook):
            hook(stage, uri, generation_id)

    def _object_dir(self, uri: str) -> Path:
        return ContextURI.parse(uri).to_source_path(self.root, tenant_id=self.tenant_id)

    def _content_path(self, uri: str) -> Path:
        parsed = ContextURI.parse(uri)
        path = parsed.to_source_path(self.root, tenant_id=self.tenant_id)
        if path.suffix:
            return path
        return path / "content.md"

    def _write_atomic(self, path: Path, content: str) -> None:
        if path.is_symlink():
            raise BundleIntegrityError(f"SourceStore path cannot be a symbolic link: {path}")
        self._ensure_private_directory(path.parent)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("x", encoding="utf-8") as handle:
                os.chmod(tmp, 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise BundleIntegrityError(f"SourceStore path cannot be a symbolic link: {path}")
            os.replace(tmp, path)
            os.chmod(path, 0o600)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _ensure_private_directory(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        durable_chain: list[Path] = []
        current = directory
        root = self.root
        while current == root or root in current.parents:
            durable_chain.append(current)
            try:
                current.chmod(0o700)
            except OSError:
                pass
            if current == root:
                break
            current = current.parent
        # ``fsync(file)`` and ``fsync(the file's directory)`` do not make a
        # newly-created ancestor entry durable.  Persist every directory in
        # the chain so a crash cannot retain a bundle pointer while losing an
        # object/generation directory that the pointer traverses.
        for current in durable_chain:
            self._fsync_directory(current)
