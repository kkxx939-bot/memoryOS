"""上下文数据库里的本地存储集合。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import (
    IndexHit,
    LeaseLostError,
    LockLostError,
    LockToken,
    QueueIdempotencyConflictError,
    QueueJob,
    QueueLeaseIdentityError,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)
from memoryos.contextdb.store.sqlite_index_store import lexical_match_count, lexical_relevance, lexical_terms


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BundleIntegrityError(RuntimeError):
    """A versioned SourceStore object bundle is incomplete or corrupt."""


class FileSystemSourceStore:
    """负责 FileSystemSourceStore 的持久化读写。"""

    def __init__(self, root: str | Path, tenant_id: str = "default") -> None:
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
        self.test_hook: Callable[[str, str, str], None] | None = None

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
        return is_canonical_memory_object(obj)

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


class InMemoryIndexStore:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[ContextObject, str]] = {}

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        self.rows[obj.uri] = (obj, content)

    def delete_index(self, uri: str) -> None:
        self.rows.pop(uri, None)

    def indexed_uris(self) -> list[str]:
        return list(self.rows)

    def get_index_metadata(self, uri: str) -> dict | None:
        row = self.rows.get(uri)
        return (
            {
                **dict(row[0].metadata or {}),
                "tenant_id": str(row[0].tenant_id or ""),
                "owner_user_id": str(row[0].owner_user_id or ""),
                "context_type": row[0].context_type.value,
                "claim_state": str(
                    dict(row[0].metadata or {}).get("state") or dict(row[0].metadata or {}).get("claim_state") or ""
                ),
                "index_content_digest": canonical_digest(row[1]),
            }
            if row is not None
            else None
        )

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str:
        del session_id
        row = self.rows.get(uri)
        if row is None:
            return "missing"
        obj = row[0]
        if str(obj.tenant_id or "default") != str(tenant_id):
            return "missing"
        if obj.lifecycle_state in {
            LifecycleState.DELETED,
            LifecycleState.ARCHIVED,
            LifecycleState.OBSOLETE,
        }:
            return "inactive"
        return "active"

    def clear(self) -> None:
        self.rows.clear()

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        filters = filters or {}
        hits = []
        for obj, content in self.rows.values():
            if "allowed_uris" in filters and obj.uri not in set(filters.get("allowed_uris", []) or []):
                continue
            if filters.get("lifecycle_state") is None and obj.lifecycle_state in {
                LifecycleState.DELETED,
                LifecycleState.ARCHIVED,
                LifecycleState.OBSOLETE,
            }:
                continue
            if filters.get("lifecycle_state") and obj.lifecycle_state.value != filters["lifecycle_state"]:
                continue
            if filters.get("principal_owner_id") is not None:
                expected_owner = str(filters["principal_owner_id"])
                metadata = dict(obj.metadata or {})
                raw_scope = dict(metadata.get("scope", {}) or {})
                raw_applicability = dict(raw_scope.get("applicability", {}) or {})
                raw_visibility = dict(raw_scope.get("visibility", {}) or {})
                workspace = str(
                    metadata.get("workspace_id")
                    or metadata.get("project_id")
                    or next(
                        (
                            str(item.get("id"))
                            for item in raw_applicability.get("all_of", []) or []
                            if isinstance(item, dict) and item.get("kind") == "workspace"
                        ),
                        "",
                    )
                )
                shared_workspaces = {
                    str(value)
                    for value in filters.get("workspace_access_ids", ()) or ()
                    if str(value) not in {"", "__memoryos_principal_only__"}
                }
                record_kind = str(metadata.get("record_kind") or "")
                if not record_kind and str(metadata.get("canonical_kind") or "") == "claim":
                    record_kind = "claim_revision"
                canonical_shared = bool(
                    obj.context_type == ContextType.MEMORY
                    and record_kind in {"current_slot", "claim_revision"}
                    and str(metadata.get("slot_id") or metadata.get("canonical_slot_id") or "")
                    and str(metadata.get("claim_id") or metadata.get("canonical_claim_id") or "")
                    and str(raw_visibility.get("tenant_id") or "") == str(obj.tenant_id or "default")
                    and (
                        expected_owner
                        in {str(item) for item in raw_visibility.get("allowed_principal_ids", ()) or ()}
                        or (
                            raw_visibility.get("private") is False
                            and raw_visibility.get("allowed_principal_ids") in ([], ())
                            and raw_visibility.get("allowed_service_ids") in ([], ())
                            and workspace in shared_workspaces
                        )
                    )
                )
                if (
                    obj.owner_user_id != expected_owner
                    and not (
                        obj.owner_user_id in {None, ""}
                        and obj.context_type in {ContextType.RESOURCE, ContextType.SKILL}
                    )
                    and not canonical_shared
                ):
                    continue
            elif filters.get("owner_user_id") is not None:
                expected_owner = str(filters["owner_user_id"])
                if expected_owner:
                    if obj.context_type not in {ContextType.RESOURCE, ContextType.SKILL}:
                        if obj.owner_user_id != expected_owner:
                            continue
                    elif obj.owner_user_id not in {None, "", expected_owner}:
                        continue
                elif obj.owner_user_id not in {None, ""}:
                    continue
            if filters.get("tenant_id") and str(obj.tenant_id or "default") != str(filters["tenant_id"]):
                continue
            if filters.get("context_type") and obj.context_type.value != filters["context_type"]:
                continue
            metadata = dict(obj.metadata or {})
            if filters.get("adapter_access_id") is not None:
                actual_adapter = str(
                    metadata.get("source_adapter_id") or dict(metadata.get("connect", {}) or {}).get("adapter_id") or ""
                )
                if actual_adapter not in {"", str(filters["adapter_access_id"])}:
                    record_kind = str(metadata.get("record_kind") or "")
                    if obj.context_type not in {ContextType.SESSION, ContextType.RESOURCE, ContextType.SKILL} and (
                        record_kind != "current_slot"
                    ):
                        continue
            context_types = filters.get("context_types")
            if context_types is not None and obj.context_type.value not in {str(value) for value in context_types}:
                continue
            source_kinds = filters.get("source_kinds")
            if source_kinds is not None and str(metadata.get("source_kind") or "context") not in {
                str(value) for value in source_kinds
            }:
                continue
            record_kinds = filters.get("record_kinds")
            if record_kinds is not None:
                actual_record_kind = str(
                    metadata.get("record_kind")
                    or ("claim_revision" if metadata.get("canonical_kind") == "claim" else "context")
                )
                if actual_record_kind not in {str(value) for value in record_kinds}:
                    continue
            try:
                from memoryos.memory.canonical.scope import scope_keys_from_payloads

                raw_scope = metadata.get("scope", {}) or {}
                if not isinstance(raw_scope, dict):
                    continue
                raw_applicability = raw_scope.get("applicability", {}) or {}
                if not isinstance(raw_applicability, dict):
                    continue
                actual_scope_keys = set(scope_keys_from_payloads(raw_applicability.get("all_of", [])))
            except (KeyError, TypeError, ValueError):
                continue
            admission = dict(metadata.get("admission", {}) or {})
            excluded_admission = {"restricted", "archive_only", "reject"}
            if not filters.get("include_candidates"):
                excluded_admission.add("pending")
            if filters.get("admission_status") is None and admission.get("decision") in excluded_admission:
                continue
            if filters.get("project_id"):
                scope = raw_scope
                fields = dict(metadata.get("fields", {}) or {})
                applicability = raw_applicability
                workspace = next(
                    (
                        str(item.get("id"))
                        for item in applicability.get("all_of", []) or []
                        if isinstance(item, dict) and item.get("kind") == "workspace"
                    ),
                    "",
                )
                project_id = str(
                    scope.get("project_id") or fields.get("project_id") or metadata.get("project_id") or workspace
                )
                memory_type = str(metadata.get("memory_type") or "")
                if memory_type in {"project_rule", "project_decision", "agent_experience"} and project_id != str(
                    filters["project_id"]
                ):
                    continue
            workspace_access = filters.get("workspace_access_ids")
            if workspace_access is not None:
                scope = raw_scope
                fields = dict(metadata.get("fields", {}) or {})
                applicability = raw_applicability
                workspace = str(
                    metadata.get("workspace_id")
                    or metadata.get("project_id")
                    or scope.get("project_id")
                    or fields.get("project_id")
                    or next(
                        (
                            str(item.get("id"))
                            for item in applicability.get("all_of", []) or []
                            if isinstance(item, dict) and item.get("kind") == "workspace"
                        ),
                        "",
                    )
                )
                if workspace not in {str(value) for value in workspace_access}:
                    continue
            metadata_matches = True
            for field in ("adapter_id", "admission_status", "claim_state", "slot_id", "memory_type"):
                expected = filters.get(field)
                if expected is None:
                    continue
                values = set(expected) if isinstance(expected, list | tuple | set | frozenset) else {expected}
                actual = {
                    "adapter_id": metadata.get("source_adapter_id")
                    or dict(metadata.get("connect", {}) or {}).get("adapter_id"),
                    "admission_status": dict(metadata.get("admission", {}) or {}).get("decision"),
                    "claim_state": metadata.get("state") or metadata.get("claim_state"),
                    "slot_id": metadata.get("slot_id"),
                    "memory_type": metadata.get("memory_type"),
                }[field]
                if actual not in values:
                    metadata_matches = False
                    break
            if not metadata_matches:
                continue
            required_scopes = set(filters.get("applicability_scope_keys", []) or [])
            if required_scopes and not actual_scope_keys.issubset(required_scopes):
                continue
            if filters.get("require_unscoped") and actual_scope_keys:
                continue
            text = " ".join([obj.title, content, json.dumps(obj.metadata, ensure_ascii=False)]).casefold()
            lexical_matches = lexical_match_count(query, text)
            lexical = lexical_relevance(query, text)
            identity = (
                1.0
                if any(
                    str(metadata.get(field, "")) == str(query).strip()
                    for field in {"scene_key", "action", "memory_anchor_uri"}
                )
                else 0.0
            )
            base_relevance = max(lexical, identity)
            if base_relevance <= 0:
                continue
            hotness = (obj.hotness + obj.semantic_hotness + obj.behavior_support_hotness) / 3.0
            score = max(float(lexical_matches), identity) + 0.05 * hotness
            canonical_projection = bool(
                is_canonical_memory_object(obj)
                or str(metadata.get("canonical_kind") or "")
                in {"slot", "claim", "pending_proposal", "current_slot_projection"}
                or str(metadata.get("schema_version") or "").startswith("canonical_")
            )
            ordinary_serving_metadata = {
                key: metadata[key]
                for key in (
                    "adapter_id",
                    "admission",
                    "connect",
                    "memory_type",
                    "project_id",
                    "record_kind",
                    "retrieval_views",
                    "scope",
                    "scope_keys",
                    "session_id",
                    "source_adapter_id",
                    "source_kind",
                    "workspace_id",
                )
                if key in metadata
            }
            hit_metadata = {
                # Preserve the pre-existing ordinary-context fallback contract:
                # Behavior/Prediction consumers re-read Source for business
                # metadata. Only controlled serving/scope fields cross the
                # legacy IndexHit boundary; arbitrary reward/event payloads
                # do not affect Behavior selection. Canonical offline
                # validation requires the full projector proof envelope.
                **(metadata if canonical_projection else ordinary_serving_metadata),
                "tenant_id": str(obj.tenant_id or "default"),
                "owner_user_id": str(obj.owner_user_id or ""),
                "context_type": obj.context_type.value,
                "retrieval_scores": {
                    "lexical": lexical,
                    "vector": 0.0,
                    "identity": identity,
                    "base_relevance": base_relevance,
                    "hotness": hotness,
                    "score": score,
                },
            }
            hits.append(
                IndexHit(
                    uri=obj.uri,
                    score=score,
                    context_type=obj.context_type.value,
                    title=obj.title,
                    metadata=hit_metadata,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _lexical_terms(self, query: str) -> tuple[str, ...]:
        return lexical_terms(query)


class InMemoryRelationStore:
    def __init__(self) -> None:
        self.relations: list[ContextRelation] = []

    def add_relation(self, relation: ContextRelation) -> None:
        if relation not in self.relations:
            self.relations.append(relation)

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
        limit: int | None = None,
    ) -> list[ContextRelation]:
        rows = [relation for relation in self.relations if relation.source_uri == uri or relation.target_uri == uri]
        if tenant_id is not None:
            rows = [relation for relation in rows if relation.metadata.get("tenant_id", "default") == tenant_id]
        if owner_user_id is not None:
            rows = [
                relation
                for relation in rows
                if relation.metadata.get("owner_user_id") in {None, "", owner_user_id}
                or relation.target_uri.startswith(("memoryos://resources/", "memoryos://skills/"))
            ]
        rows.sort(key=lambda item: (-item.weight, item.created_at, item.source_uri, item.target_uri))
        return rows[: max(0, int(limit))] if limit is not None else rows

    def delete_relation(
        self,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        matching_tenants = {
            str(relation.metadata.get("tenant_id") or "default")
            for relation in self.relations
            if relation.source_uri == source_uri
            and relation.relation_type == relation_type
            and relation.target_uri == target_uri
        }
        if tenant_id is None and len(matching_tenants) > 1:
            raise ValueError("tenant_id is required for an ambiguous relation identity")
        selected_tenant = tenant_id or next(iter(matching_tenants), None)
        self.relations = [
            relation
            for relation in self.relations
            if not (
                relation.source_uri == source_uri
                and relation.relation_type == relation_type
                and relation.target_uri == target_uri
                and (
                    selected_tenant is None
                    or str(relation.metadata.get("tenant_id") or "default") == selected_tenant
                )
            )
        ]

    def delete_projection_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        catalog_record_key: str,
        limit: int,
    ) -> int:
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if (relation.source_uri == uri or relation.target_uri == uri)
            and str(relation.metadata.get("tenant_id") or "default") == tenant_id
            and str(relation.metadata.get("catalog_record_key") or "") in {"", catalog_record_key}
        ][:maximum]
        identities = {
            (
                str(relation.metadata.get("tenant_id") or "default"),
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
            )
            for relation in selected
        }
        self.relations = [
            relation
            for relation in self.relations
            if (
                str(relation.metadata.get("tenant_id") or "default"),
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
            )
            not in identities
        ]
        return len(identities)

    def delete_uri_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int,
    ) -> int:
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if (relation.source_uri == uri or relation.target_uri == uri)
            and str(relation.metadata.get("tenant_id") or "default") == tenant_id
        ][:maximum]
        identities = {
            (
                str(relation.metadata.get("tenant_id") or "default"),
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
            )
            for relation in selected
        }
        self.relations = [
            relation
            for relation in self.relations
            if (
                str(relation.metadata.get("tenant_id") or "default"),
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
            )
            not in identities
        ]
        return len(identities)

    def clear_ordinary_relations(self, *, tenant_id: str, limit: int) -> int:
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if str(relation.metadata.get("tenant_id") or "default") == tenant_id
            and not is_canonical_memory_uri(relation.source_uri)
        ][:maximum]
        identities = {
            (relation.source_uri, relation.relation_type, relation.target_uri)
            for relation in selected
        }
        self.relations = [
            relation
            for relation in self.relations
            if not (
                str(relation.metadata.get("tenant_id") or "default") == tenant_id
                and (relation.source_uri, relation.relation_type, relation.target_uri) in identities
            )
        ]
        return len(identities)

    def reconcile_ordinary_relations(
        self,
        relations: Sequence[ContextRelation],
        *,
        tenant_id: str,
    ) -> dict[str, int]:
        values = tuple(relations)
        if len(values) > 1_000:
            raise ValueError("ordinary relation reconcile batch exceeds 1000")
        prepared: dict[tuple[str, str, str], ContextRelation] = {}
        for relation in values:
            relation_tenant = str(relation.metadata.get("tenant_id") or "default")
            if relation_tenant != tenant_id:
                raise ValueError("ordinary relation tenant differs from reconcile tenant")
            if is_canonical_memory_uri(relation.source_uri):
                raise ValueError("ordinary relation reconcile cannot mutate a canonical Source")
            if str(relation.metadata.get("catalog_record_key") or ""):
                raise ValueError("ordinary Source relation cannot claim Catalog projection ownership")
            identity = (relation.source_uri, relation.relation_type, relation.target_uri)
            prior = prepared.get(identity)
            if prior is not None and not self._ordinary_projection_equal(prior, relation):
                raise ValueError("ordinary relation batch contains a conflicting identity")
            prepared[identity] = relation
        written = 0
        skipped = 0
        for identity in sorted(prepared):
            relation = prepared[identity]
            existing = next(
                (
                    item
                    for item in self.relations
                    if item.source_uri == relation.source_uri
                    and item.relation_type == relation.relation_type
                    and item.target_uri == relation.target_uri
                    and str(item.metadata.get("tenant_id") or "default") == tenant_id
                ),
                None,
            )
            if existing is not None and self._ordinary_projection_equal(existing, relation):
                skipped += 1
                continue
            if existing is not None:
                self.delete_relation(
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    tenant_id=tenant_id,
                )
            self.add_relation(relation)
            written += 1
        return {"processed": len(prepared), "written": written, "skipped": skipped}

    @staticmethod
    def _ordinary_projection_equal(left: ContextRelation, right: ContextRelation) -> bool:
        return (
            left.source_uri == right.source_uri
            and left.relation_type == right.relation_type
            and left.target_uri == right.target_uri
            and left.weight == right.weight
            and dict(left.metadata or {}) == dict(right.metadata or {})
        )

    def all_relations(self) -> list[ContextRelation]:
        return list(self.relations)


class InMemoryQueueStore:
    def __init__(self) -> None:
        self.jobs: dict[str, QueueJob] = {}
        self._guard = threading.RLock()

    def enqueue(self, job: QueueJob) -> QueueJob:
        if job.status != "pending" or job.lease_token or job.lease_owner or job.lease_generation:
            raise ValueError("new queue jobs must be unleased and pending")
        with self._guard:
            existing = self.jobs.get(job.job_id)
            if existing is not None:
                if self._identity(existing) != self._identity(job):
                    raise QueueIdempotencyConflictError(
                        f"queue job id is already bound to another payload: {job.job_id}"
                    )
                return existing
            pending = QueueJob(
                job_id=job.job_id,
                queue_name=job.queue_name,
                action=job.action,
                target_uri=job.target_uri,
                payload=dict(job.payload),
            )
            self.jobs[job.job_id] = pending
            return pending

    def lease(
        self,
        queue_name: str,
        *,
        lease_owner: str,
        limit: int = 10,
        lease_seconds: int = 60,
        job_ids: Sequence[str] | None = None,
    ) -> list[QueueJob]:
        if not isinstance(lease_owner, str) or not lease_owner.strip():
            raise ValueError("lease_owner must be non-empty")
        if limit <= 0:
            return []
        now = datetime.now(timezone.utc)
        leased_until = (now + timedelta(seconds=max(1, lease_seconds))).isoformat()
        allowed = set(job_ids) if job_ids is not None else None
        leased: list[QueueJob] = []
        with self._guard:
            for job in self.jobs.values():
                expired = job.status == "leased" and self._expired(job, now)
                if (
                    job.queue_name == queue_name
                    and (allowed is None or job.job_id in allowed)
                    and (job.status == "pending" or expired)
                ):
                    claimed = QueueJob(
                        **{
                            **job.__dict__,
                            "status": "leased",
                            "leased_until": leased_until,
                            "lease_token": uuid.uuid4().hex,
                            "lease_generation": job.lease_generation + 1,
                            "lease_owner": lease_owner,
                        }
                    )
                    self.jobs[job.job_id] = claimed
                    leased.append(claimed)
                if len(leased) >= limit:
                    break
        return leased

    def ack(self, job: QueueJob) -> QueueJob:
        return self._settle(job, status="done")

    def fail(self, job: QueueJob, error: str) -> QueueJob:
        return self._settle(
            job,
            status="dead_letter",
            retry_count=job.retry_count + 1,
            last_error=str(error)[:500],
        )

    def retry(
        self,
        job: QueueJob,
        error: str,
        *,
        max_retries: int = 3,
        retryable: bool = True,
    ) -> QueueJob:
        retry_count = job.retry_count + 1
        status = "pending" if retryable and retry_count < max_retries else "dead_letter"
        return self._settle(
            job,
            status=status,
            retry_count=retry_count,
            last_error=str(error)[:500],
        )

    def release(self, job: QueueJob, reason: str = "") -> QueueJob:
        """Return an unattempted owned lease without consuming retry budget."""

        return self._settle(
            job,
            status="pending",
            last_error=str(reason)[:500] if reason else job.last_error,
        )

    def quarantine(self, job: QueueJob, error: str) -> QueueJob:
        return self._settle(
            job,
            status="quarantine",
            retry_count=job.retry_count + 1,
            last_error=str(error)[:500],
        )

    def quarantine_identity_conflict(self, job: QueueJob, error: str) -> QueueJob:
        """Quarantine an owned lease whose immutable payload was corrupted."""

        return self._settle(
            job,
            status="quarantine",
            retry_count=job.retry_count + 1,
            last_error=str(error)[:500],
            verify_identity=False,
        )

    def extend(self, job: QueueJob, *, lease_seconds: int = 60) -> QueueJob:
        with self._guard:
            current = self._owned(job)
            extended = QueueJob(
                **{
                    **current.__dict__,
                    "leased_until": (datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))).isoformat(),
                }
            )
            self.jobs[job.job_id] = extended
        return extended

    def get(self, job_id: str) -> QueueJob | None:
        with self._guard:
            return self.jobs.get(job_id)

    def recover_expired_leases(self, *, queue_name: str | None = None) -> int:
        recovered = 0
        now = datetime.now(timezone.utc)
        with self._guard:
            for job_id, job in tuple(self.jobs.items()):
                if (
                    job.status != "leased"
                    or (queue_name is not None and job.queue_name != queue_name)
                    or not self._expired(job, now)
                ):
                    continue
                self.jobs[job_id] = QueueJob(
                    **{
                        **job.__dict__,
                        "status": "pending",
                        "leased_until": None,
                        "lease_token": "",
                        "lease_owner": "",
                    }
                )
                recovered += 1
        return recovered

    def stats(self, *, queue_name: str | None = None) -> dict[str, int]:
        result: dict[str, int] = {}
        with self._guard:
            for job in self.jobs.values():
                if queue_name is not None and job.queue_name != queue_name:
                    continue
                result[job.status] = result.get(job.status, 0) + 1
        return result

    def stats_for_target_prefix(self, *, queue_name: str, target_uri_prefix: str) -> dict[str, int]:
        result: dict[str, int] = {}
        with self._guard:
            for job in self.jobs.values():
                if job.queue_name != queue_name or not job.target_uri.startswith(target_uri_prefix):
                    continue
                result[job.status] = result.get(job.status, 0) + 1
        return result

    def stats_for_scope(
        self,
        *,
        queue_name: str,
        tenant_id: str,
        owner_user_id: str,
        workspace_ids: Sequence[str] | None = None,
    ) -> dict[str, int]:
        allowed_workspaces = None if workspace_ids is None else {str(item) for item in workspace_ids}
        result: dict[str, int] = {}
        with self._guard:
            for job in self.jobs.values():
                if job.queue_name != queue_name:
                    continue
                job_tenant, job_owner, job_workspace = self._job_scope(job)
                scope_matches = (
                    job_tenant == tenant_id
                    and job_owner == owner_user_id
                    and (allowed_workspaces is None or job_workspace in allowed_workspaces)
                )
                # A pre-scope unresolved job cannot be attributed safely. It
                # blocks scoped CURRENT reads inside its own Tenant until
                # replay drains it, but must not affect another Tenant.
                unknown_unresolved = job_tenant == tenant_id and not job_owner and job.status in {
                    "pending",
                    "leased",
                    "dead_letter",
                    "quarantine",
                }
                if not scope_matches and not unknown_unresolved:
                    continue
                result[job.status] = result.get(job.status, 0) + 1
        return result

    @staticmethod
    def _job_scope(job: QueueJob) -> tuple[str, str, str]:
        payload = dict(job.payload or {})
        tenant_id = str(payload.get("tenant_id") or "default")
        owner_user_id = str(payload.get("owner_user_id") or "")
        if not owner_user_id and job.target_uri.startswith("memoryos://user/"):
            candidate = job.target_uri.removeprefix("memoryos://user/").split("/", 1)[0]
            if candidate and not candidate.startswith("subject_"):
                owner_user_id = candidate
        return tenant_id, owner_user_id, str(payload.get("workspace_id") or "")

    def _settle(
        self,
        job: QueueJob,
        *,
        status: str,
        retry_count: int | None = None,
        last_error: str | None = None,
        verify_identity: bool = True,
    ) -> QueueJob:
        with self._guard:
            current = self._owned(job, verify_identity=verify_identity)
            settled = QueueJob(
                **{
                    **current.__dict__,
                    "status": status,
                    "leased_until": None,
                    "lease_token": "",
                    "lease_owner": "",
                    "retry_count": current.retry_count if retry_count is None else retry_count,
                    "last_error": current.last_error if last_error is None else last_error,
                }
            )
            self.jobs[job.job_id] = settled
            return settled

    def _owned(self, job: QueueJob, *, verify_identity: bool = True) -> QueueJob:
        current = self.jobs.get(job.job_id)
        now = datetime.now(timezone.utc)
        if (
            current is None
            or current.status != "leased"
            or current.lease_token != job.lease_token
            or current.lease_generation != job.lease_generation
            or current.lease_owner != job.lease_owner
            or self._expired(current, now)
        ):
            raise LeaseLostError(f"queue lease lost for {job.job_id} generation {job.lease_generation}")
        if verify_identity and self._identity(current) != self._identity(job):
            raise QueueLeaseIdentityError(f"queue immutable identity changed while leased: {job.job_id}")
        return current

    def _expired(self, job: QueueJob, now: datetime) -> bool:
        if not job.leased_until:
            return True
        try:
            expires = datetime.fromisoformat(job.leased_until.replace("Z", "+00:00"))
        except ValueError:
            return True
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires.astimezone(timezone.utc) <= now

    def _identity(self, job: QueueJob) -> tuple[str, str, str, str]:
        payload = json.dumps(job.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return job.queue_name, job.action, job.target_uri, payload


class InMemoryLockStore:
    def __init__(self) -> None:
        self.locks: dict[str, tuple[str, int, datetime]] = {}
        self.fences: dict[str, int] = {}
        self._guard = threading.RLock()

    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken:
        with self._guard:
            now = datetime.now(timezone.utc)
            existing = self.locks.get(lock_key)
            if existing is not None and existing[2] > now:
                raise TimeoutError(f"Lock already held: {lock_key}")
            fence = self.fences.get(lock_key, 0) + 1
            self.fences[lock_key] = fence
            token = uuid.uuid4().hex
            self.locks[lock_key] = (
                token,
                fence,
                now + timedelta(seconds=max(1, ttl_seconds)),
            )
            return LockToken(lock_key=lock_key, token=token, fence=fence)

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> LockToken:
        with self._guard:
            self._assert_owned_unlocked(token)
            self.locks[token.lock_key] = (
                token.token,
                token.fence,
                datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds)),
            )
        return token

    def assert_owned(self, token: LockToken) -> None:
        with self._guard:
            self._assert_owned_unlocked(token)

    @contextmanager
    def fenced(self, tokens: Sequence[LockToken], ttl_seconds: int = 30) -> Iterator[None]:
        with self._guard:
            for token in tokens:
                self._assert_owned_unlocked(token)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
            for token in tokens:
                self.locks[token.lock_key] = (token.token, token.fence, expires_at)
            yield
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
            for token in tokens:
                self._assert_identity_unlocked(token)
                self.locks[token.lock_key] = (token.token, token.fence, expires_at)

    def release(self, token: LockToken) -> None:
        with self._guard:
            current = self.locks.get(token.lock_key)
            if current is not None and current[:2] == (token.token, token.fence):
                self.locks.pop(token.lock_key, None)

    def _assert_owned_unlocked(self, token: LockToken) -> None:
        self._assert_identity_unlocked(token)
        current = self.locks[token.lock_key]
        if current[2] <= datetime.now(timezone.utc):
            raise LockLostError(f"Lock lease lost: {token.lock_key}")

    def _assert_identity_unlocked(self, token: LockToken) -> None:
        current = self.locks.get(token.lock_key)
        if current is None or current[:2] != (token.token, token.fence):
            raise LockLostError(f"Lock lease lost: {token.lock_key}")
