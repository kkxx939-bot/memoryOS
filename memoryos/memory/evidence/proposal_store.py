"""Immutable sealed proposals plus content-free task/document lineage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memoryos.core.clock import utc_now
from memoryos.core.durable_io import atomic_create_json, atomic_write_json
from memoryos.core.durable_io.atomic_file import _open_control_parent
from memoryos.core.file_lock import open_private_lock
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest
from memoryos.memory.documents.erase import DocumentErasedError, MemoryDocumentEraseStore
from memoryos.memory.documents.frontmatter import validate_document_id
from memoryos.memory.documents.model import MemoryEditProposal

_PROPOSAL_SCHEMA = "sealed_memory_edit_proposals_v1"
_BINDING_CATALOG_SCHEMA = "sealed_proposal_document_bindings_v1"
_ERASURE_BARRIER_SCHEMA = "sealed_proposal_erasure_barrier_v1"
_MAX_PROPOSAL_BYTES = 4 * 1024 * 1024
_MAX_BINDING_CATALOG_BYTES = 16 * 1024 * 1024
_MAX_BARRIER_BYTES = 256 * 1024


class SealedProposalIntegrityError(RuntimeError):
    """A body-bearing proposal artifact or its exact lineage is unsafe."""


@dataclass(frozen=True)
class SealedProposalSet:
    task_id: str
    tenant_id: str
    owner_user_id: str
    archive_uri: str
    archive_digest: str
    manifest_digest: str
    proposals: tuple[MemoryEditProposal, ...]
    proposal_set_digest: str


@dataclass(frozen=True)
class ProposalDocumentBinding:
    """One content-free task-to-document effect fingerprint."""

    document_id: str
    change_digest: str

    def __post_init__(self) -> None:
        validate_document_id(self.document_id)
        _require_digest(self.change_digest, "proposal document change digest")

    def to_dict(self) -> dict[str, str]:
        return {"document_id": self.document_id, "change_digest": self.change_digest}

    @classmethod
    def from_dict(cls, payload: object) -> ProposalDocumentBinding:
        if not isinstance(payload, dict) or set(payload) != {"document_id", "change_digest"}:
            raise SealedProposalIntegrityError("proposal document binding is malformed")
        try:
            return cls(
                document_id=str(payload["document_id"]),
                change_digest=str(payload["change_digest"]),
            )
        except (TypeError, ValueError) as exc:
            raise SealedProposalIntegrityError("proposal document binding is invalid") from exc


@dataclass(frozen=True)
class SealedProposalBindingSet:
    """Exact content-free lineage for one sealed Session task."""

    task_id: str
    tenant_id: str
    owner_user_id: str
    proposal_set_digest: str
    documents: tuple[ProposalDocumentBinding, ...]
    binding_digest: str

    def __post_init__(self) -> None:
        _validate_task_id(self.task_id)
        require_safe_path_segment(self.tenant_id, "proposal binding tenant_id")
        require_safe_path_segment(self.owner_user_id, "proposal binding owner_user_id")
        _require_digest(self.proposal_set_digest, "proposal set digest")
        if not self.documents:
            raise ValueError("proposal binding requires at least one document")
        if tuple(sorted(self.documents, key=lambda item: item.document_id)) != self.documents:
            raise ValueError("proposal bindings must be sorted by document identity")
        if len({item.document_id for item in self.documents}) != len(self.documents):
            raise ValueError("proposal binding repeats a document identity")
        if self.binding_digest != canonical_digest(self._digest_payload()):
            raise ValueError("proposal binding digest is invalid")

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "proposal_set_digest": self.proposal_set_digest,
            "documents": [item.to_dict() for item in self.documents],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._digest_payload(), "binding_digest": self.binding_digest}

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        tenant_id: str,
        owner_user_id: str,
        proposal_set_digest: str,
        documents: Iterable[ProposalDocumentBinding],
    ) -> SealedProposalBindingSet:
        ordered = tuple(sorted(documents, key=lambda item: item.document_id))
        payload = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "owner_user_id": owner_user_id,
            "proposal_set_digest": proposal_set_digest,
            "documents": [item.to_dict() for item in ordered],
        }
        return cls(
            task_id=task_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            proposal_set_digest=proposal_set_digest,
            documents=ordered,
            binding_digest=canonical_digest(payload),
        )

    @classmethod
    def from_dict(cls, payload: object) -> SealedProposalBindingSet:
        expected = {
            "task_id",
            "tenant_id",
            "owner_user_id",
            "proposal_set_digest",
            "documents",
            "binding_digest",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise SealedProposalIntegrityError("sealed proposal binding is malformed")
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            raise SealedProposalIntegrityError("sealed proposal documents must be an array")
        try:
            return cls(
                task_id=str(payload["task_id"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                proposal_set_digest=str(payload["proposal_set_digest"]),
                documents=tuple(ProposalDocumentBinding.from_dict(item) for item in raw_documents),
                binding_digest=str(payload["binding_digest"]),
            )
        except (TypeError, ValueError) as exc:
            raise SealedProposalIntegrityError("sealed proposal binding is invalid") from exc


class SealedProposalStore:
    """Stores model output once and an exact, body-free reverse lineage index."""

    def __init__(self, root: str | Path, *, tenant_id: str) -> None:
        require_safe_path_segment(tenant_id, "proposal tenant_id")
        shared = Path(root).expanduser().resolve(strict=False)
        artifact_root = shared if tenant_id == "default" else shared / "tenants" / tenant_id
        self.root = artifact_root / "system" / "memory-documents" / "sealed-proposals"
        self.artifact_root = artifact_root
        self.shared_root = shared
        self.tenant_id = tenant_id
        self.erasure_store = MemoryDocumentEraseStore(shared)

    def path(self, owner_user_id: str, task_id: str) -> Path:
        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        return self.root / "sets" / owner / f"{_task_key(task_id)}.json"

    def binding_catalog_path(self, owner_user_id: str) -> Path:
        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        return self.root / "bindings" / owner / "catalog.json"

    def erasure_barrier_path(self, owner_user_id: str, task_id: str) -> Path:
        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        return self.root / "erasure-barriers" / owner / f"{_task_key(task_id)}.json"

    def seal(
        self,
        *,
        task_id: str,
        owner_user_id: str,
        archive_uri: str,
        archive_digest: str,
        manifest_digest: str,
        proposals: tuple[MemoryEditProposal, ...],
    ) -> SealedProposalSet:
        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        _validate_task_id(task_id)
        rows = [item.to_dict() for item in proposals]
        proposal_set_digest = canonical_digest(rows)
        core = {
            "schema_version": _PROPOSAL_SCHEMA,
            "task_id": task_id,
            "tenant_id": self.tenant_id,
            "owner_user_id": owner,
            "archive_uri": archive_uri,
            "archive_digest": archive_digest,
            "manifest_digest": manifest_digest,
            "proposals": rows,
            "proposal_set_digest": proposal_set_digest,
            "created_at": utc_now(),
        }
        payload = {**core, "artifact_digest": canonical_digest(core)}
        path = self.path(owner, task_id)
        with self._binding_lock(owner):
            # Sharing the binding lock makes barrier publication and proposal
            # publication mutually exclusive, so a concurrent erase cannot be
            # followed by a late body-bearing resurrection.
            self.assert_task_replay_allowed(owner, task_id)
            existing_payload = self._read_json(path, maximum=_MAX_PROPOSAL_BYTES, missing_ok=True)
            if existing_payload is not None:
                existing = self._decode_proposal(existing_payload, owner, task_id)
                self._assert_identity(existing, core)
                return existing
            atomic_create_json(path, payload, artifact_root=self.artifact_root)
            return self.load(owner, task_id)

    def load(self, owner_user_id: str, task_id: str) -> SealedProposalSet:
        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        _validate_task_id(task_id)
        payload = self._read_json(self.path(owner, task_id), maximum=_MAX_PROPOSAL_BYTES)
        assert payload is not None
        return self._decode_proposal(payload, owner, task_id)

    def bind_documents(
        self,
        *,
        task_id: str,
        owner_user_id: str,
        proposal_set_digest: str,
        document_bindings: Iterable[tuple[str, str]],
    ) -> SealedProposalBindingSet:
        """Atomically publish exact task/document lineage without proposal text."""

        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        _validate_task_id(task_id)
        _require_digest(proposal_set_digest, "proposal set digest")
        grouped: dict[str, set[str]] = {}
        for raw_document_id, raw_change_digest in document_bindings:
            document_id = validate_document_id(raw_document_id)
            change_digest = _require_digest(raw_change_digest, "proposal document change digest")
            grouped.setdefault(document_id, set()).add(change_digest)
        documents = tuple(
            ProposalDocumentBinding(
                document_id=document_id,
                change_digest=canonical_digest(sorted(grouped[document_id])),
            )
            for document_id in sorted(grouped)
        )
        binding = SealedProposalBindingSet.build(
            task_id=task_id,
            tenant_id=self.tenant_id,
            owner_user_id=owner,
            proposal_set_digest=proposal_set_digest,
            documents=documents,
        )
        locks = []
        try:
            for item in binding.documents:
                locked = self.erasure_store.document_lock(self.tenant_id, owner, item.document_id)
                locked.__enter__()
                locks.append(locked)
            erased = tuple(
                record
                for item in binding.documents
                if (record := self.erasure_store.load(self.tenant_id, owner, item.document_id)) is not None
            )
            if erased:
                with self._binding_lock(owner):
                    self._write_erasure_barrier(binding)
                    self._delete_sealed_set_if_present(binding)
                    catalog = self._load_catalog(owner)
                    self._write_catalog(owner, tuple(item for item in catalog if item.task_id != task_id))
                raise DocumentErasedError(
                    f"sealed Session task is blocked by document erasure epoch {erased[0].erasure_epoch}"
                )
            self.assert_task_replay_allowed(owner, task_id)
            sealed = self.load(owner, task_id)
            if sealed.proposal_set_digest != proposal_set_digest:
                raise SealedProposalIntegrityError("proposal binding is detached from its sealed proposal set")
            with self._binding_lock(owner):
                catalog = self._load_catalog(owner)
                existing = next((item for item in catalog if item.task_id == task_id), None)
                if existing is not None:
                    if existing != binding:
                        raise SealedProposalIntegrityError("Session task is bound to different documents")
                    return existing
                self._write_catalog(owner, (*catalog, binding))
                durable = next(
                    (item for item in self._load_catalog(owner) if item.task_id == task_id),
                    None,
                )
                if durable != binding:
                    raise SealedProposalIntegrityError("proposal document binding was not durably published")
                return binding
        finally:
            for locked in reversed(locks):
                locked.__exit__(None, None, None)

    def bindings_for_document(
        self,
        owner_user_id: str,
        document_id: str,
    ) -> tuple[SealedProposalBindingSet, ...]:
        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        identifier = validate_document_id(document_id)
        with self._binding_lock(owner):
            return tuple(
                binding
                for binding in self._load_catalog(owner)
                if any(item.document_id == identifier for item in binding.documents)
            )

    def delete_for_document(
        self,
        owner_user_id: str,
        document_id: str,
        *,
        erasure_epoch: str,
    ) -> int:
        """Erase exactly bound proposal sets and all bindings for their tasks."""

        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        identifier = validate_document_id(document_id)
        if not erasure_epoch.startswith("erase_"):
            raise ValueError("proposal cleanup requires an exact erasure epoch")
        _require_digest(erasure_epoch.removeprefix("erase_"), "proposal erasure epoch")
        with self._binding_lock(owner):
            catalog = self._load_catalog(owner)
            selected = tuple(
                binding
                for binding in catalog
                if any(item.document_id == identifier for item in binding.documents)
            )
            for binding in selected:
                sealed_payload = self._read_json(
                    self.path(owner, binding.task_id),
                    maximum=_MAX_PROPOSAL_BYTES,
                    missing_ok=True,
                )
                if sealed_payload is not None:
                    sealed = self._decode_proposal(sealed_payload, owner, binding.task_id)
                    if sealed.proposal_set_digest != binding.proposal_set_digest:
                        raise SealedProposalIntegrityError(
                            "proposal cleanup binding is detached from its sealed proposal set"
                        )
                self._write_erasure_barrier(binding)
                self._delete_sealed_set_if_present(binding)
            if selected:
                selected_tasks = {item.task_id for item in selected}
                self._write_catalog(
                    owner,
                    tuple(binding for binding in catalog if binding.task_id not in selected_tasks),
                )
            return len(selected)

    def assert_task_replay_allowed(self, owner_user_id: str, task_id: str) -> None:
        owner = require_safe_path_segment(owner_user_id, "proposal owner_user_id")
        _validate_task_id(task_id)
        payload = self._read_json(
            self.erasure_barrier_path(owner, task_id),
            maximum=_MAX_BARRIER_BYTES,
            missing_ok=True,
        )
        if payload is None:
            return
        _, binding_digest = self._decode_erasure_barrier(payload, owner, task_id)
        raise DocumentErasedError(
            f"sealed Session task is blocked by durable proposal erasure barrier {binding_digest}"
        )

    @staticmethod
    def _assert_identity(existing: SealedProposalSet, requested: dict[str, Any]) -> None:
        fields = (
            "task_id",
            "tenant_id",
            "owner_user_id",
            "archive_uri",
            "archive_digest",
            "manifest_digest",
            "proposal_set_digest",
        )
        for field in fields:
            if str(getattr(existing, field)) != str(requested[field]):
                raise SealedProposalIntegrityError("task is already bound to another proposal set")

    def _decode_proposal(
        self,
        payload: dict[str, Any],
        owner_user_id: str,
        task_id: str,
    ) -> SealedProposalSet:
        if payload.get("schema_version") != _PROPOSAL_SCHEMA:
            raise SealedProposalIntegrityError("sealed proposal schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "artifact_digest"}
        if payload.get("artifact_digest") != canonical_digest(core):
            raise SealedProposalIntegrityError("sealed proposal artifact digest is corrupt")
        identity = (payload.get("tenant_id"), payload.get("owner_user_id"), payload.get("task_id"))
        if identity != (self.tenant_id, owner_user_id, task_id):
            raise SealedProposalIntegrityError("sealed proposal crosses its trusted identity boundary")
        raw_rows = payload.get("proposals")
        if not isinstance(raw_rows, list):
            raise SealedProposalIntegrityError("sealed proposals must be an array")
        try:
            proposals = tuple(MemoryEditProposal.from_dict(item) for item in raw_rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise SealedProposalIntegrityError("sealed proposal set is invalid") from exc
        if payload.get("proposal_set_digest") != canonical_digest([item.to_dict() for item in proposals]):
            raise SealedProposalIntegrityError("sealed proposal set digest is corrupt")
        return SealedProposalSet(
            task_id=task_id,
            tenant_id=self.tenant_id,
            owner_user_id=owner_user_id,
            archive_uri=str(payload["archive_uri"]),
            archive_digest=str(payload["archive_digest"]),
            manifest_digest=str(payload["manifest_digest"]),
            proposals=proposals,
            proposal_set_digest=str(payload["proposal_set_digest"]),
        )

    def _load_catalog(self, owner_user_id: str) -> tuple[SealedProposalBindingSet, ...]:
        payload = self._read_json(
            self.binding_catalog_path(owner_user_id),
            maximum=_MAX_BINDING_CATALOG_BYTES,
            missing_ok=True,
        )
        if payload is None:
            return ()
        if payload.get("schema_version") != _BINDING_CATALOG_SCHEMA:
            raise SealedProposalIntegrityError("proposal binding catalog schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "artifact_digest"}
        if payload.get("artifact_digest") != canonical_digest(core):
            raise SealedProposalIntegrityError("proposal binding catalog digest is corrupt")
        if (payload.get("tenant_id"), payload.get("owner_user_id")) != (
            self.tenant_id,
            owner_user_id,
        ):
            raise SealedProposalIntegrityError("proposal binding catalog crosses its owner boundary")
        rows = payload.get("bindings")
        if not isinstance(rows, list):
            raise SealedProposalIntegrityError("proposal binding catalog entries must be an array")
        bindings = tuple(SealedProposalBindingSet.from_dict(item) for item in rows)
        if tuple(sorted(bindings, key=lambda item: item.task_id)) != bindings:
            raise SealedProposalIntegrityError("proposal binding catalog is not deterministically ordered")
        if len({item.task_id for item in bindings}) != len(bindings):
            raise SealedProposalIntegrityError("proposal binding catalog repeats a task identity")
        for binding in bindings:
            if (binding.tenant_id, binding.owner_user_id) != (self.tenant_id, owner_user_id):
                raise SealedProposalIntegrityError("proposal binding crosses its owner boundary")
        return bindings

    def _write_catalog(
        self,
        owner_user_id: str,
        bindings: Iterable[SealedProposalBindingSet],
    ) -> None:
        ordered = tuple(sorted(bindings, key=lambda item: item.task_id))
        if len({item.task_id for item in ordered}) != len(ordered):
            raise SealedProposalIntegrityError("proposal binding catalog repeats a task identity")
        core = {
            "schema_version": _BINDING_CATALOG_SCHEMA,
            "tenant_id": self.tenant_id,
            "owner_user_id": owner_user_id,
            "bindings": [item.to_dict() for item in ordered],
        }
        atomic_write_json(
            self.binding_catalog_path(owner_user_id),
            {**core, "artifact_digest": canonical_digest(core)},
            artifact_root=self.artifact_root,
        )

    def _write_erasure_barrier(self, binding: SealedProposalBindingSet) -> None:
        core = {
            "schema_version": _ERASURE_BARRIER_SCHEMA,
            "tenant_id": binding.tenant_id,
            "owner_user_id": binding.owner_user_id,
            "task_id": binding.task_id,
            "proposal_set_digest": binding.proposal_set_digest,
            "binding_digest": binding.binding_digest,
        }
        path = self.erasure_barrier_path(binding.owner_user_id, binding.task_id)
        existing = self._read_json(path, maximum=_MAX_BARRIER_BYTES, missing_ok=True)
        if existing is None:
            atomic_create_json(
                path,
                {**core, "artifact_digest": canonical_digest(core)},
                artifact_root=self.artifact_root,
            )
        durable = self._read_json(path, maximum=_MAX_BARRIER_BYTES)
        assert durable is not None
        if self._decode_erasure_barrier(
            durable,
            binding.owner_user_id,
            binding.task_id,
        ) != (binding.proposal_set_digest, binding.binding_digest):
            raise SealedProposalIntegrityError("proposal erasure barrier conflicts with exact task lineage")

    def _decode_erasure_barrier(
        self,
        payload: dict[str, Any],
        owner_user_id: str,
        task_id: str,
    ) -> tuple[str, str]:
        if payload.get("schema_version") != _ERASURE_BARRIER_SCHEMA:
            raise SealedProposalIntegrityError("proposal erasure barrier schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "artifact_digest"}
        if payload.get("artifact_digest") != canonical_digest(core):
            raise SealedProposalIntegrityError("proposal erasure barrier digest is corrupt")
        if (payload.get("tenant_id"), payload.get("owner_user_id"), payload.get("task_id")) != (
            self.tenant_id,
            owner_user_id,
            task_id,
        ):
            raise SealedProposalIntegrityError("proposal erasure barrier crosses its trusted identity")
        try:
            proposal_set_digest = _require_digest(
                payload.get("proposal_set_digest"),
                "proposal erasure set digest",
            )
            binding_digest = _require_digest(
                payload.get("binding_digest"),
                "proposal erasure binding digest",
            )
        except ValueError as exc:
            raise SealedProposalIntegrityError("proposal erasure barrier is invalid") from exc
        expected_fields = {
            "schema_version",
            "tenant_id",
            "owner_user_id",
            "task_id",
            "proposal_set_digest",
            "binding_digest",
            "artifact_digest",
        }
        if set(payload) != expected_fields:
            raise SealedProposalIntegrityError("proposal erasure barrier has unsupported fields")
        return proposal_set_digest, binding_digest

    def _delete_sealed_set_if_present(self, binding: SealedProposalBindingSet) -> None:
        path = self.path(binding.owner_user_id, binding.task_id)
        payload = self._read_json(path, maximum=_MAX_PROPOSAL_BYTES, missing_ok=True)
        if payload is None:
            return
        sealed = self._decode_proposal(payload, binding.owner_user_id, binding.task_id)
        if sealed.proposal_set_digest != binding.proposal_set_digest:
            raise SealedProposalIntegrityError("proposal cleanup would delete a detached sealed set")
        self._safe_unlink(path)

    @contextmanager
    def _binding_lock(self, owner_user_id: str):  # noqa: ANN202 - private context manager.
        path = self.root / "bindings" / owner_user_id / "catalog.lock"
        descriptor = open_private_lock(path, root=self.artifact_root)
        if os.fstat(descriptor).st_nlink != 1:
            os.close(descriptor)
            raise SealedProposalIntegrityError("proposal binding lock must be one private file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read_json(
        self,
        path: Path,
        *,
        maximum: int,
        missing_ok: bool = False,
    ) -> dict[str, Any] | None:
        try:
            parent = _open_control_parent(path, self.artifact_root)
        except Exception as exc:  # noqa: BLE001 - normalize path-integrity failures.
            raise SealedProposalIntegrityError("sealed proposal path is unsafe") from exc
        try:
            try:
                descriptor = os.open(
                    path.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent,
                )
            except FileNotFoundError:
                if missing_ok:
                    return None
                raise
            except OSError as exc:
                raise SealedProposalIntegrityError("sealed proposal artifact is unreadable") from exc
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise SealedProposalIntegrityError(
                        "sealed proposal artifact must be one private regular file"
                    )
                if metadata.st_size > maximum:
                    raise SealedProposalIntegrityError("sealed proposal artifact exceeds its size bound")
                raw = _read_bounded(descriptor, maximum)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent)
        try:
            payload = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SealedProposalIntegrityError("sealed proposal artifact is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise SealedProposalIntegrityError("sealed proposal artifact must be an object")
        return payload

    def _safe_unlink(self, path: Path) -> None:
        try:
            parent = _open_control_parent(path, self.artifact_root)
        except Exception as exc:  # noqa: BLE001 - normalize path-integrity failures.
            raise SealedProposalIntegrityError("sealed proposal deletion path is unsafe") from exc
        try:
            try:
                descriptor = os.open(
                    path.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent,
                )
            except FileNotFoundError:
                return
            except OSError as exc:
                raise SealedProposalIntegrityError("sealed proposal deletion target is unsafe") from exc
            try:
                opened = os.fstat(descriptor)
                named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or named.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
                ):
                    raise SealedProposalIntegrityError(
                        "sealed proposal deletion target must be one private regular file"
                    )
                os.unlink(path.name, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent)


def _task_key(task_id: str) -> str:
    _validate_task_id(task_id)
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()


def _validate_task_id(task_id: object) -> str:
    if not isinstance(task_id, str) or not task_id or len(task_id.encode("utf-8")) > 512:
        raise ValueError("proposal task_id is empty or too large")
    if any(ord(character) < 32 for character in task_id):
        raise ValueError("proposal task_id contains control characters")
    return task_id


def _require_digest(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _read_bounded(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > maximum:
        raise SealedProposalIntegrityError("sealed proposal artifact exceeds its size bound")
    return raw


__all__ = [
    "ProposalDocumentBinding",
    "SealedProposalBindingSet",
    "SealedProposalIntegrityError",
    "SealedProposalSet",
    "SealedProposalStore",
]
