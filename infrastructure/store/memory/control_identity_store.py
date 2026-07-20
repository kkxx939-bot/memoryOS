"""根身份和外部文档接管凭证的持久化操作。"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager

from infrastructure.store.filesystem.durable_io import (
    ImmutableArtifactConflictError,
    atomic_create_json,
)
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.filesystem.file_lock import open_private_lock
from infrastructure.store.memory.control_commit_store import ControlCommitStoreMixin
from infrastructure.store.memory.control_common import (
    _ADOPTION_IDENTITY_SCHEMA,
    _MAX_ADOPTION_RECEIPTS,
    DocumentControlIntegrityError,
)
from infrastructure.store.memory.control_common import (
    is_hex as _is_hex,
)
from infrastructure.store.memory.control_common import (
    validate_prefixed_digest as _validate_prefixed_digest,
)
from infrastructure.store.memory.control_identity import (
    DocumentAdoptionReceipt,
    DocumentRootIdentity,
    DocumentRootIdentityGuard,
    adoption_document_id,
    adoption_request_digest,
)
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy


class ControlIdentityStoreMixin(ControlCommitStoreMixin):
    """身份控制操作；文件和提交查询能力由组合存储的兄弟 Mixin 提供。"""

    @contextmanager
    def root_identity_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> Iterator[DocumentRootIdentityGuard]:
        """串行化一个所有者首次发布根目录权限的过程。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        artifact_root = self._artifact_root(tenant)
        descriptor = open_private_lock(
            self._owner_root(tenant, owner) / "locks" / "root-identity.lock",
            root=artifact_root,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield DocumentRootIdentityGuard(self, tenant, owner)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def prepare_adoption_receipt(
        self,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        expected_raw_sha256: str,
        *,
        actor_binding: str,
    ) -> DocumentAdoptionReceipt:
        """只在所有者根目录权限存在后创建或重放接管凭证。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        with self.root_identity_lock(tenant, owner):
            if self.load_root_identity(tenant, owner) is None:
                raise DocumentControlIntegrityError(
                    "document adoption receipt requires an existing source root identity"
                )
            return self._prepare_adoption_receipt_locked(
                tenant,
                owner,
                relative_path,
                expected_raw_sha256,
                actor_binding=actor_binding,
            )

    def _prepare_adoption_receipt_locked(
        self,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        expected_raw_sha256: str,
        *,
        actor_binding: str,
    ) -> DocumentAdoptionReceipt:
        """持有所有者权限锁时发布接管凭证和身份索引。"""

        request_digest = adoption_request_digest(
            tenant_id,
            owner_user_id,
            relative_path,
            expected_raw_sha256,
        )
        receipt = DocumentAdoptionReceipt(
            receipt_id=f"mdadopt_{request_digest}",
            request_digest=request_digest,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            relative_path=relative_path,
            expected_raw_sha256=expected_raw_sha256,
            document_id=adoption_document_id(request_digest),
            actor_binding=actor_binding,
            evidence_reference=f"adoption-receipt:mdadopt_{request_digest}",
            evidence_digest=expected_raw_sha256,
            idempotency_key=f"adoption:mdadopt_{request_digest}",
            edit_summary="adopt unmanaged Markdown document",
        )
        try:
            atomic_create_json(
                self._adoption_receipt_path(receipt.tenant_id, receipt.owner_user_id, receipt.receipt_id),
                receipt.to_dict(),
                artifact_root=self._artifact_root(receipt.tenant_id),
            )
        except ImmutableArtifactConflictError:
            pass
        durable = self.load_adoption_receipt(
            receipt.tenant_id,
            receipt.owner_user_id,
            receipt.receipt_id,
        )
        if durable is None:
            raise DocumentControlIntegrityError("document adoption receipt conflicts with its request identity")
        identity_payload = {
            "schema": _ADOPTION_IDENTITY_SCHEMA,
            "tenant_id": durable.tenant_id,
            "owner_user_id": durable.owner_user_id,
            "document_id": durable.document_id,
            "receipt_id": durable.receipt_id,
            "request_digest": durable.request_digest,
        }
        try:
            atomic_create_json(
                self._adoption_identity_path(durable.tenant_id, durable.owner_user_id, durable.document_id),
                identity_payload,
                artifact_root=self._artifact_root(durable.tenant_id),
            )
        except ImmutableArtifactConflictError:
            pass
        indexed = self.load_adoption_receipt_for_document(
            durable.tenant_id,
            durable.owner_user_id,
            durable.document_id,
        )
        if indexed != durable:
            raise DocumentControlIntegrityError("document adoption identity index conflicts with its receipt")
        return durable

    def ensure_root_identity(
        self,
        tenant_id: str,
        owner_user_id: str,
        root_identity: str,
        *,
        allow_prepared_bootstrap: bool = False,
    ) -> DocumentRootIdentity:
        """创建唯一的不可变根绑定，绝不认可替换后的 inode。"""

        requested = DocumentRootIdentity(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            root_identity=root_identity,
        )
        with self.root_identity_lock(
            requested.tenant_id,
            requested.owner_user_id,
        ) as guard:
            return guard.ensure(
                requested.root_identity,
                allow_prepared_bootstrap=allow_prepared_bootstrap,
            )

    def _ensure_root_identity_locked(
        self,
        requested: DocumentRootIdentity,
        *,
        allow_prepared_bootstrap: bool,
    ) -> DocumentRootIdentity:
        current = self.load_root_identity(requested.tenant_id, requested.owner_user_id)
        if current is not None:
            if current != requested:
                raise DocumentControlIntegrityError(
                    "document source root identity changed and requires explicit reset"
                )
            return current
        bootstrap = self._read_json(
            self._bootstrap_path(requested.tenant_id, requested.owner_user_id),
            requested.tenant_id,
        )
        if bootstrap is not None:
            valid_prepared = bool(
                bootstrap.get("schema") == "memory_document_bootstrap_v1"
                and bootstrap.get("status") == "PREPARED"
                and bootstrap.get("tenant_id") == requested.tenant_id
                and bootstrap.get("owner_user_id") == requested.owner_user_id
            )
            if not allow_prepared_bootstrap or not valid_prepared:
                raise DocumentControlIntegrityError(
                    "existing bootstrap authority is missing its source root identity"
                )
        elif allow_prepared_bootstrap:
            raise DocumentControlIntegrityError(
                "root identity bootstrap authority requires an exact PREPARED marker"
            )
        if self.controls(requested.tenant_id, requested.owner_user_id):
            raise DocumentControlIntegrityError(
                "existing document controls are missing their durable source root identity"
            )
        if self.incomplete_intents(requested.tenant_id, requested.owner_user_id):
            raise DocumentControlIntegrityError(
                "existing document intents are missing their durable source root identity"
            )
        if self.adoption_receipts(requested.tenant_id, requested.owner_user_id):
            raise DocumentControlIntegrityError(
                "existing adoption receipts are missing their durable source root identity"
            )
        try:
            atomic_create_json(
                self._root_identity_path(requested.tenant_id, requested.owner_user_id),
                requested.to_dict(),
                artifact_root=self._artifact_root(requested.tenant_id),
            )
        except ImmutableArtifactConflictError:
            pass
        durable = self.load_root_identity(requested.tenant_id, requested.owner_user_id)
        if durable != requested:
            raise DocumentControlIntegrityError("document source root identity publication conflicted")
        return requested

    def load_root_identity(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> DocumentRootIdentity | None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        payload = self._read_json(self._root_identity_path(tenant, owner), tenant)
        if payload is None:
            return None
        identity = DocumentRootIdentity.from_dict(payload)
        if identity.tenant_id != tenant or identity.owner_user_id != owner:
            raise DocumentControlIntegrityError(
                "document root identity path binding does not match its payload"
            )
        return identity

    def root_identity_blockers(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[str, ...]:
        """返回会阻止首次根目录发布的耐久权限记录。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        with self.root_identity_lock(tenant, owner):
            if self.load_root_identity(tenant, owner) is not None:
                return ()
            blockers: list[str] = []
            if self.controls(tenant, owner):
                blockers.append("controls")
            if self.incomplete_intents(tenant, owner):
                blockers.append("intents")
            if self.adoption_receipts(tenant, owner):
                blockers.append("adoption_receipts")
            if self._read_json(self._bootstrap_path(tenant, owner), tenant) is not None:
                blockers.append("bootstrap")
            return tuple(blockers)

    def load_adoption_receipt(
        self,
        tenant_id: str,
        owner_user_id: str,
        receipt_id: str,
    ) -> DocumentAdoptionReceipt | None:
        _validate_prefixed_digest(receipt_id, "mdadopt_", "receipt_id")
        payload = self._read_json(
            self._adoption_receipt_path(tenant_id, owner_user_id, receipt_id),
            tenant_id,
        )
        if payload is None:
            return None
        receipt = DocumentAdoptionReceipt.from_dict(payload)
        if (
            receipt.receipt_id != receipt_id
            or receipt.tenant_id != tenant_id
            or receipt.owner_user_id != owner_user_id
        ):
            raise DocumentControlIntegrityError("document adoption receipt path identity does not match")
        return receipt

    def adoption_receipts(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[DocumentAdoptionReceipt, ...]:
        """有界枚举一个所有者的精确接管权限。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        directory = self._owner_root(tenant, owner) / "adoptions"
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = tuple(sorted(os.listdir(descriptor)))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_ADOPTION_RECEIPTS:
            raise DocumentControlIntegrityError("document adoption receipt count exceeds its bound")
        receipts: list[DocumentAdoptionReceipt] = []
        for name in names:
            receipt_id = name.removesuffix(".json")
            if (
                not name.endswith(".json")
                or not receipt_id.startswith("mdadopt_")
                or not _is_hex(receipt_id.removeprefix("mdadopt_"), 64)
            ):
                raise DocumentControlIntegrityError(
                    "document adoption directory contains an unexpected artifact"
                )
            receipt = self.load_adoption_receipt(tenant, owner, receipt_id)
            if receipt is None:  # pragma: no cover - cooperative snapshots retain files.
                raise DocumentControlIntegrityError(
                    "document adoption receipt disappeared during enumeration"
                )
            receipts.append(receipt)
        return tuple(receipts)

    def load_adoption_receipt_for_document(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentAdoptionReceipt | None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        payload = self._read_json(
            self._adoption_identity_path(tenant, owner, identifier),
            tenant,
        )
        if payload is None:
            return None
        try:
            if payload.get("schema") != _ADOPTION_IDENTITY_SCHEMA:
                raise ValueError("unsupported schema")
            receipt_id = str(payload["receipt_id"])
            request_digest = str(payload["request_digest"])
            if (
                payload.get("tenant_id") != tenant
                or payload.get("owner_user_id") != owner
                or payload.get("document_id") != identifier
                or receipt_id != f"mdadopt_{request_digest}"
            ):
                raise ValueError("identity mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document adoption identity index is malformed") from exc
        receipt = self.load_adoption_receipt(tenant, owner, receipt_id)
        if (
            receipt is None
            or receipt.request_digest != request_digest
            or receipt.document_id != identifier
        ):
            raise DocumentControlIntegrityError("document adoption identity index is detached from its receipt")
        return receipt
