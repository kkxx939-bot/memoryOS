"""文档根身份与外部 Markdown 接管凭证模型。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from foundation.integrity import canonical_json
from infrastructure.store.memory.control_common import (
    _ADOPTION_RECEIPT_SCHEMA,
    _ROOT_IDENTITY_SCHEMA,
    DocumentControlIntegrityError,
)
from infrastructure.store.memory.control_common import (
    is_hex as _is_hex,
)
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy


@dataclass(frozen=True)
class DocumentRootIdentity:
    """一个所有者与受控源根目录之间不含正文的耐久绑定。"""

    tenant_id: str
    owner_user_id: str
    root_identity: str

    def __post_init__(self) -> None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        if not _is_hex(self.root_identity, 32):
            raise ValueError("document root identity must be a 128-bit lowercase hex digest")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "owner_user_id", owner)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _ROOT_IDENTITY_SCHEMA, **self.__dict__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentRootIdentity:
        if (
            payload.get("schema") != _ROOT_IDENTITY_SCHEMA
            or set(payload) != {"schema", "tenant_id", "owner_user_id", "root_identity"}
        ):
            raise DocumentControlIntegrityError("document root identity schema is unsupported")
        try:
            return cls(
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                root_identity=str(payload["root_identity"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document root identity is malformed") from exc
@dataclass(frozen=True)
class DocumentRootIdentityGuard:
    """只在持有所有者根身份锁期间有效的能力对象。"""

    _store: Any
    tenant_id: str
    owner_user_id: str

    def ensure(
        self,
        root_identity: str,
        *,
        allow_prepared_bootstrap: bool = False,
    ) -> DocumentRootIdentity:
        requested = DocumentRootIdentity(
            tenant_id=self.tenant_id,
            owner_user_id=self.owner_user_id,
            root_identity=root_identity,
        )
        return self._store._ensure_root_identity_locked(
            requested,
            allow_prepared_bootstrap=allow_prepared_bootstrap,
        )


@dataclass(frozen=True)
class DocumentAdoptionReceipt:
    """用于重试一次未受管文件接管的无正文权限凭证。

    文档身份由完整请求身份推导。因此，并发创建者会发布完全相同的不可变
    字节；实时头信息重写后的重试也能恢复同一身份，而无需保留源正文。
    """

    receipt_id: str
    request_digest: str
    tenant_id: str
    owner_user_id: str
    relative_path: str
    expected_raw_sha256: str
    document_id: str
    actor_binding: str
    evidence_reference: str
    evidence_digest: str
    idempotency_key: str
    edit_summary: str

    def __post_init__(self) -> None:
        tenant = MemoryDocumentPathPolicy.trusted_segment(self.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(self.owner_user_id, "owner_user_id")
        relative = MemoryDocumentPathPolicy.normalize_relative_path(self.relative_path)
        expected = str(self.expected_raw_sha256)
        if not _is_hex(expected, 64):
            raise ValueError("adoption expected raw digest must be SHA-256")
        request_digest = adoption_request_digest(tenant, owner, relative, expected)
        receipt_id = f"mdadopt_{request_digest}"
        document_id = adoption_document_id(request_digest)
        if (
            self.request_digest != request_digest
            or self.receipt_id != receipt_id
            or self.document_id != document_id
        ):
            raise ValueError("document adoption receipt is detached from its request identity")
        if (
            not self.actor_binding
            or len(self.actor_binding) > 512
            or any(ord(character) < 32 and character not in "\t" for character in self.actor_binding)
        ):
            raise ValueError("document adoption actor binding is invalid")
        if self.evidence_reference != f"adoption-receipt:{receipt_id}":
            raise ValueError("document adoption evidence reference is detached from its receipt")
        if self.evidence_digest != expected:
            raise ValueError("document adoption evidence digest is detached from its source digest")
        if self.idempotency_key != f"adoption:{receipt_id}":
            raise ValueError("document adoption idempotency key is detached from its receipt")
        if self.edit_summary != "adopt unmanaged Markdown document":
            raise ValueError("document adoption edit summary is invalid")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "owner_user_id", owner)
        object.__setattr__(self, "relative_path", relative)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _ADOPTION_RECEIPT_SCHEMA, **self.__dict__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentAdoptionReceipt:
        if payload.get("schema") != _ADOPTION_RECEIPT_SCHEMA:
            raise DocumentControlIntegrityError("document adoption receipt schema is unsupported")
        try:
            return cls(
                receipt_id=str(payload["receipt_id"]),
                request_digest=str(payload["request_digest"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                relative_path=str(payload["relative_path"]),
                expected_raw_sha256=str(payload["expected_raw_sha256"]),
                document_id=str(payload["document_id"]),
                actor_binding=str(payload["actor_binding"]),
                evidence_reference=str(payload["evidence_reference"]),
                evidence_digest=str(payload["evidence_digest"]),
                idempotency_key=str(payload["idempotency_key"]),
                edit_summary=str(payload["edit_summary"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document adoption receipt is malformed") from exc


def adoption_request_digest(
    tenant_id: str,
    owner_user_id: str,
    relative_path: str,
    expected_raw_sha256: str,
) -> str:
    tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
    owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
    relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
    if not _is_hex(expected_raw_sha256, 64):
        raise ValueError("adoption expected raw digest must be SHA-256")
    encoded = canonical_json(
        ["memory_document_adoption_v1", tenant, owner, relative, expected_raw_sha256]
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def adoption_document_id(request_digest: str) -> str:
    if not _is_hex(request_digest, 64):
        raise ValueError("adoption request digest must be SHA-256")
    suffix = hashlib.sha256(
        canonical_json(["memory_document_adoption_document_v1", request_digest]).encode()
    ).hexdigest()
    return validate_document_id(f"memdoc_{suffix}")
