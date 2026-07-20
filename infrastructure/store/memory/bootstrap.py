"""受控 Markdown 树的首用户初始化流程，崩溃后可继续。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infrastructure.store.filesystem.durable_io import atomic_write_json
from infrastructure.store.memory.control_store import MemoryDocumentControlStore
from infrastructure.store.memory.layout import RuntimeResetRequired, tenant_control_root
from memory.core.model import ABSENT, PresentPath, ScanGeneration
from memory.core.structure.frontmatter import (
    new_document_id,
    parse_front_matter,
    render_new_document,
    validate_document_id,
)
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import DocumentConflictError, MemoryDocumentStore

_TEMPLATES = {
    "MEMORY.md": "# Memory\n\n## Profile\n\n- [Profile](profile.md)\n\n## Preferences\n\n- [Preferences](preferences.md)\n\n## Knowledge\n\n- [Knowledge](knowledge/MEMORY.md)\n",
    "profile.md": "# Profile\n",
    "preferences.md": "# Preferences\n",
    "knowledge/MEMORY.md": "# Knowledge\n\n## Entities\n\n## Topics\n\n## Episodes\n\n## Open loops\n\n- [Open loops](open-loops.md)\n",
    "knowledge/open-loops.md": "# Open Loops\n",
}


@dataclass(frozen=True)
class MemoryDocumentBootstrapper:
    root: Path
    store: MemoryDocumentStore
    control_store: MemoryDocumentControlStore
    max_front_matter_bytes: int = 32 * 1024
    max_front_matter_depth: int = 12

    def status(self, tenant_id: str, owner_user_id: str) -> str:
        """返回一个通过校验的耐久初始化状态，不修改任何内容。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        marker = self._marker(tenant_control_root(self.root, tenant), owner)
        payload = self._read_marker(marker)
        if not payload:
            return ""
        self._validate_payload(payload, tenant=tenant, owner=owner)
        return str(payload["status"])

    def ensure_user(self, tenant_id: str, owner_user_id: str) -> ScanGeneration:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        self._probe(tenant, owner)
        control_root = tenant_control_root(self.root, tenant)
        marker = self._marker(control_root, owner)
        payload = self._read_marker(marker)
        if payload and payload.get("status") == "COMPLETED":
            scan = self.store.full_scan(tenant, owner)
            self._require_publishable_root_scan(scan)
            self._verify_completed_root(scan)
            return scan
        if not payload:
            scan = self.store.full_scan(tenant, owner)
            if not scan.complete or scan.registrations or scan.unsafe_paths:
                raise RuntimeResetRequired(
                    "user memory tree already contains data but has no bootstrap control record"
                )
            payload = {
                "schema": "memory_document_bootstrap_v1",
                "status": "PREPARED",
                "tenant_id": tenant,
                "owner_user_id": owner,
                "documents": [
                    {
                        "relative_path": relative_path,
                        "document_id": new_document_id(),
                        "source": "TEMPLATE",
                        "adopted_raw_sha256": "",
                    }
                    for relative_path in _TEMPLATES
                ],
            }
            atomic_write_json(marker, payload, artifact_root=control_root)
        return self._complete(payload, marker=marker, control_root=control_root, tenant=tenant, owner=owner)

    def ensure_adopted_user(
        self,
        tenant_id: str,
        owner_user_id: str,
        adopted_relative_path: str,
        *,
        document_id: str,
        adopted_raw_sha256: str,
    ) -> ScanGeneration:
        """围绕一个已接管文件，以崩溃安全方式初始化模板。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        relative = MemoryDocumentPathPolicy.normalize_relative_path(adopted_relative_path)
        identifier = validate_document_id(document_id)
        if len(adopted_raw_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in adopted_raw_sha256
        ):
            raise ValueError("adopted_raw_sha256 must be a lowercase SHA-256 digest")
        self._probe(tenant, owner)
        control_root = tenant_control_root(self.root, tenant)
        marker = self._marker(control_root, owner)
        payload = self._read_marker(marker)
        if payload and payload.get("status") == "COMPLETED":
            scan = self.store.full_scan(tenant, owner)
            self._require_publishable_root_scan(scan)
            self._verify_completed_root(scan)
            return scan
        scan = self.store.full_scan(tenant, owner)
        if not scan.complete or scan.errors:
            raise RuntimeResetRequired("adopt-first bootstrap requires a complete document scan")
        adopted = [
            item
            for item in scan.managed
            if item.relative_path == relative
            and item.document_id == identifier
            and item.raw_sha256 == adopted_raw_sha256
        ]
        if len(adopted) != 1:
            raise DocumentConflictError("adopt-first bootstrap is detached from the exact managed file")
        if not payload:
            documents: list[dict[str, str]] = []
            for template_path in _TEMPLATES:
                if template_path == relative:
                    documents.append(
                        {
                            "relative_path": template_path,
                            "document_id": identifier,
                            "source": "ADOPTED",
                            "adopted_raw_sha256": adopted_raw_sha256,
                        }
                    )
                    continue
                if self.store.read_state(tenant, owner, template_path) != ABSENT:
                    raise DocumentConflictError(
                        "adopt-first bootstrap found a non-adopted template path collision"
                    )
                documents.append(
                    {
                        "relative_path": template_path,
                        "document_id": new_document_id(),
                        "source": "TEMPLATE",
                        "adopted_raw_sha256": "",
                    }
                )
            payload = {
                "schema": "memory_document_bootstrap_v1",
                "status": "PREPARED",
                "tenant_id": tenant,
                "owner_user_id": owner,
                "documents": documents,
            }
            atomic_write_json(marker, payload, artifact_root=control_root)
        return self._complete(payload, marker=marker, control_root=control_root, tenant=tenant, owner=owner)

    def _complete(
        self,
        payload: dict[str, Any],
        *,
        marker: Path,
        control_root: Path,
        tenant: str,
        owner: str,
    ) -> ScanGeneration:
        self._validate_payload(payload, tenant=tenant, owner=owner)
        for item in payload["documents"]:
            relative = str(item["relative_path"])
            document_id = str(item["document_id"])
            source = str(item["source"])
            operation_id = f"bootstrap:{document_id}"
            if source == "ADOPTED":
                state = self.store.read_state(tenant, owner, relative)
                if (
                    not isinstance(state, PresentPath)
                    or state.raw_sha256 != str(item["adopted_raw_sha256"])
                ):
                    raise DocumentConflictError("adopt-first bootstrap managed path changed")
                raw = self.store.read_raw(tenant, owner, relative_path=relative)
                parsed = parse_front_matter(
                    raw,
                    max_header_bytes=self.max_front_matter_bytes,
                    max_depth=self.max_front_matter_depth,
                )
                if parsed.document_id != document_id:
                    raise DocumentConflictError("adopt-first bootstrap document identity changed")
                continue
            expected = render_new_document(document_id, _TEMPLATES[relative])
            cleanup_temps = getattr(self.store, "cleanup_operation_temps", None)
            if callable(cleanup_temps):
                cleanup_temps(
                    tenant,
                    owner,
                    {relative: hashlib.sha256(expected).hexdigest()},
                    operation_id,
                )
            state = self.store.read_state(tenant, owner, relative)
            if state == ABSENT:
                self.store.create(
                    tenant,
                    owner,
                    relative,
                    expected,
                    expected=ABSENT,
                    operation_id=operation_id,
                )
                continue
            if not isinstance(state, PresentPath):
                raise DocumentConflictError("bootstrap path is unsafe")
            raw = self.store.read_raw(tenant, owner, relative_path=relative)
            parsed = parse_front_matter(
                raw,
                max_header_bytes=self.max_front_matter_bytes,
                max_depth=self.max_front_matter_depth,
            )
            if parsed.document_id != document_id or raw != expected:
                raise DocumentConflictError("bootstrap encountered a third-state user file")
        scan = self.store.full_scan(tenant, owner)
        self._require_publishable_root_scan(scan)
        self._publish_prepared_root(scan)
        completed = {**payload, "status": "COMPLETED"}
        atomic_write_json(marker, completed, artifact_root=control_root)
        return scan

    @staticmethod
    def _require_publishable_root_scan(scan: ScanGeneration) -> None:
        if (
            not scan.complete
            or scan.errors
            or len(scan.root_identity) != 32
            or any(character not in "0123456789abcdef" for character in scan.root_identity)
        ):
            raise RuntimeResetRequired(
                "bootstrap completion requires a complete scan with a safe document root identity"
            )

    def _publish_prepared_root(self, scan: ScanGeneration) -> None:
        """在完成状态耐久化之前，先绑定安全的 PREPARED 目录树。"""

        self.control_store.ensure_root_identity(
            scan.tenant_id,
            scan.owner_user_id,
            scan.root_identity,
            allow_prepared_bootstrap=True,
        )

    def _verify_completed_root(self, scan: ScanGeneration) -> None:
        """只有不可变根绑定建立后，COMPLETED 标记才有效。"""

        durable = self.control_store.load_root_identity(
            scan.tenant_id,
            scan.owner_user_id,
        )
        if durable is None:
            raise RuntimeResetRequired(
                "completed bootstrap is missing its durable document root identity"
            )
        if durable.root_identity != scan.root_identity:
            raise RuntimeResetRequired(
                "completed bootstrap document root identity changed and requires explicit reset"
            )

    def _probe(self, tenant: str, owner: str) -> None:
        probe = getattr(self.store, "probe_write_capabilities", None)
        if callable(probe):
            # 即使控制根目录在本地，用户目录树也可能是嵌套挂载点。因此任何
            # 系统写入之前都必须先验证源文件系统。
            probe(tenant, owner)

    @staticmethod
    def _marker(control_root: Path, owner: str) -> Path:
        return control_root / "system" / "memory-documents" / owner / "bootstrap.json"

    @staticmethod
    def _read_marker(path: Path) -> dict[str, Any]:
        if not path.exists():
            if path.is_symlink():
                raise RuntimeResetRequired("bootstrap marker cannot be a symbolic link")
            return {}
        if path.is_symlink() or not path.is_file():
            raise RuntimeResetRequired("bootstrap marker is unsafe")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeResetRequired("bootstrap marker is invalid") from exc
        if not isinstance(payload, dict):
            raise RuntimeResetRequired("bootstrap marker must be an object")
        return payload

    @staticmethod
    def _validate_payload(payload: dict[str, Any], *, tenant: str, owner: str) -> None:
        if (
            payload.get("schema") != "memory_document_bootstrap_v1"
            or payload.get("status") not in {"PREPARED", "COMPLETED"}
            or payload.get("tenant_id") != tenant
            or payload.get("owner_user_id") != owner
            or not isinstance(payload.get("documents"), list)
        ):
            raise RuntimeResetRequired("bootstrap marker binding is invalid")
        paths = {str(item.get("relative_path") or "") for item in payload["documents"] if isinstance(item, dict)}
        if paths != set(_TEMPLATES):
            raise RuntimeResetRequired("bootstrap marker document set is invalid")
        adopted_count = 0
        for item in payload["documents"]:
            if not isinstance(item, dict):
                raise RuntimeResetRequired("bootstrap marker document entry is invalid")
            try:
                validate_document_id(item.get("document_id"))
            except ValueError as exc:
                raise RuntimeResetRequired("bootstrap marker document ID is invalid") from exc
            source = str(item.get("source") or "")
            adopted_digest = str(item.get("adopted_raw_sha256") or "")
            if source == "TEMPLATE" and adopted_digest:
                raise RuntimeResetRequired("template bootstrap entry cannot claim adopted bytes")
            if source == "ADOPTED":
                adopted_count += 1
                if not (
                    len(adopted_digest) == 64
                    and all(character in "0123456789abcdef" for character in adopted_digest)
                ):
                    raise RuntimeResetRequired("adopted bootstrap digest is invalid")
            elif source != "TEMPLATE":
                raise RuntimeResetRequired("bootstrap marker document source is invalid")
        if adopted_count > 1:
            raise RuntimeResetRequired("bootstrap marker contains multiple adopted template paths")


__all__ = ["MemoryDocumentBootstrapper"]
