"""按租户彻底清理召回轨迹。

召回轨迹没有持久化文档绑定，因此删除目标文档拥有者的全部轨迹，并清理未绑定
租户的旧轨迹，不再尝试推断轨迹与文档内容之间的关系。
"""

from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any

from memory.ports.erase import DerivedEraseRequest
from memory.core.structure.path_policy import MemoryDocumentPathPolicy

_MAX_TRACE_BYTES = 2 * 1024 * 1024
_MAX_TRACES_PER_TENANT = 10_000


class RecallTraceEraseIntegrityError(RuntimeError):
    """召回轨迹目录无法被安全、完整地清理。"""


class RecallTraceEraseBackend:
    """彻底删除记忆时清理属于该所有者的可重建召回轨迹。"""

    name = "derived.recall_traces"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute()

    def erase_document(self, request: DerivedEraseRequest) -> bool:
        tenant = MemoryDocumentPathPolicy.trusted_segment(request.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(request.owner_user_id, "owner_user_id")
        descriptor = self._open_trace_root(tenant)
        if descriptor is None:
            return True
        try:
            names = tuple(os.listdir(descriptor))
            if len(names) > _MAX_TRACES_PER_TENANT:
                raise RecallTraceEraseIntegrityError("recall trace count exceeds its cleanup bound")
            for name in names:
                trace_id = _trace_id_from_name(name)
                payload, opened = self._read_trace(descriptor, name, trace_id)
                try:
                    scope = payload.get("scope")
                    trace_owner = ""
                    if isinstance(scope, dict):
                        trace_tenant = str(scope.get("tenant_id") or "")
                        if trace_tenant and trace_tenant != tenant:
                            raise RecallTraceEraseIntegrityError("recall trace crosses its tenant directory")
                        trace_owner = str(scope.get("user_id") or scope.get("owner_user_id") or "")
                    if trace_owner:
                        try:
                            trace_owner = MemoryDocumentPathPolicy.trusted_segment(trace_owner, "trace owner_user_id")
                        except ValueError as exc:
                            raise RecallTraceEraseIntegrityError("recall trace owner scope is invalid") from exc
                    if trace_owner and trace_owner != owner:
                        continue
                    named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    current = os.fstat(opened)
                    if (
                        not stat.S_ISREG(named.st_mode)
                        or named.st_nlink != 1
                        or (named.st_dev, named.st_ino) != (current.st_dev, current.st_ino)
                    ):
                        raise RecallTraceEraseIntegrityError("recall trace changed while it was being erased")
                    os.unlink(name, dir_fd=descriptor)
                finally:
                    os.close(opened)
            os.fsync(descriptor)
            return True
        finally:
            os.close(descriptor)

    def _open_trace_root(self, tenant_id: str) -> int | None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.root, flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RecallTraceEraseIntegrityError("recall trace artifact root is unsafe") from exc
        parts = ("recall-traces",) if tenant_id == "default" else ("tenants", tenant_id, "recall-traces")
        try:
            for part in parts:
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    os.close(descriptor)
                    return None
                except OSError as exc:
                    raise RecallTraceEraseIntegrityError("recall trace path is unsafe") from exc
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _read_trace(directory_descriptor: int, name: str, trace_id: str) -> tuple[dict[str, Any], int]:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise RecallTraceEraseIntegrityError("recall trace is unsafe") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > _MAX_TRACE_BYTES:
                raise RecallTraceEraseIntegrityError("recall trace is not one bounded regular file")
            chunks: list[bytes] = []
            remaining = _MAX_TRACE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > _MAX_TRACE_BYTES:
                raise RecallTraceEraseIntegrityError("recall trace exceeds its cleanup bound")
            payload = json.loads(raw.decode("utf-8", errors="strict"))
            if not isinstance(payload, dict) or payload.get("trace_id") != trace_id:
                raise RecallTraceEraseIntegrityError("recall trace identity is invalid")
            return payload, descriptor
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            os.close(descriptor)
            raise RecallTraceEraseIntegrityError("recall trace is invalid JSON") from exc
        except BaseException:
            os.close(descriptor)
            raise


def _trace_id_from_name(name: str) -> str:
    if not name.endswith(".json") or "/" in name:
        raise RecallTraceEraseIntegrityError("recall trace directory contains an unexpected artifact")
    raw = name.removesuffix(".json")
    try:
        parsed = str(uuid.UUID(raw))
    except (AttributeError, TypeError, ValueError):
        raise RecallTraceEraseIntegrityError("recall trace filename is invalid") from None
    if parsed != raw:
        raise RecallTraceEraseIntegrityError("recall trace filename is not canonical")
    return parsed


__all__ = ["RecallTraceEraseBackend", "RecallTraceEraseIntegrityError"]
