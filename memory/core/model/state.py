"""记忆文档路径、登记与扫描状态。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DocumentEditKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"


class RegistrationStatus(str, Enum):
    MANAGED = "managed"
    UNMANAGED = "unmanaged"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class AbsentPath:
    """受控相对路径当前不存在。"""


@dataclass(frozen=True)
class PresentPath:
    relative_path: str
    raw_sha256: str
    size: int

    def __post_init__(self) -> None:
        if not self.relative_path or len(self.raw_sha256) != 64 or self.size < 0:
            raise ValueError("invalid PRESENT raw path state")


@dataclass(frozen=True)
class UnsafePath:
    relative_path: str
    reason: str

    def __post_init__(self) -> None:
        if not self.relative_path or not self.reason:
            raise ValueError("invalid UNSAFE raw path state")


RawPathState = AbsentPath | PresentPath | UnsafePath
ABSENT = AbsentPath()


@dataclass(frozen=True)
class ManagedDocument:
    relative_path: str
    document_id: str
    raw_sha256: str
    size: int
    status: RegistrationStatus = field(default=RegistrationStatus.MANAGED, init=False)


@dataclass(frozen=True)
class UnmanagedDocument:
    relative_path: str
    raw_sha256: str
    size: int
    reason: str
    status: RegistrationStatus = field(default=RegistrationStatus.UNMANAGED, init=False)


@dataclass(frozen=True)
class QuarantinedDocument:
    relative_path: str
    reason: str
    raw_sha256: str = ""
    size: int = 0
    status: RegistrationStatus = field(default=RegistrationStatus.QUARANTINED, init=False)


DocumentRegistrationState = ManagedDocument | UnmanagedDocument | QuarantinedDocument


@dataclass(frozen=True)
class ScanGeneration:
    generation_id: str
    tenant_id: str
    owner_user_id: str
    root_identity: str
    observed_at: str
    complete: bool
    registrations: tuple[DocumentRegistrationState, ...] = ()
    unsafe_paths: tuple[UnsafePath, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def managed(self) -> tuple[ManagedDocument, ...]:
        return tuple(item for item in self.registrations if isinstance(item, ManagedDocument))


def raw_state_to_dict(state: RawPathState) -> dict[str, Any]:
    if isinstance(state, AbsentPath):
        return {"state": "ABSENT"}
    if isinstance(state, PresentPath):
        return {
            "state": "PRESENT",
            "relative_path": state.relative_path,
            "raw_sha256": state.raw_sha256,
            "size": state.size,
        }
    return {"state": "UNSAFE", "relative_path": state.relative_path, "reason": state.reason}


def raw_state_from_dict(payload: Mapping[str, Any]) -> RawPathState:
    state = str(payload.get("state") or "")
    if state == "ABSENT":
        return ABSENT
    if state == "PRESENT":
        return PresentPath(
            relative_path=str(payload["relative_path"]),
            raw_sha256=str(payload["raw_sha256"]),
            size=int(payload["size"]),
        )
    if state == "UNSAFE":
        return UnsafePath(relative_path=str(payload["relative_path"]), reason=str(payload["reason"]))
    raise ValueError("unknown raw path state")


__all__ = [
    "ABSENT",
    "AbsentPath",
    "DocumentEditKind",
    "DocumentRegistrationState",
    "ManagedDocument",
    "PresentPath",
    "QuarantinedDocument",
    "RawPathState",
    "RegistrationStatus",
    "ScanGeneration",
    "UnmanagedDocument",
    "UnsafePath",
    "raw_state_from_dict",
    "raw_state_to_dict",
]
