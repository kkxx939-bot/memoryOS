"""跨领域共享的业务适用范围模型。

Scope 只描述“这条上下文适用于哪里”，不承担用户身份、租户可见性或写入授权。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, unquote

CORE_SCOPE_KINDS = frozenset({"principal", "workspace", "environment", "asset", "location", "episode", "global"})
HIERARCHICAL_SCOPE_KINDS = frozenset({"asset", "location"})


class ScopeResolutionSource(str, Enum):
    """记录作用域候选的有限来源，避免来源信息变成任意字符串。"""

    EXPLICIT = "explicit"
    EVENT = "event"
    ORIGIN = "origin"
    SCHEMA_DEFAULT = "schema_default"
    INFERRED = "inferred"


def _required(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("scope confidence must be a finite number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("scope confidence must be a finite number between 0 and 1")
    return result


def _path_key(parts: tuple[str, ...]) -> str:
    return "/".join(quote(part, safe="") for part in parts)


def _canonical_scope_reference_path(value: str) -> tuple[str, ...] | None:
    """将规范作用域键展开为完整的逻辑 ID 路径。"""

    parts = value.split(":", 3)
    if len(parts) < 3 or parts[1] not in CORE_SCOPE_KINDS:
        return None
    if parts[2] != "path":
        identifier = ":".join(parts[2:]).strip()
        return (identifier,) if identifier else None
    if len(parts) != 4:
        raise ValueError("canonical scope parent path is malformed")
    logical = tuple(unquote(item).strip() for item in parts[3].split("/"))
    if not logical or any(not item for item in logical):
        raise ValueError("canonical scope parent path is malformed")
    return logical


def _append_logical_path(current: list[str], incoming: tuple[str, ...]) -> None:
    """追加完整父路径，同时消除已经存在的重叠前缀。"""

    overlap = 0
    maximum = min(len(current), len(incoming))
    for width in range(maximum, 0, -1):
        if tuple(current[-width:]) == incoming[:width]:
            overlap = width
            break
    current.extend(incoming[overlap:])


def _logical_parent_path(values: Sequence[str]) -> tuple[str, ...]:
    logical: list[str] = []
    for value in values:
        normalized = _required(value, "scope parent_path item")
        expanded = _canonical_scope_reference_path(normalized)
        if expanded is None:
            logical.append(normalized)
        else:
            _append_logical_path(logical, expanded)
    return tuple(logical)


def _string_array(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be an array of non-empty strings")
    return tuple(_required(item, f"{name} item") for item in value)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class ScopeRef:
    """带来源信息且种类有限的 Identity V2 作用域引用。"""

    namespace: str
    kind: str
    id: str
    parent_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    parent_path: tuple[str, ...] = ()
    confidence: float = 1.0
    source: ScopeResolutionSource | str = ScopeResolutionSource.EXPLICIT
    inferred: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _required(self.namespace, "scope namespace"))
        kind = _required(self.kind, "scope kind").lower()
        if kind not in CORE_SCOPE_KINDS:
            raise ValueError(f"unsupported core scope kind: {kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", _required(self.id, "scope id"))
        parent_reference = _required(self.parent_id, "scope parent_id") if self.parent_id is not None else None
        if not isinstance(self.parent_path, Sequence) or isinstance(self.parent_path, str | bytes):
            raise ValueError("scope parent_path must be an array of non-empty strings")
        path = _logical_parent_path(self.parent_path)
        parent_reference_path = (
            _canonical_scope_reference_path(parent_reference) if parent_reference is not None else None
        )
        if parent_reference is not None and parent_reference_path is None:
            parent_reference_path = (parent_reference,)
        if parent_reference_path and path and parent_reference_path[-1] != path[-1]:
            raise ValueError("scope parent_id must equal the final parent_path item")
        if kind not in HIERARCHICAL_SCOPE_KINDS and (parent_reference is not None or path):
            raise ValueError(f"scope kind {kind} does not support parent hierarchy")
        if parent_reference_path and not path:
            path = parent_reference_path
        elif parent_reference_path and len(parent_reference_path) > 1 and path != parent_reference_path:
            raise ValueError("scope parent_id path must equal parent_path")
        parent_id = path[-1] if path else None
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "parent_path", path)
        if not isinstance(self.attributes, Mapping):
            raise ValueError("scope attributes must be an object")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        source = (
            self.source if isinstance(self.source, ScopeResolutionSource) else ScopeResolutionSource(str(self.source))
        )
        object.__setattr__(self, "source", source)
        if not isinstance(self.inferred, bool):
            raise ValueError("scope inferred must be a boolean")

    @property
    def hierarchy_path(self) -> tuple[str, ...]:
        return (*self.parent_path, self.id)

    @property
    def key(self) -> str:
        """生成包含必要父级层次的确定性身份键。"""

        if self.kind not in HIERARCHICAL_SCOPE_KINDS or not self.parent_path:
            return f"{self.namespace}:{self.kind}:{self.id}"
        return f"{self.namespace}:{self.kind}:path:{_path_key(self.hierarchy_path)}"

    @property
    def key_candidates(self) -> tuple[str, ...]:
        """返回安全读取键；带父级的作用域禁止模糊降级匹配。"""

        return (self.key,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "kind": self.kind,
            "id": self.id,
            "parent_id": self.parent_id,
            "parent_path": list(self.parent_path),
            "attributes": dict(self.attributes),
            "confidence": self.confidence,
            "source": ScopeResolutionSource(self.source).value,
            "inferred": self.inferred,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScopeRef:
        _mapping(payload, "scope")
        parent_id = payload.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("scope parent_id must be a non-empty string")
        attributes = payload.get("attributes", {})
        _mapping(attributes, "scope attributes")
        parent_path = _string_array(payload.get("parent_path", []), "scope parent_path")
        inferred = payload.get("inferred", False)
        if not isinstance(inferred, bool):
            raise ValueError("scope inferred must be a boolean")
        return cls(
            namespace=_required(payload.get("namespace", "memoryos"), "scope namespace"),
            kind=_required(payload.get("kind"), "scope kind"),
            id=_required(payload.get("id"), "scope id"),
            parent_id=parent_id,
            attributes=attributes,
            parent_path=parent_path,
            confidence=payload.get("confidence", 1.0),
            source=_required(
                payload.get("source", ScopeResolutionSource.EXPLICIT.value),
                "scope source",
            ),
            inferred=inferred,
        )


def scope_key_from_payload(payload: ScopeRef | Mapping[str, Any]) -> str:
    """返回身份判断和最终过滤共同使用的唯一规范键。"""

    return payload.key if isinstance(payload, ScopeRef) else ScopeRef.from_dict(payload).key


def scope_key_candidates_from_payload(
    payload: ScopeRef | Mapping[str, Any],
) -> tuple[str, ...]:
    """返回唯一安全的 Identity V2 作用域键。"""

    scope = payload if isinstance(payload, ScopeRef) else ScopeRef.from_dict(payload)
    return scope.key_candidates


def scope_keys_from_payloads(payloads: object) -> tuple[str, ...]:
    """完整解析 all_of，确保非法成员不会被静默丢弃。"""

    if not isinstance(payloads, Sequence) or isinstance(payloads, str | bytes):
        raise ValueError("scope applicability all_of must be an array")
    keys: list[str] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("scope applicability all_of must contain scope objects")
        keys.append(scope_key_from_payload(payload))
    return tuple(dict.fromkeys(keys))


@dataclass(frozen=True)
class ScopeSelector:
    """适用范围选择器，与主体身份和可见性策略相互独立。"""

    all_of: tuple[ScopeRef, ...]

    def __post_init__(self) -> None:
        if not self.all_of:
            raise ValueError("applicability scope must contain at least one scope")
        unique = {scope.key: scope for scope in self.all_of}
        object.__setattr__(self, "all_of", tuple(unique[key] for key in sorted(unique)))

    def to_dict(self) -> dict[str, Any]:
        return {"all_of": [scope.to_dict() for scope in self.all_of]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScopeSelector:
        _mapping(payload, "scope applicability")
        raw_all_of = payload.get("all_of")
        if not isinstance(raw_all_of, Sequence) or isinstance(raw_all_of, str | bytes):
            raise ValueError("scope applicability all_of must be an array")
        if any(not isinstance(item, Mapping) for item in raw_all_of):
            raise ValueError("scope applicability all_of must contain scope objects")
        return cls(tuple(ScopeRef.from_dict(item) for item in raw_all_of))


__all__ = [
    "CORE_SCOPE_KINDS",
    "HIERARCHICAL_SCOPE_KINDS",
    "ScopeRef",
    "ScopeResolutionSource",
    "ScopeSelector",
    "scope_key_candidates_from_payload",
    "scope_key_from_payload",
    "scope_keys_from_payloads",
]
