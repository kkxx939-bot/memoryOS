"""供上下文检索过滤使用的稳定、非敏感工作区标识。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SAFE_LOGICAL_ID = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")


def normalize_workspace_id(value: object) -> str:
    """保留有界逻辑 ID，并摘要化路径或远程仓库形式的身份。

    工作区 ID 会进入索引、轨迹和向量元数据，因此即使正文已经清洗，也不能把
    原始仓库路径当作普通字符串保存。
    """

    text = str(value or "").strip()
    if not text:
        return ""
    if _SAFE_LOGICAL_ID.fullmatch(text):
        return text
    if _looks_like_repository_path(text):
        return repository_workspace_id(repo_root=text)
    normalized_remote = _normalize_git_remote(text)
    identity = f"repository-reference:{normalized_remote or text}"
    return _project_digest(identity)


def repository_workspace_id(
    repo_root: object = "",
    cwd: object = "",
    git_remote: object = "",
) -> str:
    """生成与 Agent Hook 一致的项目身份，同时不暴露本地路径。"""

    remote = _normalize_git_remote(str(git_remote or ""))
    if remote:
        return _project_digest(remote)
    raw_path = str(repo_root or cwd or ".")
    path = Path(raw_path).expanduser().resolve()
    return _project_digest(f"local-repository-realpath:{path.as_posix()}")


def normalize_workspace_scope_key(value: object) -> str:
    """规范化业务适用范围键中的工作区部分。"""

    text = str(value or "").strip()
    prefix = "memoryos:workspace:"
    if not text.startswith(prefix):
        return text
    return prefix + normalize_workspace_id(text[len(prefix) :])


def workspace_ids_from_metadata(metadata: Any) -> frozenset[str]:
    """从持久化元数据中读取唯一的工作区标识，不从正文猜测。"""

    if not isinstance(metadata, Mapping):
        raise ValueError("object metadata must be a mapping")
    raw_scope = metadata.get("scope", {})
    raw_fields = metadata.get("fields", {})
    if raw_scope is not None and not isinstance(raw_scope, Mapping):
        raise ValueError("object scope must be a mapping")
    if raw_fields is not None and not isinstance(raw_fields, Mapping):
        raise ValueError("object fields must be a mapping")
    scope = dict(raw_scope or {})
    fields = dict(raw_fields or {})
    values = {
        str(value).strip()
        for value in (
            metadata.get("workspace_id"),
            metadata.get("project_id"),
            scope.get("workspace_id"),
            scope.get("project_id"),
            fields.get("workspace_id"),
            fields.get("project_id"),
        )
        if value is not None and str(value).strip()
    }
    applicability = scope.get("applicability", {})
    if applicability is not None and not isinstance(applicability, Mapping):
        raise ValueError("scope applicability must be a mapping")
    all_of = dict(applicability or {}).get("all_of", [])
    if not isinstance(all_of, Sequence) or isinstance(all_of, str | bytes):
        raise ValueError("scope applicability all_of must be an array")
    for item in all_of:
        if not isinstance(item, Mapping):
            raise ValueError("scope applicability entries must be mappings")
        if str(item.get("kind") or "").strip().casefold() == "workspace":
            identifier = str(item.get("id") or "").strip()
            if not identifier:
                raise ValueError("workspace scope requires an id")
            values.add(identifier)
    raw_scope_keys = metadata.get("scope_keys", [])
    if not isinstance(raw_scope_keys, Sequence) or isinstance(raw_scope_keys, str | bytes):
        raise ValueError("scope_keys must be an array")
    for raw_key in raw_scope_keys:
        parts = str(raw_key).split(":", 2)
        if len(parts) == 3 and parts[1] == "workspace" and parts[2]:
            values.add(parts[2])
    if len(values) > 1:
        raise ValueError("object declares multiple workspace boundaries")
    return frozenset(values)


def _project_digest(identity: str) -> str:
    return "project-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _looks_like_repository_path(value: str) -> bool:
    return bool(
        value.startswith(("/", "~/", "file://"))
        or "\\" in value
        or re.match(r"^[A-Za-z]:[\\/]", value)
    )


def _normalize_git_remote(remote: str) -> str:
    value = remote.strip().lower().removesuffix(".git").rstrip("/")
    if not value:
        return ""
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
        return f"{host}/{path}"
    value = value.removeprefix("https://").removeprefix("http://").removeprefix("ssh://")
    if "@" in value.split("/", 1)[0]:
        value = value.split("@", 1)[1]
    return value


__all__ = [
    "normalize_workspace_id",
    "normalize_workspace_scope_key",
    "repository_workspace_id",
    "workspace_ids_from_metadata",
]
