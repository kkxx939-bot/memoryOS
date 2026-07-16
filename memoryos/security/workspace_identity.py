"""Stable, non-sensitive workspace identities for serving and ACL fields."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SAFE_LOGICAL_ID = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")


def normalize_workspace_id(value: object) -> str:
    """Preserve bounded logical IDs and hash path/remote-like identities.

    Workspace IDs are indexed, traced, and copied to vector metadata.  A raw
    repository path therefore cannot be treated as an ordinary string even
    when the projection body itself has already been sanitized.
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
    """Match the Agent hook project identity without exposing a local path."""

    remote = _normalize_git_remote(str(git_remote or ""))
    if remote:
        return _project_digest(remote)
    raw_path = str(repo_root or cwd or ".")
    path = Path(raw_path).expanduser().resolve()
    return _project_digest(f"local-repository-realpath:{path.as_posix()}")


def normalize_workspace_scope_key(value: object) -> str:
    """Normalize the workspace component of a trusted applicability key."""

    text = str(value or "").strip()
    prefix = "memoryos:workspace:"
    if not text.startswith(prefix):
        return text
    return prefix + normalize_workspace_id(text[len(prefix) :])


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
]
