"""Fail-closed sanitization for every derived context projection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any
from urllib.parse import unquote, urlsplit

from memoryos.adapters.agent_hooks.sanitizer import sanitize_text

_SECRET_KEY = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|set-cookie|"
    r"password|passwd|pwd|secret|private[_-]?key|database[_-]?(?:url|dsn|password)|ssh[_-]?)"
)
_PRIVATE_FIELD_KEY = re.compile(
    r"(?i)^(?:environment|env|env_vars|environment_variables|ssh_config|ssh_info|"
    r"credit_card|card_number|cvv|social_security|national_id|private_data)$"
)
_AUTH_VALUE = re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]{6,}")
_AUTH_HEADER = re.compile(r"(?im)\b((?:proxy-)?authorization)\s*:\s*([^\r\n]+)")
_TOKEN_VALUE = re.compile(
    r"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{12,}|"
    r"AKIA[A-Z0-9]{12,}|xox[baprs]-[A-Za-z0-9-]{10,})(?![A-Za-z0-9])"
)
_COOKIE_VALUE = re.compile(r"(?im)\b(cookie|set-cookie)\s*:\s*([^\r\n]+)")
_URL_CREDENTIAL = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")
_SSH_PRIVATE = re.compile(
    r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----.*?"
    r"-----END (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----",
    re.DOTALL,
)
_PEM_BLOCK = re.compile(r"-----BEGIN [A-Z0-9 ]+-----.*?-----END [A-Z0-9 ]+-----", re.DOTALL)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)\b([a-z0-9_]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|pwd|secret)"
    r"[a-z0-9_]*)(\s*[:=]\s*)([^\s,;]+)"
)
_UNREDACTED_ASSIGNMENT_SECRET = re.compile(
    r"(?i)\b([a-z0-9_]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|pwd|secret)"
    r"[a-z0-9_]*)(\s*[:=]\s*)(?![\"']?<redacted)([^\s,;]+)"
)
_ABSOLUTE_PATH = re.compile(
    r"(?:file:///(?:Users|home|private|tmp|var)/[^\s\"'<>|;,)]*|"
    r"(?<![\w:/])(?:/Users|/home|/private|/tmp|/var|/[A-Za-z0-9_.-]+/)[^\s\"'<>|;,)]*)"
)
_TREE_SEGMENT = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_TREE_USER_DIRECTORY = re.compile(r"(?i)^(?:users|home|private|tmp|var)[._:-].+")
_TREE_USER_DIRECTORY_ROOTS = frozenset({"users", "home", "private", "tmp", "var"})
_TREE_PATH_KEYS = frozenset(
    {
        "tree_path",
        "tree_paths",
        "primary_tree_path",
        "secondary_tree_paths",
        "target_paths",
        "path_prefixes",
    }
)

MAX_PROJECTION_TEXT = 8_000
MAX_TOOL_RESULT_TEXT = 4_000
MAX_METADATA_DEPTH = 8
MAX_COLLECTION_ITEMS = 128


class ContextProjectionSanitizationError(ValueError):
    """The projection cannot be made safe without retaining raw input."""


@dataclass(frozen=True)
class SanitizedContextProjection:
    title: str
    l0_text: str
    l1_text: str
    metadata: dict[str, Any]
    resource_name: str = ""
    resource_location: str = ""
    redacted: bool = False
    truncated: bool = False


class ContextProjectionSanitizer:
    """Create a bounded derived view while leaving immutable evidence untouched."""

    def sanitize(
        self,
        *,
        title: object,
        l0_text: object = "",
        l1_text: object = "",
        metadata: Mapping[str, Any] | None = None,
        source_kind: str = "",
    ) -> SanitizedContextProjection:
        try:
            safe_metadata, metadata_redacted, metadata_truncated = self._sanitize_value(
                dict(metadata or {}),
                depth=0,
                path=(),
            )
            safe_title, title_redacted, title_truncated = self._sanitize_string(
                str(title or ""),
                max_text=1_000,
            )
            l0_limit = 2_000
            l1_limit = MAX_TOOL_RESULT_TEXT if source_kind == "tool_result" else MAX_PROJECTION_TEXT
            safe_l0, l0_redacted, l0_truncated = self._sanitize_string(
                self._text(l0_text),
                max_text=l0_limit,
            )
            safe_l1, l1_redacted, l1_truncated = self._sanitize_string(
                self._text(l1_text),
                max_text=l1_limit,
            )
        except ContextProjectionSanitizationError:
            raise
        except Exception as exc:  # A derived index must never receive partially sanitized data.
            raise ContextProjectionSanitizationError(type(exc).__name__) from exc

        resource_name, resource_location = self._resource_identity(dict(metadata or {}))
        if resource_name:
            safe_metadata["resource_name"] = resource_name
        if resource_location:
            safe_metadata["resource_location"] = resource_location
        for key in tuple(safe_metadata):
            if self._path_key(key):
                safe_metadata.pop(key, None)
        safe_metadata["projection_sanitized"] = True
        redacted = any((metadata_redacted, title_redacted, l0_redacted, l1_redacted))
        truncated = any((metadata_truncated, title_truncated, l0_truncated, l1_truncated))
        safe_metadata["projection_redacted"] = redacted
        safe_metadata["projection_truncated"] = truncated
        self.assert_safe({"title": safe_title, "l0": safe_l0, "l1": safe_l1, "metadata": safe_metadata})
        return SanitizedContextProjection(
            title=safe_title,
            l0_text=safe_l0,
            l1_text=safe_l1,
            metadata=safe_metadata,
            resource_name=resource_name,
            resource_location=resource_location,
            redacted=redacted,
            truncated=truncated,
        )

    def sanitize_trace(self, value: Any) -> Any:
        sanitized, _redacted, _truncated = self._sanitize_value(value, depth=0, path=())
        self.assert_safe(sanitized)
        return sanitized

    def sanitize_tree_segments(self, values: Sequence[object]) -> tuple[str, ...]:
        """Return stable, non-secret dynamic segments for a logical tree path.

        Tree paths are serving taxonomy, not evidence.  Legal schema-owned IDs
        stay readable so existing path queries remain stable.  A credential or
        a value that resembles a user-directory chain is deterministically
        pseudonymized instead of being copied into Catalog metadata or the
        normalized ``context_paths`` table.  When one segment identifies a
        directory chain, all sibling dynamic segments are pseudonymized so a
        username cannot survive beside a redacted ``Users``/``home`` marker.
        """

        raw = tuple(str(value).strip() for value in values)
        if any(not value or value in {".", ".."} or "\x00" in value for value in raw):
            raise ContextProjectionSanitizationError("tree path contains an invalid dynamic segment")
        directory_chain = any(self._tree_user_directory(value) for value in raw) or bool(
            len(raw) > 1 and raw[0].casefold() in _TREE_USER_DIRECTORY_ROOTS
        )
        return tuple(self.sanitize_tree_segment(value, force_hash=directory_chain) for value in raw)

    def sanitize_tree_segment(self, value: object, *, force_hash: bool = False) -> str:
        """Preserve a safe dynamic segment or replace sensitive input by digest."""

        try:
            raw = str(value).strip()
            if not raw or raw in {".", ".."} or "\x00" in raw:
                raise ContextProjectionSanitizationError("tree path contains an invalid dynamic segment")
            if force_hash or self._tree_segment_sensitive(raw):
                return f"id-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"
            if not _TREE_SEGMENT.fullmatch(raw):
                raise ContextProjectionSanitizationError("tree path dynamic segment is not normalized")
            self.assert_safe(raw)
            return raw
        except ContextProjectionSanitizationError:
            raise
        except Exception as exc:
            raise ContextProjectionSanitizationError("tree path dynamic segment sanitization failed") from exc

    def assert_safe(self, value: Any) -> None:
        for text in self._string_values(value):
            if (
                _SSH_PRIVATE.search(text)
                or _AUTH_VALUE.search(text)
                or self._header_has_secret(_AUTH_HEADER, text)
                or _TOKEN_VALUE.search(text)
                or self._header_has_secret(_COOKIE_VALUE, text)
                or _UNREDACTED_ASSIGNMENT_SECRET.search(text)
                or _URL_CREDENTIAL.search(text)
            ):
                raise ContextProjectionSanitizationError("credential remained after sanitization")
            if _ABSOLUTE_PATH.search(text):
                raise ContextProjectionSanitizationError("absolute path remained after sanitization")

    @staticmethod
    def _string_values(value: Any) -> tuple[str, ...]:
        pending = [value]
        strings: list[str] = []
        while pending:
            current = pending.pop()
            if isinstance(current, str):
                strings.append(current)
            elif isinstance(current, Mapping):
                pending.extend(current.values())
            elif isinstance(current, Sequence) and not isinstance(current, bytes | bytearray | memoryview):
                pending.extend(current)
        return tuple(strings)

    @staticmethod
    def _header_has_secret(pattern: re.Pattern[str], text: str) -> bool:
        return any(match.group(2).strip(" \t\"'").casefold() != "<redacted>" for match in pattern.finditer(text))

    def digest(self, value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _sanitize_value(
        self,
        value: Any,
        *,
        depth: int,
        path: tuple[str, ...],
    ) -> tuple[Any, bool, bool]:
        if depth > MAX_METADATA_DEPTH:
            return "<truncated-depth>", False, True
        if value is None or isinstance(value, bool | int | float):
            return value, False, False
        if isinstance(value, bytes | bytearray | memoryview):
            return "<binary>", True, False
        if isinstance(value, str):
            if path and self._path_key(path[-1]):
                name, location = self.sanitize_path(value)
                return ({"basename": name, "location": location} if name else "<redacted-path>"), True, False
            return self._sanitize_string(value, max_text=MAX_PROJECTION_TEXT)
        if isinstance(value, Mapping):
            mapping_result: dict[str, Any] = {}
            redacted = False
            truncated = False
            for index, (raw_key, item) in enumerate(value.items()):
                if index >= MAX_COLLECTION_ITEMS:
                    truncated = True
                    break
                key = str(raw_key)[:200]
                if _SECRET_KEY.search(key) or _PRIVATE_FIELD_KEY.fullmatch(key):
                    mapping_result[key] = "<redacted>"
                    redacted = True
                    continue
                if self._tree_path_key(key):
                    safe, item_redacted, item_truncated = self._sanitize_tree_path_metadata(item)
                    mapping_result[key] = safe
                    redacted = redacted or item_redacted
                    truncated = truncated or item_truncated
                    continue
                safe, item_redacted, item_truncated = self._sanitize_value(
                    item,
                    depth=depth + 1,
                    path=(*path, key),
                )
                mapping_result[key] = safe
                redacted = redacted or item_redacted
                truncated = truncated or item_truncated
            return mapping_result, redacted, truncated
        if isinstance(value, Sequence):
            sequence_result: list[Any] = []
            redacted = False
            truncated = len(value) > MAX_COLLECTION_ITEMS
            for item in list(value)[:MAX_COLLECTION_ITEMS]:
                safe, item_redacted, item_truncated = self._sanitize_value(
                    item,
                    depth=depth + 1,
                    path=path,
                )
                sequence_result.append(safe)
                redacted = redacted or item_redacted
                truncated = truncated or item_truncated
            return sequence_result, redacted, truncated
        safe, redacted, truncated = self._sanitize_string(str(value), max_text=2_000)
        return safe, redacted, truncated

    def _sanitize_tree_path_metadata(self, value: Any) -> tuple[Any, bool, bool]:
        """Sanitize path mirrors recursively without trusting metadata shape."""

        if isinstance(value, str):
            raw = value.strip().strip("/")
            if not raw or "\\" in raw or "//" in raw:
                return f"id-{self.digest(value)[:24]}", True, False
            parts = tuple(raw.split("/"))
            try:
                dynamic_start = self._tree_metadata_dynamic_start(parts)
                safe_parts = (
                    *parts[:dynamic_start],
                    *self.sanitize_tree_segments(parts[dynamic_start:]),
                )
            except ContextProjectionSanitizationError:
                return f"id-{self.digest(value)[:24]}", True, False
            safe = "/".join(safe_parts)
            return safe, safe != raw, False
        if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | memoryview):
            safe_items: list[Any] = []
            redacted = False
            truncated = len(value) > MAX_COLLECTION_ITEMS
            for item in list(value)[:MAX_COLLECTION_ITEMS]:
                safe, item_redacted, item_truncated = self._sanitize_tree_path_metadata(item)
                safe_items.append(safe)
                redacted = redacted or item_redacted
                truncated = truncated or item_truncated
            return safe_items, redacted, truncated
        return "<redacted-tree-path>", True, False

    def _sanitize_string(self, text: str, *, max_text: int) -> tuple[str, bool, bool]:
        if "\x00" in text:
            return "<binary>", True, False
        original = text
        text = _SSH_PRIVATE.sub("<redacted-private-key>", text)
        text = _PEM_BLOCK.sub("<redacted-pem>", text)
        text = _AUTH_HEADER.sub(r"\1: <redacted>", text)
        text = _AUTH_VALUE.sub("<redacted-authorization>", text)
        text = _TOKEN_VALUE.sub("<redacted-token>", text)
        text = _COOKIE_VALUE.sub(r"\1: <redacted>", text)
        text = _URL_CREDENTIAL.sub(r"\1<redacted>@", text)
        text = _ASSIGNMENT_SECRET.sub(r"\1\2<redacted>", text)
        text = sanitize_text(text, max_text=max_text)
        text = _ABSOLUTE_PATH.sub(self._replace_path_match, text)
        truncated = len(text) > max_text or " chars omitted>" in text or " lines omitted>" in text
        if len(text) > max_text:
            text = text[:max_text] + "\n... <truncated>"
            truncated = True
        return text, text != original, truncated

    def _replace_path_match(self, match: re.Match[str]) -> str:
        name, location = self.sanitize_path(match.group(0))
        if not name:
            return "<redacted-path>"
        return f"{location}/{name}" if location else name

    def sanitize_path(self, value: str) -> tuple[str, str]:
        raw = str(value or "").strip()
        if raw.startswith("file://"):
            raw = unquote(urlsplit(raw).path)
        normalized = raw.replace("\\", "/").rstrip("/")
        if not normalized:
            return "", ""
        name = PurePath(normalized).name
        if name in {"", ".", ".."}:
            return "", ""
        lowered = normalized.casefold()
        if "/desktop/" in f"{lowered}/":
            location = "desktop"
        elif any(token in lowered for token in ("/repositories/", "/repository/", "/pycharmprojects/", "/src/")):
            location = "repository"
        elif any(token in lowered for token in ("/uploads/", "/upload/")):
            location = "uploads"
        elif lowered.startswith(("/tmp/", "/private/tmp/", "/var/folders/")):
            location = "temporary"
        elif lowered.startswith(("/users/", "/home/")):
            location = "user"
        else:
            location = "external"
        safe_name, _redacted, _truncated = self._sanitize_string(name, max_text=255)
        return safe_name.replace("/", "_").replace("\\", "_"), location

    def _resource_identity(self, metadata: Mapping[str, Any]) -> tuple[str, str]:
        candidates: list[str] = []
        for key in ("resource_uri", "file_uri", "path", "file_path", "absolute_path", "source_path"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
        resource = metadata.get("resource")
        if isinstance(resource, Mapping):
            for key in ("uri", "path", "file_path"):
                value = resource.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)
        for candidate in candidates:
            name, location = self.sanitize_path(candidate)
            if name:
                return name, location
        explicit_name = metadata.get("resource_name") or metadata.get("file_name") or metadata.get("filename")
        if explicit_name:
            name, _redacted, _truncated = self._sanitize_string(str(explicit_name), max_text=255)
            return PurePath(name).name, str(metadata.get("resource_location") or "")
        return "", ""

    @staticmethod
    def _path_key(key: object) -> bool:
        normalized = str(key).casefold().replace("-", "_")
        return normalized in {
            "path",
            "file_path",
            "absolute_path",
            "source_path",
            "working_directory",
            "cwd",
            "home",
            "user_directory",
        }

    @staticmethod
    def _tree_path_key(key: object) -> bool:
        normalized = str(key).casefold().replace("-", "_")
        return normalized in _TREE_PATH_KEYS

    @staticmethod
    def _tree_user_directory(value: str) -> bool:
        return bool(_TREE_USER_DIRECTORY.search(value))

    @staticmethod
    def _tree_metadata_dynamic_start(parts: Sequence[str]) -> int:
        if not parts:
            return 0
        if parts[0] in {"sessions", "projects", "skills", "agents"}:
            return 1
        if parts[0] == "memories":
            return min(2, len(parts))
        if parts[0] in {"timeline", "resources"}:
            return len(parts)
        return 0

    def _tree_segment_sensitive(self, value: str) -> bool:
        return bool(
            self._tree_user_directory(value)
            or _SECRET_KEY.search(value)
            or _TOKEN_VALUE.search(value)
            or _AUTH_VALUE.search(value)
            or _UNREDACTED_ASSIGNMENT_SECRET.search(value)
            or _URL_CREDENTIAL.search(value)
            or _ABSOLUTE_PATH.search(value)
            or _SSH_PRIVATE.search(value)
            or _PEM_BLOCK.search(value)
        )

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


__all__ = [
    "ContextProjectionSanitizationError",
    "ContextProjectionSanitizer",
    "SanitizedContextProjection",
]
