"""Markdown 记忆的受控逻辑路径与稳定 URI。"""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath

from foundation.ids import require_safe_path_segment
from memory.core.model import MemoryDocumentKind
from memory.core.structure.frontmatter import validate_document_id

_DYNAMIC_FILE = re.compile(r"^[\w.-]{1,160}\.md$", re.UNICODE)


class MemoryDocumentPathPolicy:
    max_relative_path_bytes = 512

    @classmethod
    def trusted_segment(cls, value: object, label: str) -> str:
        segment = require_safe_path_segment(value, label)
        if unicodedata.normalize("NFC", segment) != segment:
            raise ValueError(f"{label} must use NFC normalization")
        return segment

    @classmethod
    def normalize_relative_path(cls, value: object) -> str:
        raw = str(value or "")
        if not raw or raw.startswith("/") or "\\" in raw or "\x00" in raw:
            raise ValueError("memory document path must be a safe POSIX relative path")
        if len(raw.encode("utf-8")) > cls.max_relative_path_bytes:
            raise ValueError("memory document path is too long")
        raw_parts = raw.split("/")
        if any(part in {"", ".", ".."} for part in raw_parts):
            raise ValueError("memory document path contains an unsafe segment")
        path = PurePosixPath(raw)
        parts = path.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("memory document path contains an unsafe segment")
        for part in parts:
            if unicodedata.normalize("NFC", part) != part:
                raise ValueError("memory document path must use NFC normalization")
        normalized = "/".join(parts)
        cls.kind_for(normalized)
        return normalized

    @classmethod
    def kind_for(cls, relative_path: object) -> MemoryDocumentKind:
        value = str(relative_path or "")
        fixed = {
            "MEMORY.md": MemoryDocumentKind.ROOT_INDEX,
            "profile.md": MemoryDocumentKind.PROFILE,
            "preferences.md": MemoryDocumentKind.PREFERENCES,
            "knowledge/MEMORY.md": MemoryDocumentKind.KNOWLEDGE_INDEX,
            "knowledge/open-loops.md": MemoryDocumentKind.OPEN_LOOPS,
        }
        if value in fixed:
            return fixed[value]
        parts = value.split("/")
        dynamic_roots = {
            ("knowledge", "entities"): MemoryDocumentKind.ENTITY,
            ("knowledge", "topics"): MemoryDocumentKind.TOPIC,
            ("knowledge", "episodes"): MemoryDocumentKind.EPISODE,
            ("experiences",): MemoryDocumentKind.EXPERIENCE,
        }
        for prefix, kind in dynamic_roots.items():
            if tuple(parts[:-1]) == prefix and _DYNAMIC_FILE.fullmatch(parts[-1] or ""):
                if parts[-1].startswith(".") or parts[-1] in {"MEMORY.md", "open-loops.md"}:
                    break
                return kind
        raise ValueError("memory document path is outside the controlled layout")

    @classmethod
    def collision_key(cls, relative_path: object) -> str:
        normalized = cls.normalize_relative_path(relative_path)
        return unicodedata.normalize("NFC", normalized).casefold()

    @classmethod
    def document_uri(cls, owner_user_id: object, document_id: object) -> str:
        owner = cls.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        return f"memoryos://user/{owner}/memory/documents/{identifier}"

    @classmethod
    def parse_document_uri(cls, uri: object) -> tuple[str, str]:
        raw = str(uri or "")
        prefix = "memoryos://user/"
        if not raw.startswith(prefix):
            raise ValueError("document URI must use the user authority")
        parts = raw[len(prefix) :].split("/")
        if len(parts) != 4 or parts[1:3] != ["memory", "documents"]:
            raise ValueError("document URI has an invalid shape")
        return cls.trusted_segment(parts[0], "owner_user_id"), validate_document_id(parts[3])


__all__ = ["MemoryDocumentPathPolicy"]
