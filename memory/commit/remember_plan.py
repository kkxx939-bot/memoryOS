"""显式 remember 命令的确定性 Markdown CAS 计划。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from foundation.ids import stable_hash
from memory.core.model import ABSENT, DocumentEditKind, DocumentEditPlan, PresentPath, UnsafePath
from memory.core.structure.frontmatter import parse_front_matter, render_new_document
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import DocumentConflictError, DocumentUnsafeError, MemoryDocumentStore

_H2_LINE = re.compile(r"^[ ]{0,3}##(?!#)[ \t]+(.+?)[ \t]*(?:\r?\n)?$")
_FENCE_OPEN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*?)(?:\r?\n)?$")


class RememberTargetKind(str, Enum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    ENTITY = "entity"
    TOPIC = "topic"
    EPISODE = "episode"
    OPEN_LOOP = "open_loop"


@dataclass(frozen=True)
class RememberTarget:
    kind: RememberTargetKind
    subject: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RememberTargetKind):
            raise TypeError("remember target kind is invalid")
        _heading(self.subject)


class ExplicitRememberPlanner:
    """只处理本地用户显式 remember，不接收 Session 或模型自动形成输入。"""

    def __init__(
        self,
        store: MemoryDocumentStore,
        *,
        max_front_matter_bytes: int = 32 * 1024,
        max_front_matter_depth: int = 12,
        max_edit_bytes: int = 256 * 1024,
    ) -> None:
        if max_front_matter_bytes <= 0 or max_front_matter_depth <= 0 or max_edit_bytes <= 0:
            raise ValueError("invalid explicit remember planner bounds")
        self.store = store
        self.max_front_matter_bytes = max_front_matter_bytes
        self.max_front_matter_depth = max_front_matter_depth
        self.max_edit_bytes = max_edit_bytes

    def plan(
        self,
        content: str,
        target: RememberTarget,
        *,
        tenant_id: str,
        owner_user_id: str,
        idempotency_key: str,
        command_digest: str,
    ) -> DocumentEditPlan:
        body = str(content or "").strip()
        if not body:
            raise ValueError("remember content is required")
        if len(command_digest) != 64 or any(character not in "0123456789abcdef" for character in command_digest):
            raise ValueError("remember command digest must be a lowercase SHA-256 digest")
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        relative_path = self._route(target)
        state = self.store.read_state(tenant, owner, relative_path)
        if state == ABSENT:
            document_id = f"memdoc_{stable_hash([tenant, owner, relative_path, idempotency_key], 32)}"
            raw = render_new_document(document_id, f"# {_heading(target.subject)}\n")
            edit_kind = DocumentEditKind.CREATE
            expected_registration = ""
        elif isinstance(state, PresentPath):
            raw = self.store.read_raw(tenant, owner, relative_path=relative_path)
            if hashlib.sha256(raw).hexdigest() != state.raw_sha256 or len(raw) != state.size:
                raise DocumentConflictError("memory document changed during explicit remember read")
            edit_kind = DocumentEditKind.UPDATE
            expected_registration = ""
        elif isinstance(state, UnsafePath):
            raise DocumentUnsafeError(state.reason)
        else:  # pragma: no cover - RawPathState is a closed union.
            raise DocumentUnsafeError("unknown memory document raw state")

        parsed = parse_front_matter(
            raw,
            max_header_bytes=self.max_front_matter_bytes,
            max_depth=self.max_front_matter_depth,
        )
        document_id = parsed.document_id
        if isinstance(state, PresentPath):
            expected_registration = document_id
        after_body = self._append_to_section(parsed.body, target.subject, body)
        after = parsed.header_bytes + after_body.encode("utf-8")
        if len(after) > self.max_edit_bytes:
            raise ValueError("explicit remember edit exceeds its byte bound")
        return DocumentEditPlan(
            idempotency_key=idempotency_key,
            tenant_id=tenant,
            owner_user_id=owner,
            edit_kind=edit_kind,
            expected_state=state,
            evidence_digest=command_digest,
            edit_summary=f"explicit_remember:{target.kind.value}",
            document_id=document_id,
            relative_path=relative_path,
            after_bytes=after,
            expected_registration_document_id=expected_registration,
        )

    @classmethod
    def _append_to_section(cls, body: str, subject: str, content: str) -> str:
        matches = tuple(
            span
            for heading, span in cls._section_spans(body)
            if cls._normalized_heading(heading) == cls._normalized_heading(subject)
        )
        if len(matches) > 1:
            raise DocumentConflictError("explicit remember target section is ambiguous")
        normalized_content = cls._normalized_block(content)
        if matches:
            start, end = matches[0]
            if normalized_content in cls._normalized_block(body[start:end]):
                return body
            return body[:end] + cls._append_fragment(body[:end], content, before_heading=end < len(body)) + body[end:]
        line_ending = cls._line_ending(body)
        section = f"## {_heading(subject)}{line_ending}{line_ending}{content.strip()}{line_ending}"
        return body + cls._separator(body, line_ending=line_ending, blank_line=True) + section

    @classmethod
    def _section_spans(cls, body: str) -> tuple[tuple[str, tuple[int, int]], ...]:
        headings: list[tuple[str, int]] = []
        fence_marker = ""
        fence_length = 0
        offset = 0
        for line in body.splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            if fence_marker:
                candidate = stripped.lstrip(" ")
                if (
                    len(stripped) - len(candidate) <= 3
                    and candidate.startswith(fence_marker * fence_length)
                    and not candidate.lstrip(fence_marker).strip()
                ):
                    fence_marker = ""
                    fence_length = 0
                offset += len(line)
                continue
            opened = _FENCE_OPEN.fullmatch(line)
            if opened is not None:
                run = opened.group(1)
                if run[0] == "~" or "`" not in opened.group(2):
                    fence_marker = run[0]
                    fence_length = len(run)
                offset += len(line)
                continue
            heading = _H2_LINE.fullmatch(line)
            if heading is not None:
                headings.append((heading.group(1), offset))
            offset += len(line)
        return tuple(
            (heading, (start, headings[index + 1][1] if index + 1 < len(headings) else len(body)))
            for index, (heading, start) in enumerate(headings)
        )

    @classmethod
    def _append_fragment(cls, prefix: str, content: str, *, before_heading: bool) -> str:
        line_ending = cls._line_ending(prefix)
        return (
            cls._separator(prefix, line_ending=line_ending, blank_line=True)
            + content.strip()
            + line_ending
            + (line_ending if before_heading else "")
        )

    @staticmethod
    def _separator(value: str, *, line_ending: str, blank_line: bool) -> str:
        if not value:
            return ""
        required = 2 if blank_line else 1
        trailing = 0
        cursor = len(value)
        while cursor > 0:
            if value[:cursor].endswith("\r\n"):
                trailing += 1
                cursor -= 2
            elif value[:cursor].endswith("\n") or value[:cursor].endswith("\r"):
                trailing += 1
                cursor -= 1
            else:
                break
        return line_ending * max(0, required - trailing)

    @staticmethod
    def _line_ending(value: str) -> str:
        return "\r\n" if "\r\n" in value else "\n"

    @staticmethod
    def _normalized_heading(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    @staticmethod
    def _normalized_block(value: str) -> str:
        return "\n".join(line.rstrip() for line in value.splitlines()).strip()

    @classmethod
    def _route(cls, target: RememberTarget) -> str:
        if target.kind is RememberTargetKind.PROFILE:
            return "profile.md"
        if target.kind is RememberTargetKind.PREFERENCE:
            return "preferences.md"
        if target.kind is RememberTargetKind.OPEN_LOOP:
            return "knowledge/open-loops.md"
        roots = {
            RememberTargetKind.ENTITY: "knowledge/entities",
            RememberTargetKind.TOPIC: "knowledge/topics",
            RememberTargetKind.EPISODE: "knowledge/episodes",
        }
        root = roots[target.kind]
        normalized = cls._normalized_heading(target.subject)
        visible = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE).strip("-._")[:80]
        suffix = stable_hash([target.kind.value, normalized], 12)
        return MemoryDocumentPathPolicy.normalize_relative_path(f"{root}/{visible or 'memory'}-{suffix}.md")


def _heading(value: str) -> str:
    heading = " ".join(str(value).split()).strip()
    if not heading or "\n" in heading or len(heading.encode("utf-8")) > 240:
        raise ValueError("remember target subject is invalid")
    return heading


__all__ = ["ExplicitRememberPlanner", "RememberTarget", "RememberTargetKind"]
