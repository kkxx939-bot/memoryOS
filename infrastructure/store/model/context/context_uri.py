"""上下文 URI 的解析、校验与构造规则。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from infrastructure.store.model.context.errors import InvalidContextURI


def _validate_segment(segment: str) -> None:
    if not segment or segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise InvalidContextURI(f"Unsafe URI segment: {segment!r}")


@dataclass(frozen=True)
class ContextURI:
    raw: str
    authority: str
    segments: tuple[str, ...]

    @classmethod
    def parse(cls, value: str) -> ContextURI:
        raw = str(value)
        try:
            parsed = urlsplit(raw)
            username = parsed.username
            password = parsed.password
            port = parsed.port
        except ValueError as exc:
            raise InvalidContextURI("ContextURI authority is malformed") from exc
        if parsed.scheme != "memoryos" or not parsed.netloc:
            raise InvalidContextURI(f"Expected memoryos:// URI, got {value!r}")
        if parsed.query or parsed.fragment:
            raise InvalidContextURI("ContextURI query and fragment components are forbidden")
        if username or password or port is not None:
            raise InvalidContextURI("ContextURI authority must not contain credentials or a port")
        if not parsed.path.startswith("/"):
            raise InvalidContextURI("ContextURI path must be absolute")
        raw_segments = parsed.path[1:].split("/")
        if raw_segments and raw_segments[-1] == "":
            raw_segments.pop()
        if not raw_segments or any(segment == "" for segment in raw_segments):
            raise InvalidContextURI("ContextURI contains an empty path segment")
        segments_list: list[str] = []
        encoded_segments: list[str] = []
        for raw_segment in raw_segments:
            if _has_invalid_percent_escape(raw_segment):
                raise InvalidContextURI("ContextURI contains invalid percent encoding")
            segment = unquote(raw_segment, errors="strict")
            _validate_segment(segment)
            segments_list.append(segment)
            encoded_segments.append(quote(segment, safe="-._~"))
        segments = tuple(segments_list)
        for segment in segments:
            _validate_segment(segment)
        authority = parsed.netloc.casefold()
        if authority == "user":
            if len(segments) < 2:
                raise InvalidContextURI("User URI must be memoryos://user/{user_id}/...")
            _validate_segment(segments[0])
        elif authority in {"resources", "skills"}:
            if not segments:
                raise InvalidContextURI(f"{authority} URI must include a path")
        else:
            raise InvalidContextURI(f"Unsupported URI authority: {authority}")
        canonical = f"memoryos://{authority}/{'/'.join(encoded_segments)}"
        return cls(raw=canonical, authority=authority, segments=segments)

    @property
    def user_id(self) -> str | None:
        return self.segments[0] if self.authority == "user" else None

    def child(self, *segments: str) -> ContextURI:
        for segment in segments:
            _validate_segment(segment)
        path = "/".join((*self.segments, *segments))
        return ContextURI.parse(f"memoryos://{self.authority}/{path}")

    def to_source_path(self, root: Path, tenant_id: str = "default") -> Path:
        root_resolved = Path(root).resolve()
        if self.authority == "user":
            user_id = self.segments[0]
            relative = PurePosixPath("tenants") / tenant_id / "users" / user_id / PurePosixPath(*self.segments[1:])
        elif self.authority == "resources":
            relative = PurePosixPath("resources") / PurePosixPath(*self.segments)
        elif self.authority == "skills":
            relative = PurePosixPath("skills") / PurePosixPath(*self.segments)
        else:
            raise InvalidContextURI(f"Unsupported URI authority: {self.authority}")
        lexical_path = root_resolved / Path(relative)
        if root_resolved not in lexical_path.parents and lexical_path != root_resolved:
            raise InvalidContextURI(f"URI escapes root: {self.raw}")
        resolved_path = lexical_path.resolve()
        if resolved_path != lexical_path:
            # 仅解析根内别名并不安全：租户路径仍可能被重定向到共享根下的其他租户。
            # 耐久存储必须使用由 URI 推导出的精确词法命名空间。
            raise InvalidContextURI(f"URI path traverses a symbolic link: {self.raw}")
        return lexical_path

    def __str__(self) -> str:
        return self.raw


def _has_invalid_percent_escape(value: str) -> bool:
    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        if index + 2 >= len(value) or any(
            character not in "0123456789abcdefABCDEF" for character in value[index + 1 : index + 3]
        ):
            return True
        index += 3
    return False
