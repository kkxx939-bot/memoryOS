"""上下文数据库里的上下文URI。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from memoryos.core.errors import InvalidContextURI


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
        parsed = urlparse(str(value))
        if parsed.scheme != "memoryos" or not parsed.netloc:
            raise InvalidContextURI(f"Expected memoryos:// URI, got {value!r}")
        segments = tuple(segment for segment in parsed.path.split("/") if segment)
        for segment in segments:
            _validate_segment(segment)
        authority = parsed.netloc
        if authority == "user":
            if len(segments) < 2:
                raise InvalidContextURI("User URI must be memoryos://user/{user_id}/...")
            _validate_segment(segments[0])
        elif authority in {"resources", "skills"}:
            if not segments:
                raise InvalidContextURI(f"{authority} URI must include a path")
        else:
            raise InvalidContextURI(f"Unsupported URI authority: {authority}")
        return cls(raw=str(value), authority=authority, segments=segments)

    @property
    def user_id(self) -> str | None:
        return self.segments[0] if self.authority == "user" else None

    def child(self, *segments: str) -> ContextURI:
        for segment in segments:
            _validate_segment(segment)
        path = "/".join((*self.segments, *segments))
        return ContextURI.parse(f"memoryos://{self.authority}/{path}")

    def to_source_path(self, root: Path, tenant_id: str = "default") -> Path:
        base = Path(root)
        if self.authority == "user":
            user_id = self.segments[0]
            relative = PurePosixPath("tenants") / tenant_id / "users" / user_id / PurePosixPath(*self.segments[1:])
        elif self.authority == "resources":
            relative = PurePosixPath("resources") / PurePosixPath(*self.segments)
        elif self.authority == "skills":
            relative = PurePosixPath("skills") / PurePosixPath(*self.segments)
        else:
            raise InvalidContextURI(f"Unsupported URI authority: {self.authority}")
        path = (base / Path(relative)).resolve()
        root_resolved = base.resolve()
        if root_resolved not in path.parents and path != root_resolved:
            raise InvalidContextURI(f"URI escapes root: {self.raw}")
        return path

    def __str__(self) -> str:
        return self.raw
