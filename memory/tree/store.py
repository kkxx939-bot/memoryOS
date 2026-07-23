"""严格限定于已确认目录结构的 Markdown 记忆树。"""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from memory.tree.model import MemoryAddress, MemoryKind


class MemoryTreeIntegrityError(ValueError):
    """记忆树包含路径逃逸、符号链接或不符合目录结构的条目。"""


class MemoryTree:
    """解析并持久化记忆树中的 Markdown 原文，不解释正文语义。"""

    _STATIC_DIRECTORIES = ("preferences", "entities", "tools", "events", "intentions")
    _MAX_CHILDREN_PER_DIRECTORY = 10_000

    def __init__(self, root: str | Path) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise MemoryTreeIntegrityError("memory tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)

    def initialize(self) -> Path:
        """只创建静态目录；没有真实内容时不创建空的 profile.md。"""

        self._ensure_directory(self.root)
        for name in self._STATIC_DIRECTORIES:
            self._ensure_directory(self.root / name)
        return self.root

    def path_for(self, address: MemoryAddress) -> Path:
        """把经过验证的语义地址确定性映射为唯一 Markdown 路径。"""

        if not isinstance(address, MemoryAddress):
            raise TypeError("address must be a MemoryAddress")
        relative = self._relative_path(address)
        candidate = self.root / relative
        self._require_inside_root(candidate)
        return candidate

    def write(self, address: MemoryAddress, markdown: str) -> Path:
        """原子写入 Markdown 原文；更新同一地址时替换完整文件。"""

        if not isinstance(markdown, str):
            raise TypeError("memory markdown must be a string")
        self.initialize()
        path = self.path_for(address)
        self._ensure_directory(path.parent)
        self._atomic_write(path, markdown.encode("utf-8"))
        return path

    def read(self, address: MemoryAddress) -> str:
        """逐字节读取一个普通 Markdown 文件并按 UTF-8 解码。"""

        path = self.path_for(address)
        self._require_regular_file(path)
        try:
            return path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryTreeIntegrityError("memory file is not valid UTF-8") from exc

    def exists(self, address: MemoryAddress) -> bool:
        path = self.path_for(address)
        if path.is_symlink():
            raise MemoryTreeIntegrityError("memory file cannot be a symbolic link")
        if not path.exists():
            return False
        self._require_regular_file(path)
        return True

    def delete(self, address: MemoryAddress) -> bool:
        """删除一个记忆文件，并只清理其下方已经为空的动态目录。"""

        path = self.path_for(address)
        if path.is_symlink():
            raise MemoryTreeIntegrityError("memory file cannot be a symbolic link")
        if not path.exists():
            return False
        self._require_regular_file(path)
        path.unlink()
        self._fsync_directory(path.parent)
        self._prune_dynamic_directories(address, path.parent)
        return True

    def list_addresses(
        self,
        kind: MemoryKind | None = None,
        *,
        limit: int = 256,
    ) -> tuple[MemoryAddress, ...]:
        """按固定类型顺序和路径字典序有界枚举记忆地址。"""

        maximum = int(limit)
        if maximum <= 0 or maximum > 10_000:
            raise ValueError("memory tree list limit must be between 1 and 10000")
        if not self.root.exists():
            return ()
        self._require_directory(self.root)
        kinds = (MemoryKind(kind),) if kind is not None else tuple(MemoryKind)
        result: list[MemoryAddress] = []
        for selected in kinds:
            for address in self._iter_kind(selected):
                result.append(address)
                if len(result) >= maximum:
                    return tuple(result)
        return tuple(result)

    @staticmethod
    def _relative_path(address: MemoryAddress) -> Path:
        if address.kind is MemoryKind.PROFILE:
            return Path("profile.md")
        if address.kind is MemoryKind.PREFERENCE:
            return Path("preferences", f"{address.name}.md")
        if address.kind is MemoryKind.ENTITY:
            return Path("entities", address.category, f"{address.name}.md")
        if address.kind is MemoryKind.TOOL:
            return Path("tools", f"{address.name}.md")
        if address.kind is MemoryKind.EVENT:
            assert address.event_date is not None
            return Path(
                "events",
                f"{address.event_date.year:04d}",
                f"{address.event_date.month:02d}",
                f"{address.event_date.day:02d}",
                f"{address.name}.md",
            )
        return Path("intentions", f"{address.name}.md")

    def _iter_kind(self, kind: MemoryKind) -> Iterator[MemoryAddress]:
        if kind is MemoryKind.PROFILE:
            path = self.root / "profile.md"
            if path.is_symlink():
                raise MemoryTreeIntegrityError("profile.md cannot be a symbolic link")
            if path.exists():
                self._require_regular_file(path)
                yield MemoryAddress.profile()
            return
        if kind is MemoryKind.PREFERENCE:
            yield from (
                MemoryAddress.preference(name)
                for name in self._markdown_names(self.root / "preferences")
            )
            return
        if kind is MemoryKind.ENTITY:
            for category_path in self._directories(self.root / "entities"):
                for name in self._markdown_names(category_path):
                    yield MemoryAddress.entity(category_path.name, name)
            return
        if kind is MemoryKind.TOOL:
            yield from (MemoryAddress.tool(name) for name in self._markdown_names(self.root / "tools"))
            return
        if kind is MemoryKind.EVENT:
            yield from self._iter_events()
            return
        yield from (MemoryAddress.intention(name) for name in self._markdown_names(self.root / "intentions"))

    def _iter_events(self) -> Iterator[MemoryAddress]:
        for year_path in self._directories(self.root / "events"):
            if len(year_path.name) != 4 or not year_path.name.isdigit():
                raise MemoryTreeIntegrityError("event year directory must use YYYY")
            for month_path in self._directories(year_path):
                if len(month_path.name) != 2 or not month_path.name.isdigit():
                    raise MemoryTreeIntegrityError("event month directory must use MM")
                for day_path in self._directories(month_path):
                    if len(day_path.name) != 2 or not day_path.name.isdigit():
                        raise MemoryTreeIntegrityError("event day directory must use DD")
                    try:
                        event_date = date(int(year_path.name), int(month_path.name), int(day_path.name))
                    except ValueError as exc:
                        raise MemoryTreeIntegrityError("event directory contains an invalid calendar date") from exc
                    for name in self._markdown_names(day_path):
                        yield MemoryAddress.event(event_date, name)

    def _markdown_names(self, directory: Path) -> tuple[str, ...]:
        names: list[str] = []
        for child in self._children(directory):
            if not child.is_file() or child.suffix.casefold() != ".md" or not child.stem:
                raise MemoryTreeIntegrityError("memory leaf directory may contain only Markdown files")
            names.append(child.stem)
        return tuple(names)

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        children = self._children(directory)
        if any(not child.is_dir() for child in children):
            raise MemoryTreeIntegrityError("memory branch may contain only directories")
        return children

    def _children(self, directory: Path) -> tuple[Path, ...]:
        if not directory.exists():
            return ()
        self._require_directory(directory)
        children: list[Path] = []
        for child in directory.iterdir():
            if child.is_symlink():
                raise MemoryTreeIntegrityError("memory tree cannot contain symbolic links")
            children.append(child)
            if len(children) > self._MAX_CHILDREN_PER_DIRECTORY:
                raise MemoryTreeIntegrityError("memory directory exceeded its enumeration bound")
        return tuple(sorted(children, key=lambda item: item.name))

    def _ensure_directory(self, directory: Path) -> None:
        self._require_inside_root(directory)
        relative = directory.relative_to(self.root)
        paths = (
            self.root,
            *(self.root / Path(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)),
        )
        for current in paths:
            if current.is_symlink():
                raise MemoryTreeIntegrityError("memory directory cannot be a symbolic link")
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            if not current.is_dir():
                raise MemoryTreeIntegrityError("memory directory path is not a directory")
            try:
                current.chmod(0o700)
            except OSError:
                pass

    def _require_inside_root(self, path: Path) -> None:
        candidate = path.resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise MemoryTreeIntegrityError("memory path escapes its tree root")

    @staticmethod
    def _require_regular_file(path: Path) -> None:
        if path.is_symlink():
            raise MemoryTreeIntegrityError("memory file cannot be a symbolic link")
        try:
            metadata = path.stat()
        except FileNotFoundError:
            raise
        if not stat.S_ISREG(metadata.st_mode):
            raise MemoryTreeIntegrityError("memory path is not a regular file")

    @staticmethod
    def _require_directory(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise MemoryTreeIntegrityError("memory path is not a safe directory")

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        if path.is_symlink():
            raise MemoryTreeIntegrityError("memory file cannot be a symbolic link")
        if path.exists():
            self._require_regular_file(path)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise MemoryTreeIntegrityError("memory file cannot be a symbolic link")
            if path.exists():
                self._require_regular_file(path)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _prune_dynamic_directories(self, address: MemoryAddress, parent: Path) -> None:
        stop = {
            MemoryKind.ENTITY: self.root / "entities",
            MemoryKind.EVENT: self.root / "events",
        }.get(address.kind)
        if stop is None:
            return
        current = parent
        while current != stop:
            try:
                current.rmdir()
            except OSError:
                break
            self._fsync_directory(current.parent)
            current = current.parent

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["MemoryTree", "MemoryTreeIntegrityError"]
