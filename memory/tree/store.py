"""严格限定于已确认目录结构的 Markdown 记忆树。"""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from memory.document import (
    MemoryDocument,
    MemoryDocumentCodec,
    MemoryDocumentConfig,
    MemoryDocumentIntegrityError,
    MemoryDocumentLimitError,
)
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.uri import MemoryURI, MemoryURINodeType


class MemoryTreeIntegrityError(ValueError):
    """记忆树包含路径逃逸、符号链接或不符合目录结构的条目。"""


class MemoryTree:
    """安全持久化结构化 L2 文档和可重建的目录语义层。"""

    _STATIC_DIRECTORIES = ("preferences", "entities", "tools", "events", "intentions")
    _MAX_CHILDREN_PER_DIRECTORY = 10_000

    def __init__(
        self,
        root: str | Path,
        *,
        document_codec: MemoryDocumentCodec | None = None,
        document_config: MemoryDocumentConfig | None = None,
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise MemoryTreeIntegrityError("memory tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)
        if document_codec is None:
            from memory.schema import MemorySchemaRegistry

            document_codec = MemoryDocumentCodec(MemorySchemaRegistry.load_default())
        if not isinstance(document_codec, MemoryDocumentCodec):
            raise TypeError("document_codec must be a MemoryDocumentCodec")
        self._document_codec = document_codec
        if document_config is not None and not isinstance(document_config, MemoryDocumentConfig):
            raise TypeError("document_config must be MemoryDocumentConfig")
        self.document_config = document_config or MemoryDocumentConfig()

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

    def directory_path(self, directory: MemoryDirectory) -> Path:
        """把受控目录地址映射到记忆树内的真实目录。"""

        if not isinstance(directory, MemoryDirectory):
            raise TypeError("directory must be a MemoryDirectory")
        candidate = self.root.joinpath(*directory.parts)
        self._require_inside_root(candidate)
        return candidate

    def layer_path(self, directory: MemoryDirectory, level: MemoryLevel) -> Path:
        """返回目录 L0 或 L1 侧车文件的确定性路径。"""

        normalized = MemoryLevel(level)
        return self.directory_path(directory) / normalized.sidecar_filename

    def path_for_uri(self, uri: MemoryURI | str) -> Path:
        """把合法 ``memory://`` 节点确定性映射为树内物理路径。"""

        parsed = MemoryURI.parse(uri)
        if parsed.node_type is MemoryURINodeType.DOCUMENT:
            return self.path_for(parsed.to_address())
        if parsed.node_type is MemoryURINodeType.DIRECTORY:
            return self.directory_path(parsed.to_directory())
        directory, level = parsed.to_layer()
        return self.layer_path(directory, level)

    def write(
        self,
        document: MemoryDocument,
    ) -> MemoryDocument:
        """原子写入已由上层构造的规范 L2；不读取旧记忆或推进版本。"""

        if not isinstance(document, MemoryDocument):
            raise TypeError("document must be a MemoryDocument")
        encoded = self._document_codec.encode(document).encode("utf-8")
        self.document_config.validate_body(document.markdown_body)
        self.document_config.validate_encoded(encoded)
        self.initialize()
        path = self.path_for(document.address)
        self._ensure_directory(path.parent)
        self._atomic_write(path, encoded)
        return document

    def write_layers(
        self,
        directory: MemoryDirectory,
        *,
        abstract: str,
        overview: str,
    ) -> tuple[Path, Path]:
        """在已有目录中先写 L1、再写由它派生的 L0。"""

        if not isinstance(abstract, str) or not isinstance(overview, str):
            raise TypeError("memory semantic layers must be strings")
        if not abstract.strip() or not overview.strip():
            raise ValueError("memory semantic layers must be non-empty")
        directory_path = self.directory_path(directory)
        self._require_directory(directory_path)
        overview_path = self.layer_path(directory, MemoryLevel.OVERVIEW)
        abstract_path = self.layer_path(directory, MemoryLevel.ABSTRACT)
        self._atomic_write(overview_path, overview.encode("utf-8"))
        self._atomic_write(abstract_path, abstract.encode("utf-8"))
        return abstract_path, overview_path

    def read(self, address: MemoryAddress) -> MemoryDocument:
        """读取并完整验证一个结构化 L2 文档。"""

        return self._read_document(address)

    def read_layer(self, directory: MemoryDirectory, level: MemoryLevel) -> str:
        """读取目录 L0 或 L1 的 UTF-8 Markdown 原文。"""

        return self._read_utf8(
            self.layer_path(directory, level),
            label="memory semantic layer",
        )

    def read_layer_bounded(
        self,
        directory: MemoryDirectory,
        level: MemoryLevel,
        *,
        max_bytes: int,
    ) -> str:
        """在读取前校验字节上限，避免加载损坏或异常膨胀的派生层。"""

        return self._read_utf8(
            self.layer_path(directory, level),
            label="memory semantic layer",
            max_bytes=max_bytes,
        )

    def exists(self, address: MemoryAddress) -> bool:
        path = self.path_for(address)
        if path.is_symlink():
            raise MemoryTreeIntegrityError("memory file cannot be a symbolic link")
        if not path.exists():
            return False
        self._require_regular_file(path)
        return True

    def layer_exists(self, directory: MemoryDirectory, level: MemoryLevel) -> bool:
        path = self.layer_path(directory, level)
        if path.is_symlink():
            raise MemoryTreeIntegrityError("memory semantic layer cannot be a symbolic link")
        if not path.exists():
            return False
        self._require_regular_file(path)
        return True

    def directory_exists(self, directory: MemoryDirectory) -> bool:
        path = self.directory_path(directory)
        if path.is_symlink():
            raise MemoryTreeIntegrityError("memory directory cannot be a symbolic link")
        if not path.exists():
            return False
        self._require_directory(path)
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

    def delete_layers(self, directory: MemoryDirectory) -> bool:
        """删除目录的派生 L0/L1，不触碰任何 L2。"""

        changed = False
        directory_path = self.directory_path(directory)
        if not directory_path.exists():
            return False
        self._require_directory(directory_path)
        for level in (MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW):
            path = self.layer_path(directory, level)
            if path.is_symlink():
                raise MemoryTreeIntegrityError("memory semantic layer cannot be a symbolic link")
            if not path.exists():
                continue
            self._require_regular_file(path)
            path.unlink()
            changed = True
        if changed:
            self._fsync_directory(directory_path)
        return changed

    def direct_addresses(
        self,
        directory: MemoryDirectory,
        *,
        limit: int = 1_000,
    ) -> tuple[MemoryAddress, ...]:
        """有界枚举目录中直接存在的 L2 地址，不递归进入子目录。"""

        maximum = self._directory_limit(limit)
        path = self.directory_path(directory)
        if not path.exists():
            return ()
        parts = directory.parts
        addresses: tuple[MemoryAddress, ...]
        if not parts:
            children = self._content_children(path)
            allowed_directories = set(self._STATIC_DIRECTORIES)
            for child in children:
                if child.name == "profile.md" and child.is_file():
                    continue
                if child.name in allowed_directories and child.is_dir():
                    continue
                raise MemoryTreeIntegrityError("memory root contains an unsupported entry")
            profile = self.root / "profile.md"
            addresses = (MemoryAddress.profile(),) if profile.exists() else ()
        elif parts == ("preferences",):
            addresses = tuple(
                MemoryAddress.preference(name) for name in self._markdown_names(path)
            )
        elif parts[0] == "entities" and len(parts) == 2:
            addresses = tuple(
                MemoryAddress.entity(parts[1], name) for name in self._markdown_names(path)
            )
        elif parts == ("tools",):
            addresses = tuple(MemoryAddress.tool(name) for name in self._markdown_names(path))
        elif parts[0] == "events" and len(parts) == 4:
            event_date = date(int(parts[1]), int(parts[2]), int(parts[3]))
            addresses = tuple(
                MemoryAddress.event(event_date, name) for name in self._markdown_names(path)
            )
        elif parts == ("intentions",):
            addresses = tuple(
                MemoryAddress.intention(name) for name in self._markdown_names(path)
            )
        else:
            if any(child.is_file() for child in self._content_children(path)):
                raise MemoryTreeIntegrityError("memory branch directory cannot contain L2 files")
            addresses = ()
        if len(addresses) > maximum:
            raise MemoryTreeIntegrityError("memory directory exceeded its direct L2 bound")
        return addresses

    def child_directories(
        self,
        directory: MemoryDirectory,
        *,
        limit: int = 1_000,
    ) -> tuple[MemoryDirectory, ...]:
        """有界枚举目录的直接子目录，不读取更深层内容。"""

        maximum = self._directory_limit(limit)
        path = self.directory_path(directory)
        if not path.exists():
            return ()
        parts = directory.parts
        if not parts:
            children = tuple(MemoryDirectory((name,)) for name in self._STATIC_DIRECTORIES)
        elif parts == ("entities",):
            children = tuple(
                MemoryDirectory.entities(child.name) for child in self._directories(path)
            )
        elif parts and parts[0] == "events" and len(parts) < 4:
            children = tuple(
                MemoryDirectory((*parts, child.name)) for child in self._directories(path)
            )
        else:
            if any(child.is_dir() for child in self._content_children(path)):
                raise MemoryTreeIntegrityError("memory leaf directory cannot contain subdirectories")
            children = ()
        existing = tuple(child for child in children if self.directory_exists(child))
        if len(existing) > maximum:
            raise MemoryTreeIntegrityError("memory directory exceeded its child directory bound")
        return existing

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
        for child in self._content_children(directory):
            if not child.is_file() or child.suffix != ".md" or not child.stem:
                raise MemoryTreeIntegrityError("memory leaf directory may contain only Markdown files")
            names.append(child.stem)
        return tuple(names)

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        children = self._content_children(directory)
        if any(not child.is_dir() for child in children):
            raise MemoryTreeIntegrityError("memory branch may contain only directories")
        return children

    def _content_children(self, directory: Path) -> tuple[Path, ...]:
        content: list[Path] = []
        semantic_names = {
            MemoryLevel.ABSTRACT.sidecar_filename,
            MemoryLevel.OVERVIEW.sidecar_filename,
        }
        for child in self._children(directory):
            if child.name in semantic_names:
                self._require_regular_file(child)
                continue
            if child.name.startswith("."):
                raise MemoryTreeIntegrityError("memory directory contains an unsupported hidden entry")
            content.append(child)
        return tuple(content)

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

    def _read_utf8(
        self,
        path: Path,
        *,
        label: str,
        max_bytes: int | None = None,
    ) -> str:
        self._require_regular_file(path)
        if max_bytes is not None:
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
                raise ValueError("max_bytes must be a positive integer")
            if path.stat().st_size > max_bytes:
                raise MemoryTreeIntegrityError(f"{label} exceeds its configured read bound")
        try:
            payload = path.read_bytes()
            if max_bytes is not None and len(payload) > max_bytes:
                raise MemoryTreeIntegrityError(f"{label} exceeds its configured read bound")
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryTreeIntegrityError(f"{label} is not valid UTF-8") from exc

    def _read_document(
        self,
        address: MemoryAddress,
    ) -> MemoryDocument:
        raw = self._read_utf8(
            self.path_for(address),
            label="memory document",
            max_bytes=self.document_config.max_encoded_bytes,
        )
        try:
            document = self._document_codec.decode(raw, expected_address=address)
            self.document_config.validate_body(document.markdown_body)
            return document
        except (MemoryDocumentIntegrityError, MemoryDocumentLimitError) as exc:
            raise MemoryTreeIntegrityError("memory L2 document failed integrity validation") from exc

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
            if self._content_children(current):
                break
            for filename in (
                MemoryLevel.ABSTRACT.sidecar_filename,
                MemoryLevel.OVERVIEW.sidecar_filename,
            ):
                sidecar = current / filename
                if sidecar.exists():
                    self._require_regular_file(sidecar)
                    sidecar.unlink()
            self._fsync_directory(current)
            try:
                current.rmdir()
            except OSError:
                break
            self._fsync_directory(current.parent)
            current = current.parent

    @staticmethod
    def _directory_limit(limit: int) -> int:
        maximum = int(limit)
        if isinstance(limit, bool) or maximum <= 0 or maximum > 10_000:
            raise ValueError("memory directory limit must be between 1 and 10000")
        return maximum

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["MemoryTree", "MemoryTreeIntegrityError"]
