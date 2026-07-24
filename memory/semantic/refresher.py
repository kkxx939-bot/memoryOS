"""按照目录依赖关系自下向上刷新可重建的 L0/L1。"""

from __future__ import annotations

import hashlib

from infrastructure.store.contracts.path_lock import PathLock
from memory.model import MemoryAddress, MemoryDirectory, MemoryLevel
from memory.semantic.config import MemorySemanticConfig
from memory.semantic.generator import MemoryOverviewGenerator
from memory.semantic.model import (
    MemoryDirectorySnapshot,
    MemorySemanticEntry,
    MemorySemanticEntryKind,
    MemorySemanticRefreshResult,
    MemorySemanticRefreshStatus,
)
from memory.tree.store import MemoryTree


class MemorySemanticRefreshError(RuntimeError):
    """目录快照持续变化或生成结果无法安全发布。"""


class MemorySemanticRefresher:
    """读取 L2 和子目录 L0，生成 L1 后确定性派生 L0。"""

    def __init__(
        self,
        tree: MemoryTree,
        generator: MemoryOverviewGenerator,
        path_lock: PathLock,
        *,
        config: MemorySemanticConfig | None = None,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be a MemoryTree")
        if not callable(getattr(generator, "generate", None)):
            raise TypeError("generator must implement generate(snapshot)")
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be a PathLock")
        self.tree = tree
        self.generator = generator
        self.path_lock = path_lock
        self.config = config or MemorySemanticConfig()
        root_identity = hashlib.sha256(str(self.tree.root).encode("utf-8")).hexdigest()[:24]
        self._lock_prefix = f"memory-semantic:{root_identity}"

    def refresh_for(self, address: MemoryAddress) -> tuple[MemorySemanticRefreshResult, ...]:
        """从发生变化的 L2 父目录开始，依次刷新到记忆根目录。"""

        if not isinstance(address, MemoryAddress):
            raise TypeError("address must be a MemoryAddress")
        self.tree.initialize()
        return tuple(
            self.refresh_directory(directory)
            for directory in MemoryDirectory.for_address(address).lineage()
        )

    def refresh_directory(self, directory: MemoryDirectory) -> MemorySemanticRefreshResult:
        """基于稳定快照刷新一个目录；生成期间不长期占用目录锁。"""

        if not isinstance(directory, MemoryDirectory):
            raise TypeError("directory must be a MemoryDirectory")
        if not self.tree.directory_exists(directory):
            return MemorySemanticRefreshResult(
                directory=directory,
                status=MemorySemanticRefreshStatus.MISSING,
            )

        attempts = self.config.stale_retries + 1
        for _ in range(attempts):
            snapshot = self._locked_snapshot(directory)
            if not snapshot.entries:
                empty_result = self._delete_empty_layers(snapshot)
                if empty_result is not None:
                    return empty_result
                continue

            overview = self.generator.generate(snapshot)
            if not isinstance(overview, str) or not overview.strip():
                raise MemorySemanticRefreshError("memory overview generator returned empty text")
            if len(overview) > self.config.max_overview_chars:
                raise MemorySemanticRefreshError("memory overview exceeds its configured bound")
            normalized_overview = overview.strip() + "\n"
            abstract = self._abstract_from_overview(normalized_overview)

            with self.path_lock.acquire(
                self._lock_key(directory),
                ttl_seconds=self.config.lock_ttl_seconds,
            ) as guard:
                with guard.fenced():
                    if not self.tree.directory_exists(directory):
                        return MemorySemanticRefreshResult(
                            directory=directory,
                            status=MemorySemanticRefreshStatus.MISSING,
                        )
                    current = self._snapshot(directory)
                    if current.digest != snapshot.digest:
                        continue
                    if self._layers_equal(directory, abstract, normalized_overview):
                        return self._result(
                            directory,
                            MemorySemanticRefreshStatus.UNCHANGED,
                            snapshot.digest,
                        )
                    self.tree.write_layers(
                        directory,
                        abstract=abstract,
                        overview=normalized_overview,
                    )
                    return self._result(
                        directory,
                        MemorySemanticRefreshStatus.WRITTEN,
                        snapshot.digest,
                    )
        raise MemorySemanticRefreshError(
            "memory directory changed repeatedly while semantic layers were generated"
        )

    def rebuild(
        self,
        directory: MemoryDirectory | None = None,
    ) -> tuple[MemorySemanticRefreshResult, ...]:
        """有界遍历指定子树，并按照最深目录优先的顺序完整重建。"""

        self.tree.initialize()
        root = directory or MemoryDirectory.root()
        if not isinstance(root, MemoryDirectory):
            raise TypeError("directory must be a MemoryDirectory")
        if not self.tree.directory_exists(root):
            return ()

        pending = [root]
        discovered: list[MemoryDirectory] = []
        while pending:
            current = pending.pop()
            discovered.append(current)
            if len(discovered) > self.config.max_rebuild_directories:
                raise MemorySemanticRefreshError(
                    "memory semantic rebuild exceeded its configured directory bound"
                )
            children = self.tree.child_directories(
                current,
                limit=self.config.max_direct_entries,
            )
            pending.extend(reversed(children))
        return tuple(self.refresh_directory(current) for current in reversed(discovered))

    def _locked_snapshot(self, directory: MemoryDirectory) -> MemoryDirectorySnapshot:
        with self.path_lock.acquire(
            self._lock_key(directory),
            ttl_seconds=self.config.lock_ttl_seconds,
        ) as guard:
            with guard.fenced():
                return self._snapshot(directory)

    def _snapshot(self, directory: MemoryDirectory) -> MemoryDirectorySnapshot:
        entries: list[MemorySemanticEntry] = []
        addresses = self.tree.direct_addresses(
            directory,
            limit=self.config.max_direct_entries,
        )
        for address in addresses:
            document = self.tree.read(address)
            content = document.markdown_body
            entries.append(
                MemorySemanticEntry(
                    name=self.tree.path_for(address).name,
                    kind=MemorySemanticEntryKind.MEMORY,
                    content=content,
                )
            )

        children = self.tree.child_directories(
            directory,
            limit=self.config.max_direct_entries,
        )
        for child in children:
            child_summary = self._child_summary(child)
            if child_summary is None:
                continue
            entries.append(
                MemorySemanticEntry(
                    name=child.parts[-1],
                    kind=MemorySemanticEntryKind.DIRECTORY,
                    content=child_summary,
                )
            )
        if len(entries) > self.config.max_direct_entries:
            raise MemorySemanticRefreshError(
                "memory directory exceeded its configured semantic entry bound"
            )
        return MemoryDirectorySnapshot(directory, tuple(entries))

    def _child_summary(self, child: MemoryDirectory) -> str | None:
        if self.tree.layer_exists(child, MemoryLevel.ABSTRACT):
            abstract = self.tree.read_layer_bounded(
                child,
                MemoryLevel.ABSTRACT,
                max_bytes=self.config.max_abstract_chars * 4 + 4,
            )
            if len(abstract) > self.config.max_abstract_chars + 1:
                raise MemorySemanticRefreshError(
                    "child memory abstract exceeds its configured character bound"
                )
            return abstract
        if self.tree.layer_exists(child, MemoryLevel.OVERVIEW):
            overview = self.tree.read_layer_bounded(
                child,
                MemoryLevel.OVERVIEW,
                max_bytes=self.config.max_overview_chars * 4 + 4,
            )
            if len(overview) > self.config.max_overview_chars + 1:
                raise MemorySemanticRefreshError(
                    "child memory overview exceeds its configured character bound"
                )
            return self._abstract_from_overview(overview)
        return None

    def _delete_empty_layers(
        self,
        snapshot: MemoryDirectorySnapshot,
    ) -> MemorySemanticRefreshResult | None:
        directory = snapshot.directory
        stale = False
        with self.path_lock.acquire(
            self._lock_key(directory),
            ttl_seconds=self.config.lock_ttl_seconds,
        ) as guard:
            with guard.fenced():
                if not self.tree.directory_exists(directory):
                    return MemorySemanticRefreshResult(
                        directory=directory,
                        status=MemorySemanticRefreshStatus.MISSING,
                    )
                current = self._snapshot(directory)
                if current.digest != snapshot.digest:
                    stale = True
                else:
                    deleted = self.tree.delete_layers(directory)
                    return MemorySemanticRefreshResult(
                        directory=directory,
                        status=(
                            MemorySemanticRefreshStatus.DELETED
                            if deleted
                            else MemorySemanticRefreshStatus.UNCHANGED
                        ),
                        source_digest=snapshot.digest,
                    )
        if stale:
            return None
        raise AssertionError("empty memory semantic refresh ended without a result")

    def _abstract_from_overview(self, overview: str) -> str:
        lines = overview.splitlines()
        content: list[str] = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                if not stripped or stripped.startswith("#"):
                    continue
                started = True
            if stripped.startswith("##"):
                break
            if stripped:
                content.append(stripped)
        compact = " ".join(content)
        if not compact:
            compact = " ".join(
                line.strip()
                for line in lines
                if line.strip() and not line.lstrip().startswith("#")
            )
        if not compact:
            raise MemorySemanticRefreshError("memory overview cannot produce a non-empty abstract")
        return self._truncate_abstract(compact) + "\n"

    def _truncate_abstract(self, text: str) -> str:
        maximum = self.config.max_abstract_chars
        if len(text) <= maximum:
            return text
        if maximum <= 3:
            return text[:maximum]
        candidate = text[: maximum - 3].rstrip()
        for marker in ("。", "！", "？", ". ", "! ", "? "):
            boundary = candidate.rfind(marker)
            if boundary >= maximum // 2:
                return candidate[: boundary + len(marker)].rstrip()
        return candidate + "..."

    def _layers_equal(
        self,
        directory: MemoryDirectory,
        abstract: str,
        overview: str,
    ) -> bool:
        if not self.tree.layer_exists(directory, MemoryLevel.ABSTRACT):
            return False
        if not self.tree.layer_exists(directory, MemoryLevel.OVERVIEW):
            return False
        return (
            self.tree.read_layer_bounded(
                directory,
                MemoryLevel.ABSTRACT,
                max_bytes=self.config.max_abstract_chars * 4 + 4,
            )
            == abstract
            and self.tree.read_layer_bounded(
                directory,
                MemoryLevel.OVERVIEW,
                max_bytes=self.config.max_overview_chars * 4 + 4,
            )
            == overview
        )

    def _result(
        self,
        directory: MemoryDirectory,
        status: MemorySemanticRefreshStatus,
        digest: str,
    ) -> MemorySemanticRefreshResult:
        return MemorySemanticRefreshResult(
            directory=directory,
            status=status,
            source_digest=digest,
            abstract_path=self.tree.layer_path(directory, MemoryLevel.ABSTRACT),
            overview_path=self.tree.layer_path(directory, MemoryLevel.OVERVIEW),
        )

    def _lock_key(self, directory: MemoryDirectory) -> str:
        suffix = "/".join(directory.parts) or "/"
        return f"{self._lock_prefix}:{suffix}"


__all__ = [
    "MemorySemanticRefreshError",
    "MemorySemanticRefresher",
]
