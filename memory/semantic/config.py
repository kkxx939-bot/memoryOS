"""记忆目录语义层的显式边界配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySemanticConfig:
    """限制单次目录刷新和全树重建的资源使用。"""

    max_direct_entries: int = 256
    max_rebuild_directories: int = 10_000
    max_prompt_chars: int = 120_000
    max_entry_summary_chars: int = 1_000
    max_directory_summary_chars: int = 2_000
    max_overview_chars: int = 64_000
    max_abstract_chars: int = 800
    lock_ttl_seconds: int = 120
    stale_retries: int = 1

    def __post_init__(self) -> None:
        positive = {
            "max_direct_entries": self.max_direct_entries,
            "max_rebuild_directories": self.max_rebuild_directories,
            "max_prompt_chars": self.max_prompt_chars,
            "max_entry_summary_chars": self.max_entry_summary_chars,
            "max_directory_summary_chars": self.max_directory_summary_chars,
            "max_overview_chars": self.max_overview_chars,
            "max_abstract_chars": self.max_abstract_chars,
            "lock_ttl_seconds": self.lock_ttl_seconds,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.stale_retries, bool)
            or not isinstance(self.stale_retries, int)
            or not 0 <= self.stale_retries <= 5
        ):
            raise ValueError("stale_retries must be between zero and five")
        if self.max_abstract_chars > self.max_overview_chars:
            raise ValueError("max_abstract_chars cannot exceed max_overview_chars")


__all__ = ["MemorySemanticConfig"]
