"""运行时装配使用的纯数据配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import MemoryOSConfig, RuntimeMode
from infrastructure.model.config import ModelConfig


@dataclass(frozen=True)
class RetrievalConfig:
    """上下文检索在进程启动时需要的开关。"""

    vectorize_important_session_events: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.vectorize_important_session_events, bool):
            raise TypeError("vectorize_important_session_events must be boolean")


@dataclass(frozen=True)
class RetentionConfig:
    """Catalog 分层保留和批处理配置。"""

    hot_days: int = 7
    warm_days: int = 30
    cold_days: int = 90
    batch_size: int = 100

    def __post_init__(self) -> None:
        for name in ("hot_days", "warm_days", "cold_days", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.hot_days <= self.warm_days <= self.cold_days:
            raise ValueError("retention days must satisfy hot_days <= warm_days <= cold_days")

    def to_mapping(self) -> dict[str, int]:
        return {
            "hot_days": self.hot_days,
            "warm_days": self.warm_days,
            "cold_days": self.cold_days,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class RuntimeConfig(MemoryOSConfig):
    """只描述运行参数，不持有 Store、模型客户端或测试替身。"""

    memory_document_max_bytes: int = 2 * 1024 * 1024
    memory_front_matter_max_bytes: int = 32 * 1024
    memory_front_matter_max_depth: int = 12
    memory_scan_stability_seconds: float = 1.0
    memory_scan_max_files: int = 10_000
    memory_mass_delete_threshold: int = 50
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.tenant_id, str)
            or not self.tenant_id.strip()
            or self.tenant_id in {".", ".."}
            or "/" in self.tenant_id
            or "\\" in self.tenant_id
        ):
            raise ValueError("tenant_id must be one safe non-empty path segment")
        if not isinstance(self.retrieval, RetrievalConfig):
            raise TypeError("retrieval must be a RetrievalConfig")
        if not isinstance(self.retention, RetentionConfig):
            raise TypeError("retention must be a RetentionConfig")
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")
        for field_name in (
            "memory_document_max_bytes",
            "memory_front_matter_max_bytes",
            "memory_front_matter_max_depth",
            "memory_scan_max_files",
            "memory_mass_delete_threshold",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.memory_front_matter_max_bytes >= self.memory_document_max_bytes:
            raise ValueError("memory_front_matter_max_bytes must be smaller than memory_document_max_bytes")
        if self.memory_scan_stability_seconds < 0:
            raise ValueError("memory_scan_stability_seconds cannot be negative")

    @classmethod
    def from_env(
        cls,
        *,
        default_mode: RuntimeMode | str = RuntimeMode.LOCAL,
    ) -> RuntimeConfig:
        """组合公共进程配置、模型配置和 Runtime 默认参数。"""

        common = MemoryOSConfig.from_env(default_mode=default_mode)
        return cls(
            root=common.root,
            mode=common.mode,
            log_level=common.log_level,
            model=ModelConfig.from_env(),
        )


__all__ = ["RetrievalConfig", "RetentionConfig", "RuntimeConfig"]
