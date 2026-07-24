"""长期记忆 L2 文档的统一物理边界。"""

from __future__ import annotations

from dataclasses import dataclass


class MemoryDocumentLimitError(ValueError):
    """L2 正文或完整物理文件超过显式配置边界。"""


@dataclass(frozen=True)
class MemoryDocumentConfig:
    """由记忆树写入和语义读取共同使用的 L2 大小限制。"""

    max_markdown_body_chars: int = 6_000
    max_encoded_bytes: int = 128_000

    def __post_init__(self) -> None:
        for name, value in {
            "max_markdown_body_chars": self.max_markdown_body_chars,
            "max_encoded_bytes": self.max_encoded_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def validate_body(self, markdown_body: str) -> None:
        if not isinstance(markdown_body, str):
            raise TypeError("memory Markdown body must be a string")
        if len(markdown_body) > self.max_markdown_body_chars:
            raise MemoryDocumentLimitError("memory Markdown body exceeds its configured limit")

    def validate_encoded(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("encoded memory document must be bytes")
        if len(payload) > self.max_encoded_bytes:
            raise MemoryDocumentLimitError("encoded memory document exceeds its configured limit")


__all__ = ["MemoryDocumentConfig", "MemoryDocumentLimitError"]
