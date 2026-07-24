"""长期记忆 L2 文档模型、边界与规范编解码入口。"""

from memory.document.codec import MemoryDocumentCodec, MemoryDocumentIntegrityError
from memory.document.config import MemoryDocumentConfig, MemoryDocumentLimitError
from memory.document.model import MemoryDocument, MemoryDocumentMetadata

__all__ = [
    "MemoryDocument",
    "MemoryDocumentCodec",
    "MemoryDocumentConfig",
    "MemoryDocumentIntegrityError",
    "MemoryDocumentLimitError",
    "MemoryDocumentMetadata",
]
