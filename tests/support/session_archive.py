"""SessionArchive 测试使用的显式证据编码器。"""

from __future__ import annotations

from infrastructure.store.contracts.session_evidence import SessionEvidenceEncoder
from infrastructure.store.filesystem.session_archive import SessionArchiveStore
from memory.commit.evidence.archive_encoder import SessionEvidenceArchiveEncoder


def session_evidence_encoder() -> SessionEvidenceArchiveEncoder:
    """为每个 Store 返回独立、无全局状态的领域编码器。"""

    return SessionEvidenceArchiveEncoder()


def build_session_archive_store(
    root,  # noqa: ANN001
    tenant_id: str = "default",
    *,
    evidence_encoder: SessionEvidenceEncoder | None = None,
    test_hook=None,  # noqa: ANN001
) -> SessionArchiveStore:
    """构造显式注入 Encoder 的真实文件 SessionArchiveStore。"""

    return SessionArchiveStore(
        root,
        tenant_id=tenant_id,
        evidence_encoder=evidence_encoder or session_evidence_encoder(),
        test_hook=test_hook,
    )


__all__ = ["build_session_archive_store", "session_evidence_encoder"]
