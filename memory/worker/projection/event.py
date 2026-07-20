"""校验 Markdown Memory 投影队列事件并提取耐久身份。"""

from __future__ import annotations

from collections.abc import Mapping

from foundation.identity import require_internal_job_namespace
from infrastructure.store.contracts.queue import QueueJob
from infrastructure.store.memory.control_store import deletion_event_digest
from memory.core.model import DocumentEditKind
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.worker.projection.model import coerce_persisted_int

_PROJECTION_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "tenant_id",
        "owner_user_id",
        "document_id",
        "intent_id",
        "event_id",
        "edit_kind",
        "old_relative_path",
        "new_relative_path",
        "before_raw_digest",
        "after_raw_digest",
        "logical_revision",
        "projection_generation",
    }
)


def parse_projection_job(job: QueueJob) -> dict[str, object]:
    """验证队列动作、固定命名空间、文档 URI 和 generation。"""

    if job.queue_name != "memory_projection" or job.action != "memory_committed":
        raise ValueError("job is not a Markdown memory projection event")
    payload: dict[str, object] = dict(job.payload)
    if (
        frozenset(payload) != _PROJECTION_PAYLOAD_KEYS
        or payload.get("schema") != "memory_document_projection_v1"
    ):
        raise ValueError("memory projection queue payload schema is invalid")
    tenant = require_internal_job_namespace(payload)
    owner = MemoryDocumentPathPolicy.trusted_segment(
        payload["owner_user_id"],
        "owner_user_id",
    )
    document_id = str(payload["document_id"])
    if job.target_uri != MemoryDocumentPathPolicy.document_uri(owner, document_id):
        raise ValueError("memory projection queue target is detached from its document")
    DocumentEditKind(str(payload["edit_kind"]))
    if (
        coerce_persisted_int(payload["logical_revision"]) <= 0
        or coerce_persisted_int(payload["projection_generation"]) <= 0
    ):
        raise ValueError("memory projection queue generation is invalid")
    payload["tenant_id"] = tenant
    payload["owner_user_id"] = owner
    return payload


def projection_deletion_digest(payload: Mapping[str, object]) -> str:
    """根据已校验事件计算删除屏障使用的确定性摘要。"""

    return deletion_event_digest(
        event_id=str(payload["event_id"]),
        document_id=str(payload["document_id"]),
        before_raw_digest=str(payload["before_raw_digest"]),
        projection_generation=coerce_persisted_int(payload["projection_generation"]),
    )


__all__ = ["parse_projection_job", "projection_deletion_digest"]
