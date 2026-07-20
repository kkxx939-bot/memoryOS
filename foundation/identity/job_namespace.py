"""校验耐久队列任务只能使用本地固定存储命名空间。"""

from __future__ import annotations

from collections.abc import Mapping

from foundation.identity.local import LOCAL_STORAGE_NAMESPACE


class InternalJobNamespaceError(ValueError):
    """队列任务缺少固定命名空间，或试图切换到其他命名空间。"""


def require_internal_job_namespace(payload: Mapping[str, object]) -> str:
    """返回固定命名空间，并拒绝缺失或不一致的耐久任务。"""

    declared = payload.get("tenant_id")
    if declared != LOCAL_STORAGE_NAMESPACE:
        raise InternalJobNamespaceError(
            "queued job must use the local fixed storage namespace"
        )
    return LOCAL_STORAGE_NAMESPACE


__all__ = ["InternalJobNamespaceError", "require_internal_job_namespace"]
