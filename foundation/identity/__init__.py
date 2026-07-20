"""本地运行时使用的稳定身份值。"""

from foundation.identity.job_namespace import (
    InternalJobNamespaceError,
    require_internal_job_namespace,
)
from foundation.identity.local import LOCAL_STORAGE_NAMESPACE, LocalUserContext
from foundation.identity.workspace import (
    normalize_workspace_id,
    normalize_workspace_scope_key,
    repository_workspace_id,
    workspace_ids_from_metadata,
)

__all__ = [
    "InternalJobNamespaceError",
    "LOCAL_STORAGE_NAMESPACE",
    "LocalUserContext",
    "normalize_workspace_id",
    "normalize_workspace_scope_key",
    "repository_workspace_id",
    "require_internal_job_namespace",
    "workspace_ids_from_metadata",
]
