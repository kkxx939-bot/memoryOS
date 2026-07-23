"""领域分流前保存的原始 Session 业务归档。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from foundation.clock import utc_now
from foundation.ids import new_id


@dataclass
class SessionArchive:
    """汇集一次 Session 的事实输入，不承担任何领域提交职责。"""

    user_id: str
    session_id: str
    archive_uri: str
    messages: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)
    action_results: list[dict] = field(default_factory=list)
    feedback: list[dict] = field(default_factory=list)
    used_contexts: list[dict] = field(default_factory=list)
    used_skills: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: new_id("commit"))
    created_at: str = field(default_factory=utc_now)
    schema_version: str = "session_archive_v3"
    archive_digest: str = ""
    manifest_digest: str = ""
    manifest_uri: str = ""

    def manifest(self) -> dict:
        """生成归档提交清单，不包含任何领域消费者的执行结果。"""

        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "archive_uri": self.archive_uri,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "archive_digest": self.archive_digest,
            "manifest_digest": self.manifest_digest,
            "manifest_uri": self.manifest_uri,
            "metadata": self.metadata,
            "phase": "sync_archive",
            "files": [
                "commit_head.json",
                "events/",
                "objects/",
                "manifests/",
            ],
        }


__all__ = ["SessionArchive"]
