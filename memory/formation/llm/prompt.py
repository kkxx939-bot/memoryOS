"""从不可变证据生成受限的记忆语义提取 Prompt。"""

from __future__ import annotations

import json
from collections.abc import Sequence

from memory.core.formation.schema import MEMORY_SCHEMA_VERSION, MemoryCandidateSchema
from pre.evidence import EvidenceEpisode
from pre.session import SessionArchive


class MemoryExtractionPromptBuilder:
    """只暴露证据和语义字段，排除全部可信存储控制字段。"""

    def build(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
        episode: EvidenceEpisode,
    ) -> str:
        contract = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "task": "从不可变 Session 证据中提取可长期保存的语义记忆候选。",
            "output": {
                "candidates": [
                    {
                        "candidate_kind": "一个已配置的候选类型",
                        "title": "简短标题",
                        "subject": "语义主体",
                        "body": "有证据支撑的 Markdown 正文",
                        "entity_hints": ["语义实体标签"],
                        "topic_hints": ["语义主题标签"],
                        "occurred_at": "已知时必须包含时区的 ISO-8601 时间",
                        "temporal_status": "可选的语义状态",
                        "relation_hints": ["语义关系"],
                        "evidence_refs": ["event_id"],
                        "field_evidence_refs": {"body": ["event_id"]},
                        "confidence": 0.0,
                    }
                ]
            },
            "forbidden": [
                "文件路径",
                "document_id",
                "tenant",
                "owner",
                "workspace authority",
                "ACL",
                "SQL",
                "删除或彻底擦除",
                "projection generation",
                "final authority",
            ],
            "candidate_kinds": [
                {
                    "candidate_kind": item.candidate_kind.value,
                    "description": item.description,
                    "requires_occurred_at": item.requires_occurred_at,
                }
                for item in schemas
            ],
            "evidence": [
                {
                    "event_id": item.event_id,
                    "event_type": item.event_type,
                    "actor": item.actor.to_dict(),
                    "occurred_at": item.occurred_at.isoformat(),
                    "text": item.text(),
                }
                for item in episode.events
            ],
            "archive_binding": {
                "session_id": archive.session_id,
                "archive_uri": archive.archive_uri,
                "archive_digest": archive.archive_digest,
                "manifest_digest": archive.manifest_digest,
            },
        }
        return json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["MemoryExtractionPromptBuilder"]
