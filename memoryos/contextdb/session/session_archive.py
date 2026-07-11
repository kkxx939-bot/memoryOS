"""上下文数据库里的会话归档。"""

from __future__ import annotations

import json
from pathlib import Path

from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.session.session_model import SessionArchive


class SessionArchiveStore:
    def __init__(self, root: str | Path, tenant_id: str = "default") -> None:
        self.root = Path(root)
        self.tenant_id = tenant_id

    def write_sync_archive(self, archive: SessionArchive) -> Path:
        directory = self._dir(archive.archive_uri, tenant_id=self._archive_tenant(archive))
        directory.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(directory / "messages.jsonl", archive.messages)
        self._write_jsonl(directory / "observations.jsonl", archive.observations)
        self._write_jsonl(directory / "predictions.jsonl", archive.predictions)
        self._write_jsonl(directory / "action_results.jsonl", archive.action_results)
        self._write_jsonl(directory / "feedback.jsonl", archive.feedback)
        self._write_json(directory / "used_contexts.json", archive.used_contexts)
        self._write_json(directory / "used_skills.json", archive.used_skills)
        self._write_jsonl(directory / "tool_results.jsonl", archive.tool_results)
        self._write_json(directory / "commit_manifest.json", archive.manifest())
        return directory

    def write_async_outputs(
        self,
        archive_uri: str,
        abstract: str,
        overview: str,
        memory_diff: dict,
        behavior_diff: dict,
        action_policy_diff: dict,
        context_diff: dict,
        tenant_id: str | None = None,
    ) -> Path:
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".abstract.md").write_text(abstract, encoding="utf-8")
        (directory / ".overview.md").write_text(overview, encoding="utf-8")
        self._write_json(directory / "memory_diff.json", memory_diff)
        self._write_json(directory / "behavior_diff.json", behavior_diff)
        self._write_json(directory / "action_policy_diff.json", action_policy_diff)
        self._write_json(directory / "context_diff.json", context_diff)
        (directory / ".done").write_text("done\n", encoding="utf-8")
        return directory

    def async_outputs_done_for_task(self, archive: SessionArchive) -> bool:
        directory = self._dir(archive.archive_uri, tenant_id=self._archive_tenant(archive))
        if not (directory / ".done").exists():
            return False
        try:
            payload = json.loads((directory / "memory_diff.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("task_id") == archive.task_id

    def read_archive(self, archive_uri: str, *, tenant_id: str | None = None) -> SessionArchive:
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        manifest = self._read_json(directory / "commit_manifest.json")
        return SessionArchive(
            user_id=str(manifest["user_id"]),
            session_id=str(manifest["session_id"]),
            archive_uri=archive_uri,
            messages=self._read_jsonl(directory / "messages.jsonl"),
            observations=self._read_jsonl(directory / "observations.jsonl"),
            predictions=self._read_jsonl(directory / "predictions.jsonl"),
            action_results=self._read_jsonl(directory / "action_results.jsonl"),
            feedback=self._read_jsonl(directory / "feedback.jsonl"),
            used_contexts=list(self._read_json(directory / "used_contexts.json") or []),
            used_skills=list(self._read_json(directory / "used_skills.json") or []),
            tool_results=self._read_jsonl(directory / "tool_results.jsonl"),
            metadata=dict(manifest.get("metadata", {}) or {}),
            task_id=str(manifest["task_id"]),
            created_at=str(manifest.get("created_at", "")),
        )

    def _dir(self, archive_uri: str, *, tenant_id: str | None = None) -> Path:
        return ContextURI.parse(archive_uri).to_source_path(self.root, tenant_id=tenant_id or self.tenant_id)

    def _archive_tenant(self, archive: SessionArchive) -> str:
        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        return str(metadata.get("tenant_id") or scope.get("tenant_id") or self.tenant_id)

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_json(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
