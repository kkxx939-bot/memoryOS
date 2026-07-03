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
        directory = self._dir(archive.archive_uri)
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
    ) -> Path:
        directory = self._dir(archive_uri)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".abstract.md").write_text(abstract, encoding="utf-8")
        (directory / ".overview.md").write_text(overview, encoding="utf-8")
        self._write_json(directory / "memory_diff.json", memory_diff)
        self._write_json(directory / "behavior_diff.json", behavior_diff)
        self._write_json(directory / "action_policy_diff.json", action_policy_diff)
        self._write_json(directory / "context_diff.json", context_diff)
        (directory / ".done").write_text("done\n", encoding="utf-8")
        return directory

    def _dir(self, archive_uri: str) -> Path:
        return ContextURI.parse(archive_uri).to_source_path(self.root, tenant_id=self.tenant_id)

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_json(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
