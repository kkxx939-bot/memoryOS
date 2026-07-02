from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from ..session.memory.lifecycle import classify_lifecycle, hotness_score
from ..session.memory.markdown import parse_memory_markdown, render_memory_markdown
from ..session.memory.weights import score_memory_weight
from ..session.memory.models import MEMORY_TYPES, TYPE_DIR, MemoryItem, summarize_text, utc_now
from ..session.memory.schema import MEMORY_TYPE_SPECS, memory_type_spec, render_template, validate_metadata
from ..models.embeddings import EmbeddingProvider, HashingEmbeddingProvider, cosine_similarity
from ..models.rerank import RerankProvider, rerank_with_fallback


class MemoryStore:
    def __init__(
        self,
        root: str | Path,
        embedding_provider: EmbeddingProvider | None = None,
        rerank_provider: RerankProvider | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.index_dir = self.root / "index"
        self.db_path = self.index_dir / "memory.sqlite3"
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self.rerank_provider = rerank_provider

    def init(self, user_id: str) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        user_root = self.root / "user" / user_id
        for directory in TYPE_DIR.values():
            path = user_root / directory
            path.mkdir(parents=True, exist_ok=True)
            self._ensure_layer_files(path, directory)
        (user_root / "sessions").mkdir(parents=True, exist_ok=True)
        (user_root / "episodes").mkdir(parents=True, exist_ok=True)
        (user_root / "daily").mkdir(parents=True, exist_ok=True)
        (user_root / "archive").mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _ensure_layer_files(self, directory: Path, label: str) -> None:
        abstract = directory / ".abstract.md"
        overview = directory / ".overview.md"
        if not abstract.exists():
            abstract.write_text(f"{label} memory directory.\n", encoding="utf-8")
        if not overview.exists():
            overview.write_text(f"# {label}\n\nNo overview generated yet.\n", encoding="utf-8")

    def _connect(self) -> sqlite3.Connection:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    tags TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    active_count INTEGER NOT NULL DEFAULT 0,
                    hotness REAL NOT NULL DEFAULT 0.0,
                    lifecycle_state TEXT NOT NULL DEFAULT 'cold',
                    temporal_scope TEXT NOT NULL,
                    base_weight REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    positive_count INTEGER NOT NULL,
                    negative_count INTEGER NOT NULL,
                    effective_weight REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(id UNINDEXED, user_id UNINDEXED, type UNINDEXED, title, tags, abstract, content)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    embedding TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_directory_layers (
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    directory_path TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    overview TEXT NOT NULL,
                    abstract_embedding TEXT NOT NULL,
                    overview_embedding TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, type)
                )
                """
            )

    def add_memory(self, item: MemoryItem) -> Path:
        self._init_db()
        path = self.root / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._normalize_metadata(item.metadata(), item.text)
        validate_metadata(metadata)
        content = render_memory_markdown(metadata, item.text)
        path.write_text(content, encoding="utf-8")
        self._upsert_index(metadata, item.text)
        self._refresh_directory_layers(path.parent)
        return path

    def upsert_profile(self, user_id: str, text: str, mode: str = "append") -> dict[str, Any]:
        return self._upsert_fixed_memory(
            user_id=user_id,
            rel_path=str(PurePosixPath("user") / user_id / "profile" / "user-profile.md"),
            memory_type="profile",
            title="User Profile",
            text=text,
            tags=["profile", "user_model"],
            source="profile-update",
            mode=mode,
        )

    def update_daily_behavior(self, user_id: str, text: str, day: str | None = None, mode: str = "append") -> dict[str, Any]:
        day = day or date.today().isoformat()
        return self._upsert_fixed_memory(
            user_id=user_id,
            rel_path=str(PurePosixPath("user") / user_id / "daily" / day / "behavior.md"),
            memory_type="event",
            title=f"Daily Behavior {day}",
            text=text,
            tags=["daily", "behavior", day],
            source=f"daily:{day}",
            mode=mode,
        )

    def record_event(
        self,
        user_id: str,
        event_type: str,
        text: str,
        day: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        day = day or date.today().isoformat()
        tags = ["event", event_type, day, *(tags or [])]
        event = MemoryItem(
            user_id=user_id,
            memory_type="event",
            title=f"{day} {event_type}",
            text=text,
            tags=tags,
            source=f"event:{event_type}",
        )
        event_path = self.add_memory(event)
        log_path = self.root / "user" / user_id / "daily" / day / "events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "created_at": utc_now(),
            "event_type": event_type,
            "text": text,
            "memory_path": event.path,
            "tags": tags,
        }
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        daily_update = self.update_daily_behavior(
            user_id=user_id,
            day=day,
            text=f"- {event_type}: {text}",
            mode="append",
        )
        return {
            "event_uri": str(PurePosixPath(event_path.relative_to(self.root).as_posix())),
            "daily_log": str(PurePosixPath(log_path.relative_to(self.root).as_posix())),
            "daily_behavior_update": daily_update,
        }

    def read_memory(self, rel_path: str) -> tuple[dict[str, Any], str]:
        path = self.root / rel_path
        content = path.read_text(encoding="utf-8")
        return parse_memory_markdown(content)

    def resolve_memory(self, identifier: str, user_id: str) -> dict[str, Any]:
        self._init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? AND (id = ? OR path = ?)",
                (user_id, identifier, identifier),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Memory not found for user {user_id}: {identifier}")
        data = self._row_to_dict(row)
        metadata, body = self.read_memory(data["path"])
        data.update(metadata)
        data["content"] = body
        return data

    def update_memory(
        self,
        identifier: str,
        user_id: str,
        title: str | None = None,
        text: str | None = None,
        tags: list[str] | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.resolve_memory(identifier, user_id)
        metadata, body = self.read_memory(current["path"])
        before = {"metadata": deepcopy(metadata), "content": body}
        if title is not None:
            metadata["title"] = title
        if tags is not None:
            metadata["tags"] = tags
        if text is not None:
            body = text
        if metadata_patch:
            metadata.update(metadata_patch)
        metadata["updated_at"] = utc_now()
        metadata["abstract"] = summarize_text(body)
        metadata = self._normalize_metadata(metadata, body)
        validate_metadata(metadata)
        path = self.root / current["path"]
        path.write_text(render_memory_markdown(metadata, body), encoding="utf-8")
        self._upsert_index(metadata, body)
        self._refresh_directory_layers(path.parent)
        return {"uri": current["path"], "before": before, "after": {"metadata": metadata, "content": body}}

    def delete_memory(self, identifier: str, user_id: str) -> dict[str, Any]:
        current = self.resolve_memory(identifier, user_id)
        path = self.root / current["path"]
        metadata, body = self.read_memory(current["path"])
        if path.exists():
            path.unlink()
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (current["id"],))
            conn.execute("DELETE FROM memory_fts WHERE id = ?", (current["id"],))
            conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (current["id"],))
        self._refresh_directory_layers(path.parent)
        return {"uri": current["path"], "metadata": metadata, "deleted_content": body}

    def merge_memory(self, target_identifier: str, source_identifier: str, user_id: str) -> dict[str, Any]:
        target = self.resolve_memory(target_identifier, user_id)
        source = self.resolve_memory(source_identifier, user_id)
        if target["id"] == source["id"]:
            raise ValueError("Cannot merge a memory into itself")
        target_meta, target_body = self.read_memory(target["path"])
        source_meta, source_body = self.read_memory(source["path"])
        merged_body = (
            target_body.rstrip()
            + "\n\n## Merged Evidence\n\n"
            + f"Source: {source_meta.get('title')} ({source['path']})\n\n"
            + source_body.strip()
            + "\n"
        )
        update = self.update_memory(target["path"], user_id, text=merged_body)
        deletion = self.delete_memory(source["path"], user_id)
        return {
            "target_uri": target["path"],
            "source_uri": source["path"],
            "update": update,
            "delete": deletion,
        }

    def _upsert_fixed_memory(
        self,
        user_id: str,
        rel_path: str,
        memory_type: str,
        title: str,
        text: str,
        tags: list[str],
        source: str,
        mode: str,
    ) -> dict[str, Any]:
        if mode not in {"append", "replace"}:
            raise ValueError("mode must be 'append' or 'replace'")
        path = self.root / rel_path
        if path.exists():
            metadata, body = self.read_memory(rel_path)
            before = {"metadata": deepcopy(metadata), "content": body}
            if mode == "append":
                new_body = body.rstrip() + f"\n\n## Update {utc_now()}\n\n{text.strip()}\n"
            else:
                new_body = text.strip()
            metadata["title"] = title
            metadata["type"] = memory_type
            metadata["tags"] = tags
            metadata["source"] = source
            metadata["updated_at"] = utc_now()
            metadata["abstract"] = summarize_text(new_body)
            metadata["path"] = rel_path
            metadata = self._normalize_metadata(metadata, new_body)
            validate_metadata(metadata)
            path.write_text(render_memory_markdown(metadata, new_body), encoding="utf-8")
            self._upsert_index(metadata, new_body)
            self._refresh_directory_layers(path.parent)
            return {
                "uri": rel_path,
                "operation": "update",
                "before": before,
                "after": {"metadata": metadata, "content": new_body},
            }

        item = MemoryItem(
            user_id=user_id,
            memory_type=memory_type,
            title=title,
            text=text,
            tags=tags,
            source=source,
            path=rel_path,
        )
        self.add_memory(item)
        return {
            "uri": rel_path,
            "operation": "create",
            "before": None,
            "after": {"metadata": self._normalize_metadata(item.metadata(), text), "content": text},
        }

    def _upsert_index(self, metadata: dict[str, Any], body: str) -> None:
        metadata = self._normalize_metadata(metadata, body)
        validate_metadata(metadata)
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (metadata["id"],))
            conn.execute("DELETE FROM memory_fts WHERE id = ?", (metadata["id"],))
            row = (
                metadata["id"],
                metadata["user_id"],
                metadata["type"],
                metadata["title"],
                metadata["path"],
                json.dumps(metadata.get("tags", []), ensure_ascii=False),
                metadata.get("abstract", ""),
                body,
                metadata.get("source", "unknown"),
                float(metadata.get("confidence", 1.0)),
                metadata.get("created_at", utc_now()),
                metadata.get("updated_at", utc_now()),
                metadata.get("last_accessed_at"),
                int(metadata.get("active_count", 0)),
                float(metadata.get("hotness", 0.0)),
                metadata.get("lifecycle_state", "cold"),
                metadata["temporal_scope"],
                float(metadata["base_weight"]),
                int(metadata["evidence_count"]),
                int(metadata["positive_count"]),
                int(metadata["negative_count"]),
                float(metadata["effective_weight"]),
            )
            conn.execute(
                """
                INSERT INTO memories
                (
                    id, user_id, type, title, path, tags, abstract, content,
                    source, confidence, created_at, updated_at,
                    last_accessed_at, active_count, hotness, lifecycle_state,
                    temporal_scope, base_weight, evidence_count, positive_count,
                    negative_count, effective_weight
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            conn.execute(
                """
                INSERT INTO memory_fts (id, user_id, type, title, tags, abstract, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (row[0], row[1], row[2], row[3], row[5], row[6], row[7]),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (id, user_id, path, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    metadata["id"],
                    metadata["user_id"],
                    metadata["path"],
                    json.dumps(self.embedding_provider.embed(self._embedding_text(metadata, body))),
                    metadata["updated_at"],
                ),
            )

    def search(
        self,
        query: str,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        self._init_db()
        rows = self._keyword_search_rows(query, user_id, memory_type, limit)
        rows = self._touch_access(rows)
        return [self._row_to_dict(row) for row in rows]

    def hybrid_search(
        self,
        query: str,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        self._init_db()
        candidate_limit = max(limit * 3, 12)
        keyword_rows = self._keyword_search_rows(query, user_id, memory_type, candidate_limit)
        embedding_rows = self._embedding_search_rows(query, user_id, memory_type, candidate_limit)
        candidates: dict[str, dict[str, Any]] = {}

        for row in keyword_rows:
            data = dict(row)
            data["keyword_score"] = float(data.get("keyword_score", 0.0))
            data.setdefault("embedding_score", 0.0)
            candidates[data["id"]] = data

        for row in embedding_rows:
            data = dict(row)
            existing = candidates.get(data["id"], data)
            existing["embedding_score"] = max(
                float(existing.get("embedding_score", 0.0)),
                float(data.get("embedding_score", 0.0)),
            )
            existing.setdefault("keyword_score", 0.0)
            candidates[data["id"]] = existing

        ranked = []
        for data in candidates.values():
            final_score = self._hybrid_score(data)
            data["final_score"] = final_score
            data["score"] = final_score
            ranked.append(data)
        ranked.sort(key=lambda row: row["final_score"], reverse=True)
        self._rerank_memory_candidates(query, ranked)
        ranked.sort(key=lambda row: row["final_score"], reverse=True)
        rows = self._touch_access(ranked[:limit])
        return [self._row_to_dict(row) for row in rows]

    def _keyword_search_rows(
        self,
        query: str,
        user_id: str,
        memory_type: str | None,
        limit: int,
    ) -> list[sqlite3.Row | dict[str, Any]]:
        fts_query = self._to_fts_query(query)
        params: list[Any] = [fts_query, user_id]
        where = ["memory_fts MATCH ?", "memory_fts.user_id = ?"]
        if memory_type:
            where.append("memory_fts.type = ?")
            params.append(memory_type)
        params.append(limit)
        sql = f"""
            SELECT memories.*, bm25(memory_fts) AS score
            FROM memory_fts
            JOIN memories ON memories.id = memory_fts.id
            WHERE {" AND ".join(where)}
            ORDER BY score
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return self._fallback_like_search(query, user_id, memory_type, limit)
        ranked = []
        total = max(len(rows), 1)
        for index, row in enumerate(rows):
            data = dict(row)
            data["keyword_score"] = 1.0 - (index / total)
            ranked.append(data)
        return ranked

    def list_recent(self, user_id: str, limit: int = 8) -> list[dict[str, Any]]:
        self._init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_by_type(self, user_id: str, memory_type: str, limit: int = 8) -> list[dict[str, Any]]:
        self._init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND type = ?
                ORDER BY effective_weight DESC, hotness DESC, updated_at DESC
                LIMIT ?
                """,
                (user_id, memory_type, limit),
            ).fetchall()
        rows = self._touch_access([dict(row) for row in rows])
        return [self._row_to_dict(row) for row in rows]

    def lifecycle_report(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        self._init_db()
        self._refresh_lifecycle_scores(user_id=user_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                ORDER BY
                    CASE lifecycle_state
                        WHEN 'cold' THEN 0
                        WHEN 'warm' THEN 1
                        ELSE 2
                    END,
                    hotness ASC,
                    updated_at ASC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def archive_cold_memories(
        self,
        user_id: str,
        limit: int = 20,
        max_hotness: float = 0.12,
        allowed_types: set[str] | None = None,
    ) -> dict[str, Any]:
        self._init_db()
        self._refresh_lifecycle_scores(user_id=user_id)
        allowed_types = allowed_types or {"event", "case", "feedback", "intervention"}
        placeholders = ", ".join("?" for _ in allowed_types)
        params: list[Any] = [user_id, max_hotness, *sorted(allowed_types), limit]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE user_id = ?
                  AND hotness <= ?
                  AND type IN ({placeholders})
                ORDER BY hotness ASC, updated_at ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        archived = []
        archive_path = self.root / "user" / user_id / "archive" / "cold_memories.jsonl"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("a", encoding="utf-8") as fp:
            for row in rows:
                memory = self._row_to_dict(row)
                metadata, body = self.read_memory(memory["path"])
                record = {
                    "archived_at": utc_now(),
                    "metadata": metadata,
                    "content": body,
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.delete_memory(memory["path"], user_id=user_id)
                archived.append(
                    {
                        "path": memory["path"],
                        "type": memory["type"],
                        "title": memory["title"],
                        "hotness": memory["hotness"],
                    }
                )
        return {
            "archive_path": str(PurePosixPath(archive_path.relative_to(self.root).as_posix())),
            "archived": archived,
            "summary": {"total_archived": len(archived)},
        }

    def reindex(self, user_id: str | None = None) -> None:
        self._init_db()
        with self._connect() as conn:
            if user_id:
                conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM memory_fts WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM memory_embeddings WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM memory_directory_layers WHERE user_id = ?", (user_id,))
            else:
                conn.execute("DELETE FROM memories")
                conn.execute("DELETE FROM memory_fts")
                conn.execute("DELETE FROM memory_embeddings")
                conn.execute("DELETE FROM memory_directory_layers")
        base = self.root / "user"
        if not base.exists():
            return
        for path in base.rglob("*.md"):
            if path.name.startswith("."):
                continue
            rel = str(PurePosixPath(path.relative_to(self.root).as_posix()))
            metadata, body = parse_memory_markdown(path.read_text(encoding="utf-8"))
            if not metadata.get("id"):
                continue
            if user_id and metadata.get("user_id") != user_id:
                continue
            metadata["path"] = rel
            metadata["abstract"] = summarize_text(body)
            metadata = self._normalize_metadata(metadata, body)
            self._upsert_index(metadata, body)
        for directory in (self.root / "user").glob("*/*"):
            if directory.is_dir() and directory.name in set(TYPE_DIR.values()):
                if user_id and directory.parent.name != user_id:
                    continue
                self._refresh_directory_layers(directory)

    def _refresh_directory_layers(self, directory: Path) -> None:
        memories = []
        for path in directory.glob("*.md"):
            if path.name.startswith("."):
                continue
            metadata, _ = parse_memory_markdown(path.read_text(encoding="utf-8"))
            if metadata:
                memories.append(metadata)
        if not memories:
            memory_type = self._memory_type_for_directory(directory)
            if memory_type:
                self._delete_directory_layer(directory, memory_type)
            return
        memory_type = self._memory_type_for_directory(directory)
        spec = memory_type_spec(memory_type) if memory_type else None
        memories.sort(
            key=lambda item: (
                float(item.get("effective_weight", 0.0)),
                float(item.get("hotness", 0.0)),
                str(item.get("updated_at", "")),
            ),
            reverse=True,
        )
        abstract_lines = self._directory_abstract_lines(memories)
        overview_lines = self._directory_overview_lines(spec, memories)
        abstract = "\n".join(abstract_lines).strip() + "\n"
        overview = "\n".join(overview_lines).strip() + "\n"
        (directory / ".abstract.md").write_text(abstract, encoding="utf-8")
        (directory / ".overview.md").write_text(overview, encoding="utf-8")
        if memory_type:
            self._upsert_directory_layer(directory, memory_type, abstract, overview)

    def _directory_abstract_lines(self, memories: list[dict[str, Any]]) -> list[str]:
        lines = []
        for memory in memories[:8]:
            tags = ", ".join(str(tag) for tag in memory.get("tags", [])[:6])
            prefix = f"- {memory.get('title')}: {memory.get('abstract', '')}"
            if tags:
                prefix += f" [tags: {tags}]"
            lines.append(prefix)
        return lines

    def _directory_overview_lines(self, spec: Any, memories: list[dict[str, Any]]) -> list[str]:
        count = len(memories)
        evidence_count = sum(int(memory.get("evidence_count", 1) or 1) for memory in memories)
        top_tags: dict[str, int] = {}
        for memory in memories:
            for tag in memory.get("tags", []):
                key = str(tag)
                top_tags[key] = top_tags.get(key, 0) + 1
        ranked_tags = sorted(top_tags.items(), key=lambda item: item[1], reverse=True)[:8]
        strongest_lines = []
        for memory in memories[:12]:
            strongest_lines.append(
                (
                    f"- {memory.get('title')} | weight={float(memory.get('effective_weight', 0.0)):.3f} "
                    f"hotness={float(memory.get('hotness', 0.0)):.3f} | {memory.get('abstract', '')} "
                    f"({memory.get('path')})"
                )
            )
        context = {
            "overview_title": spec.overview_title if spec else "Overview",
            "memory_count": count,
            "evidence_count": evidence_count,
            "top_tags": ", ".join(f"{tag}:{total}" for tag, total in ranked_tags),
            "strongest_memories": "\n".join(strongest_lines),
        }
        template = spec.overview_template if spec and spec.overview_template else (
            "# {{ overview_title }}\n\n"
            "- memory_count: {{ memory_count }}\n"
            "- evidence_count: {{ evidence_count }}\n"
            "- top_tags: {{ top_tags }}\n\n"
            "## Strongest memories\n"
            "{{ strongest_memories }}\n"
        )
        return render_template(template, context).splitlines()

    def _memory_type_for_directory(self, directory: Path) -> str:
        directory_name = directory.name
        for memory_type, spec in MEMORY_TYPE_SPECS.items():
            if spec.directory == directory_name:
                return memory_type
        return ""

    def _upsert_directory_layer(self, directory: Path, memory_type: str, abstract: str, overview: str) -> None:
        try:
            rel_path = str(PurePosixPath(directory.relative_to(self.root).as_posix()))
        except ValueError:
            return
        parts = PurePosixPath(rel_path).parts
        if len(parts) < 3 or parts[0] != "user":
            return
        user_id = parts[1]
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_directory_layers
                (
                    user_id, type, directory_path, abstract, overview,
                    abstract_embedding, overview_embedding, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    memory_type,
                    rel_path,
                    abstract,
                    overview,
                    json.dumps(self.embedding_provider.embed(self._directory_embedding_text(memory_type, abstract))),
                    json.dumps(self.embedding_provider.embed(self._directory_embedding_text(memory_type, overview))),
                    now,
                ),
            )

    def _delete_directory_layer(self, directory: Path, memory_type: str) -> None:
        try:
            rel_path = str(PurePosixPath(directory.relative_to(self.root).as_posix()))
        except ValueError:
            return
        parts = PurePosixPath(rel_path).parts
        if len(parts) < 3 or parts[0] != "user":
            return
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM memory_directory_layers WHERE user_id = ? AND type = ?",
                (parts[1], memory_type),
            )

    def _directory_embedding_text(self, memory_type: str, text: str) -> str:
        spec = memory_type_spec(memory_type)
        return render_template(
            spec.embedding_template,
            {
                "memory_type": memory_type,
                "title": spec.overview_title,
                "tags": spec.directory,
                "abstract": spec.description,
                "content": text,
            },
        )

    def rank_directory_layers(
        self,
        query: str,
        user_id: str,
        memory_types: set[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        self._init_db()
        where = ["user_id = ?"]
        params: list[Any] = [user_id]
        if memory_types:
            placeholders = ", ".join("?" for _ in memory_types)
            where.append(f"type IN ({placeholders})")
            params.extend(sorted(memory_types))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_directory_layers
                WHERE {" AND ".join(where)}
                """,
                params,
            ).fetchall()
        query_embedding = self.embedding_provider.embed(query)
        query_tokens = self._search_tokens(query)
        ranked = []
        for row in rows:
            data = dict(row)
            abstract_score = cosine_similarity(query_embedding, json.loads(data["abstract_embedding"]))
            overview_score = cosine_similarity(query_embedding, json.loads(data["overview_embedding"]))
            keyword_score = self._token_overlap_score(
                query_tokens,
                self._search_tokens(
                    f"{data['type']} {data['directory_path']} {data['abstract']} {data['overview']}"
                ),
            )
            layer = "L1" if overview_score >= abstract_score else "L0"
            semantic_score = max(abstract_score, overview_score)
            final_score = (
                keyword_score * 0.35
                + semantic_score * 0.40
                + self._type_boost(str(data["type"])) * 0.15
                + 0.10
            )
            data["keyword_score"] = keyword_score
            data["semantic_score"] = semantic_score
            data["score"] = round(max(0.0, min(1.0, final_score)), 6)
            data["level"] = layer
            ranked.append(data)
        self._rerank_directory_candidates(query, ranked)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    def _rerank_memory_candidates(self, query: str, ranked: list[dict[str, Any]]) -> None:
        if not ranked or self.rerank_provider is None:
            return
        documents = [self._rerank_document(row) for row in ranked]
        fallback_scores = [float(row.get("final_score", row.get("score", 0.0)) or 0.0) for row in ranked]
        rerank_scores = rerank_with_fallback(self.rerank_provider, query, documents, fallback_scores)
        for row, rerank_score, fallback in zip(ranked, rerank_scores, fallback_scores, strict=True):
            row["rerank_score"] = rerank_score
            row["final_score"] = round(max(0.0, min(1.0, rerank_score * 0.65 + fallback * 0.35)), 6)
            row["score"] = row["final_score"]

    def _rerank_directory_candidates(self, query: str, ranked: list[dict[str, Any]]) -> None:
        if not ranked or self.rerank_provider is None:
            return
        documents = [
            "\n".join(
                [
                    f"type: {row.get('type', '')}",
                    f"path: {row.get('directory_path', '')}",
                    f"abstract: {row.get('abstract', '')}",
                    f"overview: {row.get('overview', '')}",
                ]
            )
            for row in ranked
        ]
        fallback_scores = [float(row.get("score", 0.0) or 0.0) for row in ranked]
        rerank_scores = rerank_with_fallback(self.rerank_provider, query, documents, fallback_scores)
        for row, rerank_score, fallback in zip(ranked, rerank_scores, fallback_scores, strict=True):
            row["rerank_score"] = rerank_score
            row["score"] = round(max(0.0, min(1.0, rerank_score * 0.60 + fallback * 0.40)), 6)

    def _rerank_document(self, row: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"type: {row.get('type', '')}",
                f"title: {row.get('title', '')}",
                f"tags: {row.get('tags', '')}",
                f"abstract: {row.get('abstract', '')}",
                f"content: {row.get('content', '')}",
            ]
        )

    def _normalize_metadata(self, metadata: dict[str, Any], body: str) -> dict[str, Any]:
        metadata = dict(metadata)
        metadata["abstract"] = summarize_text(body)
        metadata["confidence"] = float(metadata["confidence"])
        metadata["active_count"] = max(0, int(metadata["active_count"]))
        metadata["hotness"] = hotness_score(metadata["active_count"], metadata["updated_at"])
        metadata["lifecycle_state"] = classify_lifecycle(float(metadata["hotness"]))
        metadata["evidence_count"] = max(1, int(metadata["evidence_count"]))
        metadata["positive_count"] = max(0, int(metadata["positive_count"]))
        metadata["negative_count"] = max(0, int(metadata["negative_count"]))
        weight = score_memory_weight(metadata)
        metadata["base_weight"] = weight.base_weight
        metadata["effective_weight"] = weight.effective_weight
        return metadata

    def _embedding_text(self, metadata: dict[str, Any], body: str) -> str:
        memory_type = str(metadata.get("type", ""))
        spec = memory_type_spec(memory_type)
        return render_template(
            spec.embedding_template,
            {
                "memory_type": memory_type,
                "title": metadata.get("title", ""),
                "tags": " ".join(metadata.get("tags", [])),
                "abstract": metadata.get("abstract", ""),
                "content": body,
            },
        )

    def _embedding_search_rows(
        self,
        query: str,
        user_id: str,
        memory_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embedding_provider.embed(query)
        where = ["memories.user_id = ?"]
        params: list[Any] = [user_id]
        if memory_type:
            where.append("memories.type = ?")
            params.append(memory_type)
        sql = f"""
            SELECT memories.*, memory_embeddings.embedding
            FROM memory_embeddings
            JOIN memories ON memories.id = memory_embeddings.id
            WHERE {" AND ".join(where)}
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        scored = []
        for row in rows:
            data = dict(row)
            embedding = json.loads(data.pop("embedding"))
            score = cosine_similarity(query_embedding, embedding)
            if score <= 0:
                continue
            data["embedding_score"] = score
            data["score"] = score
            scored.append(data)
        scored.sort(key=lambda item: item["embedding_score"], reverse=True)
        return scored[:limit]

    def _hybrid_score(self, row: dict[str, Any]) -> float:
        return (
            float(row.get("keyword_score", 0.0)) * 0.25
            + float(row.get("embedding_score", 0.0)) * 0.35
            + float(row.get("hotness", 0.0)) * 0.15
            + self._type_boost(str(row.get("type", ""))) * 0.15
            + float(row.get("confidence", 0.0)) * 0.10
        )

    def _type_boost(self, memory_type: str) -> float:
        boosts = {
            "profile": 0.85,
            "preference": 0.9,
            "habit": 0.95,
            "trigger": 0.95,
            "policy": 0.85,
            "feedback": 0.75,
            "intervention": 0.8,
            "case": 0.7,
            "event": 0.55,
        }
        return boosts.get(memory_type, 0.5)

    def _search_tokens(self, text: str) -> set[str]:
        lowered = text.lower()
        tokens = set()
        current = []
        for ch in lowered:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.add("".join(current))
                    current = []
                if "\u4e00" <= ch <= "\u9fff":
                    tokens.add(ch)
        if current:
            tokens.add("".join(current))
        return {token for token in tokens if token}

    def _token_overlap_score(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left & right)
        if overlap == 0:
            return 0.0
        return min(1.0, overlap / max(3, len(left)))

    def _touch_access(self, rows: list[sqlite3.Row | dict[str, Any]]) -> list[sqlite3.Row | dict[str, Any]]:
        if not rows:
            return []
        now = utc_now()
        updates = []
        extras = {}
        for row in rows:
            active_count = int(row["active_count"] or 0) + 1
            score = hotness_score(active_count, row["updated_at"])
            state = classify_lifecycle(score)
            updates.append((now, active_count, score, state, row["id"], row["path"]))
            keys = row.keys()
            extras[row["id"]] = {
                key: row[key]
                for key in ("score", "keyword_score", "embedding_score", "rerank_score", "final_score")
                if key in keys
            }

        with self._connect() as conn:
            for last_accessed_at, active_count, score, state, memory_id, _ in updates:
                conn.execute(
                    """
                    UPDATE memories
                    SET last_accessed_at = ?, active_count = ?, hotness = ?, lifecycle_state = ?
                    WHERE id = ?
                    """,
                    (last_accessed_at, active_count, score, state, memory_id),
                )
        for last_accessed_at, active_count, score, state, _, rel_path in updates:
            self._write_lifecycle_metadata(rel_path, last_accessed_at, active_count, score, state)
        return self._fetch_rows_in_order([row["id"] for row in rows], extras)

    def _write_lifecycle_metadata(
        self,
        rel_path: str,
        last_accessed_at: str,
        active_count: int,
        score: float,
        state: str,
    ) -> None:
        path = self.root / rel_path
        if not path.exists():
            return
        metadata, body = parse_memory_markdown(path.read_text(encoding="utf-8"))
        metadata = self._normalize_metadata(metadata, body)
        metadata["path"] = rel_path
        metadata["last_accessed_at"] = last_accessed_at
        metadata["active_count"] = active_count
        metadata["hotness"] = score
        metadata["lifecycle_state"] = state
        validate_metadata(metadata)
        path.write_text(render_memory_markdown(metadata, body), encoding="utf-8")

    def _fetch_rows_in_order(self, memory_ids: list[str], extras: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        if not memory_ids:
            return []
        placeholders = ", ".join("?" for _ in memory_ids)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids).fetchall()
        by_id = {row["id"]: dict(row) for row in rows}
        ordered = []
        for memory_id in memory_ids:
            if memory_id not in by_id:
                continue
            row = by_id[memory_id]
            row.update(extras.get(memory_id, {}))
            ordered.append(row)
        return ordered

    def _refresh_lifecycle_scores(self, user_id: str | None = None) -> None:
        where = ""
        params: list[Any] = []
        if user_id:
            where = "WHERE user_id = ?"
            params.append(user_id)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT id, active_count, updated_at FROM memories {where}", params).fetchall()
            for row in rows:
                score = hotness_score(int(row["active_count"] or 0), row["updated_at"])
                state = classify_lifecycle(score)
                conn.execute(
                    "UPDATE memories SET hotness = ?, lifecycle_state = ? WHERE id = ?",
                    (score, state, row["id"]),
                )

    def _row_to_dict(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        try:
            data["tags"] = json.loads(data["tags"])
        except json.JSONDecodeError:
            data["tags"] = []
        data["active_count"] = int(data.get("active_count") or 0)
        data["hotness"] = float(data.get("hotness") or 0.0)
        data["base_weight"] = float(data.get("base_weight") or 0.0)
        data["effective_weight"] = float(data.get("effective_weight") or 0.0)
        data["evidence_count"] = int(data.get("evidence_count") or 0)
        data["positive_count"] = int(data.get("positive_count") or 0)
        data["negative_count"] = int(data.get("negative_count") or 0)
        return data

    def _to_fts_query(self, query: str) -> str:
        terms = [part.strip().replace('"', "") for part in query.split() if part.strip()]
        if not terms:
            return '""'
        return " OR ".join(f'"{term}"' for term in terms)

    def _fallback_like_search(
        self,
        query: str,
        user_id: str,
        memory_type: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        terms = [part.strip() for part in query.split() if part.strip()] or [query.strip()]
        terms = [term for term in terms if term]
        if not terms:
            return []
        where = ["user_id = ?"]
        params: list[Any] = [user_id]
        if memory_type:
            where.append("type = ?")
            params.append(memory_type)
        like_parts = []
        for term in terms:
            like_parts.append("(title LIKE ? OR tags LIKE ? OR abstract LIKE ? OR content LIKE ?)")
            params.extend([f"%{term}%"] * 4)
        params.append(limit)
        sql = f"""
            SELECT *, 0.0 AS score FROM memories
            WHERE {" AND ".join(where)} AND ({" OR ".join(like_parts)})
            ORDER BY updated_at DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        ranked = []
        total = max(len(rows), 1)
        for index, row in enumerate(rows):
            data = dict(row)
            data["keyword_score"] = 0.65 - (index / total * 0.2)
            ranked.append(data)
        return ranked


def validate_memory_type(memory_type: str) -> str:
    if memory_type not in MEMORY_TYPES:
        known = ", ".join(sorted(MEMORY_TYPES))
        raise ValueError(f"Unknown memory type: {memory_type}. Known types: {known}")
    return memory_type
