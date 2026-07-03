from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryChunk:
    chunk_id: str
    source_type: str
    source_id: str
    chunk_type: str
    text: str
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "metadata": self.metadata,
            "content_hash": self.content_hash or stable_hash(self.text),
        }


class ChunkingService:
    def chunks_for_memory(self, memory: dict) -> list[MemoryChunk]:
        memory_type = str(memory.get("type", memory.get("memory_type", "memory")))
        source_id = str(memory.get("id") or memory.get("path") or "")
        content = str(memory.get("content") or memory.get("text") or "")
        title = str(memory.get("title") or "")
        if memory_type == "profile":
            return self._section_chunks(source_id, "profile", content, {"title": title, "memory_type": memory_type})
        if memory_type == "case":
            return self._case_chunks(source_id, content, {"title": title, "memory_type": memory_type})
        text = "\n".join(item for item in (title, content) if item).strip()
        return [self._chunk(source_id, memory_type, "full", text, {"title": title, "memory_type": memory_type})] if text else []

    def _section_chunks(self, source_id: str, source_type: str, content: str, metadata: dict) -> list[MemoryChunk]:
        sections = [section.strip() for section in content.split("\n## ") if section.strip()]
        return [
            self._chunk(source_id, source_type, "section", section, {**metadata, "section_index": index})
            for index, section in enumerate(sections or [content])
            if section
        ]

    def _case_chunks(self, source_id: str, content: str, metadata: dict) -> list[MemoryChunk]:
        chunks = []
        buckets: dict[str, list[str]] = {"scene": [], "action": [], "feedback": []}
        for line in content.splitlines():
            lowered = line.lower()
            if lowered.startswith("scene"):
                buckets["scene"].append(line)
            elif "action" in lowered or "intervention" in lowered:
                buckets["action"].append(line)
            elif "reward" in lowered or "feedback" in lowered:
                buckets["feedback"].append(line)
        for chunk_type, lines in buckets.items():
            text = "\n".join(lines).strip()
            if text:
                chunks.append(self._chunk(source_id, "case", chunk_type, text, metadata))
        return chunks or [self._chunk(source_id, "case", "full", content, metadata)]

    def _chunk(self, source_id: str, source_type: str, chunk_type: str, text: str, metadata: dict) -> MemoryChunk:
        material = f"{source_type}:{source_id}:{chunk_type}:{stable_hash(text)}"
        return MemoryChunk(
            chunk_id=stable_hash(material)[:24],
            source_type=source_type,
            source_id=source_id,
            chunk_type=chunk_type,
            text=text,
            metadata=metadata,
            content_hash=stable_hash(text),
        )


def stable_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
