from __future__ import annotations

import json
from typing import Any


def render_memory_markdown(metadata: dict[str, Any], body: str) -> str:
    frontmatter = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    title = metadata.get("title", "Memory")
    return f"---\n{frontmatter}\n---\n# {title}\n\n{body.strip()}\n"


def parse_memory_markdown(content: str) -> tuple[dict[str, Any], str]:
    if not content.startswith("---\n"):
        return {}, _strip_generated_title(content)
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}, _strip_generated_title(content)
    raw_meta = content[4:end]
    body = content[end + 5 :]
    try:
        metadata = json.loads(raw_meta)
    except json.JSONDecodeError:
        metadata = {}
    return metadata, _strip_generated_title(body)


def _strip_generated_title(body: str) -> str:
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("# "):
        lines.pop(0)
        if lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip() + ("\n" if lines else "")
