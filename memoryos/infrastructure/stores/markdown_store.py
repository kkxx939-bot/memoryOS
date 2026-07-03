from __future__ import annotations

import os
from pathlib import Path


class MarkdownStore:
    def write_text_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")
