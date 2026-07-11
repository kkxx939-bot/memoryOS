"""操作提交里的差异写入器。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from memoryos.operations.model.context_diff import ContextDiff


class DiffWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, diff: ContextDiff) -> Path:
        path = self.root / "system" / "diffs" / f"{diff.diff_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(diff.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path
