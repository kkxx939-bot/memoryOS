"""操作提交里的差异写入器。"""

from __future__ import annotations

import json
from pathlib import Path

from memoryos.core.ids import require_safe_path_segment
from memoryos.operations.commit.effect_marker import atomic_create_bytes
from memoryos.operations.model.context_diff import ContextDiff


class DiffWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, diff: ContextDiff) -> Path:
        diff_id = require_safe_path_segment(diff.diff_id, "diff_id")
        path = self.root / "system" / "diffs" / f"{diff_id}.json"
        if path.is_symlink():
            raise ValueError("diff control path cannot be a symbolic link")
        encoded = json.dumps(diff.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        atomic_create_bytes(path, encoded, artifact_root=self.root)
        return path
