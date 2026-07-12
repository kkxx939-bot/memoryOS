"""操作提交里的差异写入器。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from memoryos.core.ids import require_safe_path_segment
from memoryos.operations.model.context_diff import ContextDiff


class DiffWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, diff: ContextDiff) -> Path:
        diff_id = require_safe_path_segment(diff.diff_id, "diff_id")
        path = self.root / "system" / "diffs" / f"{diff_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("x", encoding="utf-8") as handle:
                os.chmod(tmp, 0o600)
                handle.write(json.dumps(diff.to_dict(), ensure_ascii=False, indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            os.chmod(path, 0o600)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            tmp.unlink(missing_ok=True)
        return path
