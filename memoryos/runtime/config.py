from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    root: str

    @property
    def root_path(self) -> Path:
        return Path(self.root)
