"""统一事务内核能够承载的操作动作。"""

from __future__ import annotations

from enum import Enum


class OperationAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    SUPERSEDE = "supersede"
    MERGE = "merge"
    REWARD = "reward"
    PENALIZE = "penalize"
    COOLDOWN = "cooldown"
    SUPPRESS = "suppress"
    DISABLE = "disable"
    ARCHIVE = "archive"
    COMPRESS = "compress"
    REFRESH_LAYERS = "refresh_layers"
    REINDEX = "reindex"
