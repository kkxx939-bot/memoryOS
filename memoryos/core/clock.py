"""Small, domain-independent clock helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["utc_now"]
