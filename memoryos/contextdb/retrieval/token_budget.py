"""上下文数据库里的令牌预算。"""

from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str, model: str | None = None) -> int: ...


class HeuristicTokenCounter:
    def count(self, text: str, model: str | None = None) -> int:
        if not text:
            return 0
        cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
        other = len(text) - cjk
        return max(1, cjk + other // 4)


class TokenBudgetController:
    def __init__(self, total_budget: int) -> None:
        self.total_budget = max(0, int(total_budget))

    def allocate(self, weights: dict[str, float]) -> dict[str, int]:
        total = sum(max(0.0, value) for value in weights.values()) or 1.0
        return {name: int(self.total_budget * max(0.0, weight) / total) for name, weight in weights.items()}
