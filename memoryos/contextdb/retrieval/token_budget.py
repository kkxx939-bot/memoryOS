from __future__ import annotations


class TokenBudgetController:
    def __init__(self, total_budget: int) -> None:
        self.total_budget = max(0, int(total_budget))

    def allocate(self, weights: dict[str, float]) -> dict[str, int]:
        total = sum(max(0.0, value) for value in weights.values()) or 1.0
        return {name: int(self.total_budget * max(0.0, weight) / total) for name, weight in weights.items()}
