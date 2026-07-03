from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetSlice:
    name: str
    budget: int
    used: int = 0
    items: list[dict] = field(default_factory=list)

    def remaining(self) -> int:
        return max(0, self.budget - self.used)


class ContextPacker:
    def __init__(self, total_budget: int, allocations: dict[str, int] | None = None) -> None:
        self.total_budget = max(0, int(total_budget))
        self.allocations = allocations or {}

    def pack(self, sections: dict[str, list[dict]]) -> dict:
        slices = {}
        fallback_budget = self._fallback_budget(sections)
        for name, items in sections.items():
            budget = int(self.allocations.get(name, fallback_budget))
            selected: list[dict] = []
            used = 0
            for item in items:
                estimate = int(item.get("token_estimate", self._estimate_tokens(str(item.get("content", "")))))
                if selected and used + estimate > budget:
                    continue
                if estimate > budget and selected:
                    continue
                selected.append({**item, "token_estimate": estimate})
                used += estimate
                if used >= budget:
                    break
            slices[name] = BudgetSlice(name=name, budget=budget, used=used, items=selected)
        return {
            "total_budget": self.total_budget,
            "used": sum(item.used for item in slices.values()),
            "slices": {
                name: {
                    "budget": item.budget,
                    "used": item.used,
                    "remaining": item.remaining(),
                    "items": item.items,
                }
                for name, item in slices.items()
            },
        }

    def _fallback_budget(self, sections: dict[str, list[dict]]) -> int:
        return self.total_budget // max(1, len(sections))

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
