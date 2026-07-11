"""上下文数据库里的上下文打包器。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.retrieval.token_budget import HeuristicTokenCounter, TokenCounter


@dataclass(frozen=True)
class BudgetSlice:
    name: str
    budget: int
    used: int = 0
    items: list[dict] = field(default_factory=list)

    def remaining(self) -> int:
        return max(0, self.budget - self.used)


class ContextPacker:
    def __init__(self, total_budget: int, allocations: dict[str, int] | None = None, token_counter: TokenCounter | None = None) -> None:
        self.total_budget = max(0, int(total_budget))
        self.allocations = allocations or {}
        self.token_counter = token_counter or HeuristicTokenCounter()

    def pack(self, sections: dict[str, list[dict]]) -> dict:
        slices = {}
        load_plan = []
        dropped_contexts: list[dict] = []
        fallback_budget = self._fallback_budget(sections)
        remaining_total = self.total_budget
        for name, items in sections.items():
            budget = min(int(self.allocations.get(name, fallback_budget)), remaining_total)
            selected: list[dict] = []
            used = 0
            for index, item in enumerate(items):
                if remaining_total <= 0:
                    dropped_contexts.extend(
                        self._drop_payload(later, name, "total_budget_exhausted")
                        for later in items[index:]
                    )
                    break
                estimate = int(item.get("token_estimate", self._estimate_tokens(str(item.get("content", "")))))
                if estimate > remaining_total:
                    dropped_contexts.append(self._drop_payload(item, name, "total_budget_exceeded", estimate))
                    continue
                if selected and used + estimate > budget:
                    dropped_contexts.append(self._drop_payload(item, name, "section_budget_exceeded", estimate))
                    continue
                if estimate > budget:
                    dropped_contexts.append(self._drop_payload(item, name, "section_budget_exceeded", estimate))
                    continue
                selected_item = {**item, "token_estimate": estimate}
                selected.append(selected_item)
                load_plan.append(
                    {
                        "uri": item.get("uri", ""),
                        "section": name,
                        "layer": item.get("layer", "fallback"),
                        "token_estimate": estimate,
                        "reason": "selected_within_budget",
                    }
                )
                used += estimate
                remaining_total -= estimate
                if used >= budget:
                    dropped_contexts.extend(
                        self._drop_payload(later, name, "section_budget_exhausted")
                        for later in items[index + 1 :]
                    )
                    break
            slices[name] = BudgetSlice(name=name, budget=budget, used=used, items=selected)
            if not items and budget == 0:
                continue
        return {
            "total_budget": self.total_budget,
            "used": sum(item.used for item in slices.values()),
            "remaining": max(0, remaining_total),
            "slices": {
                name: {
                    "budget": item.budget,
                    "used": item.used,
                    "remaining": item.remaining(),
                    "items": item.items,
                }
                for name, item in slices.items()
            },
            "load_plan": load_plan,
            "dropped_contexts": dropped_contexts,
        }

    def _fallback_budget(self, sections: dict[str, list[dict]]) -> int:
        return self.total_budget // max(1, len(sections))

    def _estimate_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    def _drop_payload(self, item: dict, section: str, reason: str, estimate: int | None = None) -> dict:
        token_estimate = int(item.get("token_estimate", estimate or self._estimate_tokens(str(item.get("content", "")))))
        return {
            "uri": item.get("uri", ""),
            "section": section,
            "layer": item.get("layer", "fallback"),
            "token_estimate": token_estimate,
            "reason": reason,
        }
