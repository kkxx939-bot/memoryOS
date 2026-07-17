from __future__ import annotations

from memoryos.contextdb.layers.context_packer import ContextPacker


def test_context_packer_returns_load_plan_and_dropped_contexts() -> None:
    packed = ContextPacker(total_budget=80, allocations={"memory_anchor": 50}).pack(
        {
            "memory_anchor": [
                {"uri": "memoryos://a", "content": "a", "token_estimate": 40, "layer": "l1"},
                {"uri": "memoryos://b", "content": "b", "token_estimate": 40, "layer": "l1"},
            ]
        }
    )

    assert packed["remaining"] == 40
    assert packed["load_plan"] == [
        {
            "uri": "memoryos://a",
            "section": "memory_anchor",
            "layer": "l1",
            "token_estimate": 40,
            "reason": "selected_within_budget",
        }
    ]
    assert packed["dropped_contexts"][0]["uri"] == "memoryos://b"
    assert packed["dropped_contexts"][0]["reason"] == "section_budget_exceeded"
