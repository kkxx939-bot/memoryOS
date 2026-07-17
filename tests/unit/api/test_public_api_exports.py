from __future__ import annotations


def test_public_api_exports() -> None:
    from memoryos import ActionPolicy, ContextDB, MemoryOSClient, PredictionRequest

    assert MemoryOSClient.__name__ == "MemoryOSClient"
    assert PredictionRequest.__name__ == "PredictionRequest"
    assert ActionPolicy.__name__ == "ActionPolicy"
    assert ContextDB.__name__ == "ContextDB"
