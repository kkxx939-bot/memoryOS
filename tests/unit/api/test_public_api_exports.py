from __future__ import annotations


def test_public_api_exports() -> None:
    from openApi.sdk import ActionCandidate, ActionPolicy, MemoryOSClient, PredictionRequest

    assert MemoryOSClient.__name__ == "MemoryOSClient"
    assert PredictionRequest.__name__ == "PredictionRequest"
    assert ActionCandidate.__name__ == "ActionCandidate"
    assert ActionPolicy.__name__ == "ActionPolicy"
