from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    from memoryos import MemoryOSClient, PredictionRequest

    with tempfile.TemporaryDirectory() as root:
        client = MemoryOSClient(root)
        result = client.predict(
            PredictionRequest(
                user_id="smoke-user",
                episode_id="smoke-session",
                observation="The user says the office is too bright.",
                available_actions=["ask_user", "do_nothing"],
            )
        )
        print(
            {
                "decision": result.decision.action,
                "candidate_count": len(result.candidates),
                "context_uri_count": len(result.action_context.source_uris),
            }
        )


if __name__ == "__main__":
    main()
