from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from memoryos import MemoryOSClient, PredictionRequest

    with tempfile.TemporaryDirectory() as root:
        client = MemoryOSClient(root)
        result = client.predict(
            PredictionRequest(
                user_id="example-user",
                episode_id="example-session",
                observation="The room is warm and the user is working.",
                available_actions=["ask_user", "do_nothing"],
            )
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
