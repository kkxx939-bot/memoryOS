from __future__ import annotations

import json


def l2_content(payload: dict | str) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2)
