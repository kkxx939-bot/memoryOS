"""上下文数据库里的资源解析器。"""

from __future__ import annotations

import json


class ResourceParser:
    def parse(self, text: str) -> dict:
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {"content": payload}
        except json.JSONDecodeError:
            return {"content": text}
