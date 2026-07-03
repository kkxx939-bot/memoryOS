from __future__ import annotations

from memoryos.api.http.app import handle
from memoryos.api.sdk.client import MemoryOSClient


class MemoryOSMCPServer:
    def __init__(self, client: MemoryOSClient) -> None:
        self.client = client

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name == "memoryos_predict":
            return handle("POST /predict", self.client, arguments)
        raise KeyError(f"Unknown tool: {name}")
