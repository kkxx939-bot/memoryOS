from __future__ import annotations

from memoryos.api.http.app import handle
from memoryos.api.sdk.client import MemoryOSClient


class MemoryOSMCPServer:
    def __init__(self, client: MemoryOSClient) -> None:
        self.client = client

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name == "memoryos_predict":
            return handle("POST /predict", self.client, arguments)
        if name == "memoryos_search_context":
            return handle("POST /context/search", self.client, arguments)
        if name == "memoryos_assemble_context":
            return handle("POST /context/assemble", self.client, arguments)
        if name == "memoryos_commit_session":
            return handle("POST /sessions/commit", self.client, arguments)
        raise KeyError(f"Unknown tool: {name}")
