from __future__ import annotations

import os
import sys

import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402


def test_official_mcp_sdk_initialize_list_call(tmp_path) -> None:  # noqa: ANN001
    async def run() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "openApi.mcp.stdio"],
            cwd=os.getcwd(),
            env={**os.environ, "MEMORYOS_ROOT": str(tmp_path), "MEMORYOS_ADAPTER_ID": "cursor"},
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                tools = await session.list_tools()
                health = await session.call_tool("memoryos_health", {})
                names = {tool.name for tool in tools.tools}
                assert initialized.serverInfo.name == "memoryos"
                assert "memoryos_search_context" in names
                assert "memoryos_predict" not in names
                assert health.isError is False

    import asyncio

    asyncio.run(run())
