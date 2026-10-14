"""MCP client side: start the servers over stdio and log every call."""
from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import config
from .toollog import Timer, ToolLog

CUSTOM = {
    "queue_state": "servers.queue_state.server",
    "priority_scorer": "servers.priority_scorer.server",
    "deterioration_watcher": "servers.deterioration_watcher.server",
    "referral_drafter": "servers.referral_drafter.server",
}
BORROWED_PKG = "@modelcontextprotocol/server-filesystem@2026.8.31"
# The borrowed server can write files; the agent may only read reference documents.
BORROWED_ALLOW = {"read_text_file", "list_directory", "read_file"}


class ToolCallError(RuntimeError):
    pass


class McpHub:
    def __init__(self, log: ToolLog, borrowed: bool | None = None):
        self.log = log
        self.borrowed = config.use_borrowed_fs() if borrowed is None else borrowed
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, list[str]] = {}
        self.sim_time: str | None = None
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "McpHub":
        env = {**os.environ, "TRIAGE_DB": str(config.db_path()), "PYTHONPATH": str(config.ROOT)}
        for name, module in CUSTOM.items():
            params = StdioServerParameters(command=sys.executable, args=["-m", module], env=env, cwd=str(config.ROOT))
            await self._open(name, params)
        if self.borrowed:
            npx = shutil.which("npx")
            if npx is None:
                self.log.event("borrowed_server_unavailable", "npx not found; referral facility lookup disabled")
                self.borrowed = False
            else:
                params = StdioServerParameters(command=npx, args=["-y", BORROWED_PKG, str(config.REFERENCE)], env=env)
                try:
                    await self._open("filesystem", params)
                except Exception as e:  # noqa: BLE001
                    self.log.event("borrowed_server_unavailable", repr(e))
                    self.borrowed = False
        return self

    async def _open(self, name: str, params: StdioServerParameters) -> None:
        errlog = open(os.devnull, "w")
        self._stack.callback(errlog.close)
        read, write = await self._stack.enter_async_context(stdio_client(params, errlog=errlog))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tl = await session.list_tools()
        self.sessions[name] = session
        self.tools[name] = [t.name for t in tl.tools]
        self.log.event("server_connected", {"server": name, "tools": self.tools[name]})

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        if server == "filesystem" and tool not in BORROWED_ALLOW:
            self.log.tool(server, tool, args, False, 0, error="blocked by client allowlist", sim_time=self.sim_time)
            raise ToolCallError(f"{server}.{tool} is not on the read-only allowlist")
        session = self.sessions.get(server)
        if session is None:
            raise ToolCallError(f"server {server} not connected")
        with Timer() as t:
            try:
                res = await session.call_tool(tool, args)
            except Exception as e:  # noqa: BLE001
                self.log.tool(server, tool, args, False, 0, error=repr(e), sim_time=self.sim_time)
                raise ToolCallError(f"{server}.{tool}: {e}") from e
        value = _unwrap(res)
        ok = not res.isError and not (isinstance(value, dict) and "error" in value and len(value) == 1)
        self.log.tool(server, tool, args, ok, t.ms, result=value, error=None if ok else str(value)[:300], sim_time=self.sim_time)
        if res.isError:
            raise ToolCallError(f"{server}.{tool}: {value}")
        return value


def _unwrap(res: Any) -> Any:
    sc = getattr(res, "structuredContent", None)
    if sc is not None:
        if isinstance(sc, dict) and set(sc) == {"result"}:
            return sc["result"]
        return sc
    texts = [c.text for c in res.content if getattr(c, "type", "") == "text"]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except (json.JSONDecodeError, TypeError):
        return joined
