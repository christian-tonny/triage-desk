"""Append-only JSONL log of every MCP tool call and every LLM call in a run."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PREVIEW = 600


def _preview(obj: Any) -> Any:
    s = json.dumps(obj, default=str)
    return obj if len(s) <= PREVIEW else s[:PREVIEW] + "…"


class ToolLog:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "tool_calls.jsonl"
        self.run_id = run_id
        self.seq = 0
        self._lock = threading.Lock()
        self.llm_tokens_in = 0
        self.llm_tokens_out = 0
        self.llm_seconds = 0.0
        self.llm_calls = 0
        self.tool_calls = 0
        self.tool_errors = 0

    def write(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.seq += 1
            rec = {"run_id": self.run_id, "seq": self.seq,
                   "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **rec}
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def tool(self, server: str, tool: str, args: dict, ok: bool, latency_ms: float,
             result: Any = None, error: str | None = None, sim_time: str | None = None) -> None:
        self.tool_calls += 1
        if not ok:
            self.tool_errors += 1
        self.write({"kind": "mcp_tool", "server": server, "tool": tool, "sim_time": sim_time, "args": _preview(args),
                    "ok": ok, "latency_ms": round(latency_ms, 1), "result": _preview(result), "error": error})

    def llm(self, purpose: str, model: str, backend: str, tokens_in: int, tokens_out: int, seconds: float,
            ok: bool, detail: Any = None, error: str | None = None) -> None:
        self.llm_calls += 1
        self.llm_tokens_in += tokens_in
        self.llm_tokens_out += tokens_out
        self.llm_seconds += seconds
        self.write({"kind": "llm", "purpose": purpose, "backend": backend, "model": model, "tokens_in": tokens_in,
                    "tokens_out": tokens_out, "seconds": round(seconds, 2), "ok": ok, "detail": _preview(detail), "error": error})

    def event(self, name: str, detail: Any = None) -> None:
        self.write({"kind": "event", "name": name, "detail": _preview(detail)})


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1000
