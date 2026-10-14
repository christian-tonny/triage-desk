"""deterioration-watcher MCP server: compares a waiting patient's latest picture with their triage colour."""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from triage_desk import rules

mcp = FastMCP("deterioration-watcher")


@mcp.tool()
def flag_deterioration(patient: dict[str, Any], observations: list[dict[str, Any]], now: str) -> dict[str, Any]:
    """Return ESCALATE / WATCH / INFO flags for one patient. Each flag cites its rule and carries a
    dedupe_key so the same finding on the same data is raised only once. Never proposes a downgrade."""
    return rules.flag_deterioration(patient, observations, now)


@mcp.tool()
def schedule_recheck(colour: str, last_observed_at: str, now: str) -> dict[str, Any]:
    """When this patient's vitals should next be measured, under local policy for their colour."""
    return rules.recheck(colour, last_observed_at, now)


if __name__ == "__main__":
    mcp.run()
