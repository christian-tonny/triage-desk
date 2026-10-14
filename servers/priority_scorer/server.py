"""priority-scorer MCP server: deterministic SATS adult scoring. No model, no state."""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from triage_desk import rules
from triage_desk import scale as sc

mcp = FastMCP("priority-scorer")


@mcp.tool()
def score_vitals(vitals: dict[str, Any], signs: list[str] | None = None, age_years: float | None = None,
                 pregnant: bool = False, diabetic: bool = False) -> dict[str, Any]:
    """Score one set of observations with the SATS adult chart. Returns colour, TEWS breakdown,
    every driver with its citation, what is unknown, and how bad it could be."""
    return sc.score(vitals, signs or [], age_years, pregnant, diabetic)


@mcp.tool()
def score_patient(patient: dict[str, Any], observations: list[dict[str, Any]], now: str) -> dict[str, Any]:
    """Score a patient's current picture: the latest value of each parameter across their observations."""
    return rules.score_history(patient, observations, now)


@mcp.tool()
def rerank_queue(patients: list[dict[str, Any]], now: str) -> dict[str, Any]:
    """Propose a seeing order (colour, then minutes left to SATS target, then arrival).
    This is a proposal only; the queue does not change until a nurse approves."""
    current = [p["id"] for p in sorted((p for p in patients if not p.get("seen_at")),
                                       key=lambda p: p.get("queue_position") or 0)]
    rows = rules.rerank(patients, now)
    proposed = [r["patient_id"] for r in rows]
    moves = []
    for r in rows:
        old = current.index(r["patient_id"]) + 1
        if old != r["position"]:
            moves.append({"patient_id": r["patient_id"], "from": old, "to": r["position"], "colour": r["colour"],
                          "minutes_to_target": r["minutes_to_target"]})
    return {"proposed_order": proposed, "current_order": current, "changed": proposed != current, "rows": rows,
            "moves": moves, "rule": "Order by SATS colour, then minutes left to target time, then arrival",
            "cite": "SATS-2012 Table 3 target times"}


@mcp.tool()
def scale_reference() -> dict[str, Any]:
    """Return the SATS source, priorities and TEWS bands this scorer uses (for display and citation)."""
    s = sc.scale()
    return {"source": s["source"], "priorities": s["priorities"], "tews_bands": s["tews_bands"],
            "local_policy_status": sc.policy()["_status"]}


if __name__ == "__main__":
    mcp.run()
