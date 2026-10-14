"""queue-state MCP server: the only server that owns state (the SQLite ledger).

Read tools return the waiting room. Write tools can only create PROPOSED rows.
There is deliberately no approve/veto/sign tool: those decisions exist only in
the nurse console.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from triage_desk import ledger
from triage_desk.rules import minutes_between

mcp = FastMCP("queue-state")


def _summary(con, p: dict[str, Any], now: str) -> dict[str, Any]:
    obs = ledger.observations(con, p["id"], include_proposed=True, until=now)
    acts = ledger.actions(con, pid=p["id"])
    return {
        **{k: p[k] for k in ("id", "name", "age", "sex", "complaint", "arrival_time", "assigned_colour",
                             "colour_source", "queue_position", "seen_at", "pregnant", "diabetic", "history")},
        "waited_min": round(minutes_between(p["arrival_time"], now)),
        "last_observed_at": obs[-1]["observed_at"] if obs else None,
        "observation_count": len(obs),
        "pending_actions": [{"id": a["id"], "kind": a["kind"], "severity": a["severity"], "dedupe_key": a["dedupe_key"]}
                            for a in acts if a["status"] == "PROPOSED"],
        "vetoed": [{"id": a["id"], "kind": a["kind"], "reason": a["decision_reason"], "by": a["decided_by"]}
                   for a in acts if a["status"] == "VETOED"],
        "unprocessed_notes": [{"id": n["id"], "written_at": n["written_at"], "text": n["text"]}
                              for n in ledger.notes(con, p["id"], unprocessed=True, until=now)],
    }


@mcp.tool()
def get_queue(now: str, include_seen: bool = False) -> dict[str, Any]:
    """List patients in the waiting room at time `now` (ISO 8601), in current queue order."""
    with ledger.connect() as con:
        ps = [p for p in ledger.patients(con, include_seen=include_seen) if p["arrival_time"] <= now]
        return {"now": now, "patients": [_summary(con, p, now) for p in ps]}


@mcp.tool()
def get_patient(patient_id: str, now: str) -> dict[str, Any]:
    """Full record for one patient: observations (recorded and proposed), notes, actions."""
    with ledger.connect() as con:
        p = ledger.patient(con, patient_id)
        if p is None:
            return {"error": f"no patient {patient_id}"}
        obs = ledger.observations(con, patient_id, include_proposed=True, until=now)
        return {
            "patient": _summary(con, p, now),
            "observations": [{k: o[k] for k in ("id", "observed_at", "vitals", "discriminators", "resolved_discriminators",
                                                "source", "status", "entered_by", "note_id")} for o in obs],
            "notes": ledger.notes(con, patient_id, until=now),
            "actions": [{k: a[k] for k in ("id", "kind", "status", "severity", "rationale", "dedupe_key", "decided_by",
                                           "decision_reason")} for a in ledger.actions(con, pid=patient_id)],
        }


@mcp.tool()
def record_vitals(patient_id: str, observed_at: str, vitals: dict[str, Any], signs: list[str], run_id: str,
                  note_id: str | None = None, extraction: dict[str, Any] | None = None) -> dict[str, Any]:
    """Propose an observation extracted from a nurse note. It is stored as PROPOSED and
    does not count as recorded until a nurse confirms it in the console."""
    with ledger.connect() as con:
        if ledger.patient(con, patient_id) is None:
            return {"error": f"no patient {patient_id}"}
        oid = ledger.propose_observation(con, patient_id, observed_at, vitals, signs, note_id, extraction, run_id)
        evidence = (extraction or {}).get("evidence", {})
        res = ledger.propose(
            con, "record_observation", patient_id, {"observation_id": oid, "vitals": vitals, "signs": signs,
                                                    "note_id": note_id, "evidence": evidence,
                                                    "sign_evidence": (extraction or {}).get("sign_evidence", {}),
                                                    "rejected": (extraction or {}).get("rejected", [])},
            "Values extracted from nurse note; each quotes the note.", ["nurse note " + str(note_id)], run_id,
            severity="CONFIRM", dedupe_key=f"{patient_id}:OBS:{note_id}" if note_id else None, created_at=observed_at)
        return {"observation_id": oid, **res, "note": "PROPOSED: awaiting nurse confirmation"}


@mcp.tool()
def propose_action(kind: str, patient_id: str | None, payload: dict[str, Any], rationale: str, citations: list[str],
                   run_id: str, severity: str = "WATCH", dedupe_key: str | None = None,
                   depends_on: list[str] | None = None, created_at: str | None = None) -> dict[str, Any]:
    """Create a PROPOSED action for the nurse (change_colour, schedule_recheck, reorder_queue,
    draft_referral, flag). Identical proposals (same dedupe_key) that are open or were vetoed are not repeated."""
    with ledger.connect() as con:
        try:
            res = ledger.propose(con, kind, patient_id, payload, rationale, citations, run_id, severity,
                                 dedupe_key, depends_on, created_at)
        except ValueError as e:
            return {"error": str(e)}
        if not res["deduplicated"] and kind in ("change_colour", "reorder_queue", "schedule_recheck"):
            res["superseded"] = ledger.supersede_open(con, patient_id, kind, res["action_id"])
        return res


@mcp.tool()
def mark_note_processed(note_id: str, run_id: str) -> dict[str, Any]:
    """Record that the agent has read a nurse note, so it is not extracted twice."""
    with ledger.connect() as con:
        ledger.mark_note_processed(con, note_id, run_id)
        return {"note_id": note_id, "processed_by_run": run_id}


if __name__ == "__main__":
    mcp.run()
