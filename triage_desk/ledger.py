"""SQLite ledger: patients, observations, proposals and the approval gate.

Agents can only *propose*. Approve / edit / veto / sign exist only as Python
functions called by the nurse console; none of them is exposed as an MCP tool.
apply_action() refuses anything a named human has not approved.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
  id TEXT PRIMARY KEY, name TEXT, age REAL, sex TEXT, complaint TEXT,
  arrival_time TEXT NOT NULL, assigned_colour TEXT NOT NULL, colour_source TEXT,
  pregnant INTEGER DEFAULT 0, diabetic INTEGER DEFAULT 0, history TEXT,
  queue_position INTEGER, seen_at TEXT
);
CREATE TABLE IF NOT EXISTS observations (
  id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, observed_at TEXT NOT NULL,
  vitals TEXT NOT NULL, discriminators TEXT NOT NULL, resolved_discriminators TEXT DEFAULT '[]',
  source TEXT NOT NULL, entered_by TEXT, note_id TEXT, status TEXT NOT NULL,
  extraction TEXT
);
CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, written_at TEXT NOT NULL,
  author TEXT, text TEXT NOT NULL, processed_by_run TEXT
);
CREATE TABLE IF NOT EXISTS actions (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, run_id TEXT, kind TEXT NOT NULL,
  patient_id TEXT, payload TEXT NOT NULL, rationale TEXT, citations TEXT,
  severity TEXT, dedupe_key TEXT, depends_on TEXT DEFAULT '[]',
  status TEXT NOT NULL, decided_by TEXT, decided_at TEXT, decision_reason TEXT,
  edited_payload TEXT, applied_at TEXT
);
CREATE INDEX IF NOT EXISTS actions_dedupe ON actions(dedupe_key);
CREATE TABLE IF NOT EXISTS referrals (
  id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, action_id TEXT, created_at TEXT,
  body TEXT NOT NULL, status TEXT NOT NULL, signed_by TEXT, signed_at TEXT
);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, actor TEXT NOT NULL,
  event TEXT NOT NULL, detail TEXT
);
"""

ACTION_KINDS = {"record_observation", "change_colour", "schedule_recheck", "reorder_queue", "draft_referral", "flag"}
OPEN = ("PROPOSED",)
CLOSED_FOR_DEDUPE = ("PROPOSED", "APPROVED", "EDITED", "VETOED")


class GateError(PermissionError):
    """Raised when something tries to apply an action no human approved."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@contextmanager
def connect(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    p = Path(path or config.db_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def audit(con: sqlite3.Connection, actor: str, event: str, detail: dict | None = None, ts: str | None = None) -> None:
    con.execute("INSERT INTO audit (ts, actor, event, detail) VALUES (?,?,?,?)",
                (ts or _now(), actor, event, json.dumps(detail or {}, default=str)))


def reset(path: Path | str | None = None) -> None:
    p = Path(path or config.db_path())
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(p) + suffix)
        if f.exists():
            f.unlink()
    with connect(p):
        pass


# ---------- loading fixtures ----------

def load_fixture(fixture: dict[str, Any], path: Path | str | None = None) -> None:
    with connect(path) as con:
        for i, p in enumerate(fixture["patients"], 1):
            con.execute(
                "INSERT INTO patients (id,name,age,sex,complaint,arrival_time,assigned_colour,colour_source,pregnant,diabetic,history,queue_position,seen_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (p["id"], p["name"], p.get("age"), p.get("sex"), p.get("complaint"), p["arrival_time"],
                 p["assigned_colour"], "nurse triage (fixture)", int(p.get("pregnant", False)),
                 int(p.get("diabetic", False)), p.get("history", ""), i, p.get("seen_at")),
            )
            for ob in p.get("observations", []):
                con.execute(
                    "INSERT INTO observations (id,patient_id,observed_at,vitals,discriminators,source,entered_by,status)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (ob.get("id") or new_id("OBS"), p["id"], ob["observed_at"], json.dumps(ob.get("vitals", {})),
                     json.dumps(ob.get("discriminators", [])), "nurse", ob.get("entered_by", "triage nurse"), "recorded"),
                )
            for n in p.get("notes", []):
                con.execute("INSERT INTO notes (id,patient_id,written_at,author,text) VALUES (?,?,?,?,?)",
                            (n.get("id") or new_id("NOTE"), p["id"], n["written_at"], n.get("author", "nurse"), n["text"]))
        audit(con, "system", "fixture_loaded", {"name": fixture.get("name"), "patients": len(fixture["patients"])})


# ---------- reads ----------

def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("vitals", "discriminators", "resolved_discriminators", "payload", "citations", "depends_on", "edited_payload", "extraction"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    for k in ("pregnant", "diabetic"):
        if k in d:
            d[k] = bool(d[k])
    return d


def patients(con: sqlite3.Connection, include_seen: bool = True) -> list[dict[str, Any]]:
    q = "SELECT * FROM patients" + ("" if include_seen else " WHERE seen_at IS NULL") + " ORDER BY queue_position"
    return [_row(r) for r in con.execute(q)]


def patient(con: sqlite3.Connection, pid: str) -> dict[str, Any] | None:
    return _row(con.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone())


def observations(con: sqlite3.Connection, pid: str, include_proposed: bool = False, until: str | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM observations WHERE patient_id=?"
    args: list[Any] = [pid]
    if not include_proposed:
        q += " AND status='recorded'"
    else:
        q += " AND status IN ('recorded','proposed')"
    if until:
        q += " AND observed_at<=?"
        args.append(until)
    return [_row(r) for r in con.execute(q + " ORDER BY observed_at", args)]


def notes(con: sqlite3.Connection, pid: str | None = None, unprocessed: bool = False, until: str | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM notes WHERE 1=1"
    args: list[Any] = []
    if pid:
        q += " AND patient_id=?"
        args.append(pid)
    if unprocessed:
        q += " AND processed_by_run IS NULL"
    if until:
        q += " AND written_at<=?"
        args.append(until)
    return [_row(r) for r in con.execute(q + " ORDER BY written_at", args)]


def actions(con: sqlite3.Connection, status: str | None = None, pid: str | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM actions WHERE 1=1"
    args: list[Any] = []
    if status:
        q += " AND status=?"
        args.append(status)
    if pid:
        q += " AND patient_id=?"
        args.append(pid)
    return [_row(r) for r in con.execute(q + " ORDER BY created_at, rowid", args)]


def action(con: sqlite3.Connection, aid: str) -> dict[str, Any] | None:
    return _row(con.execute("SELECT * FROM actions WHERE id=?", (aid,)).fetchone())


def referrals(con: sqlite3.Connection) -> list[dict[str, Any]]:
    return [_row(r) for r in con.execute("SELECT * FROM referrals ORDER BY created_at")]


# ---------- agent-side writes (proposals only) ----------

def add_note(con: sqlite3.Connection, pid: str, text: str, written_at: str, author: str = "nurse") -> str:
    nid = new_id("NOTE")
    con.execute("INSERT INTO notes (id,patient_id,written_at,author,text) VALUES (?,?,?,?,?)", (nid, pid, written_at, author, text))
    audit(con, author, "note_added", {"note_id": nid, "patient_id": pid}, ts=written_at)
    return nid


def mark_note_processed(con: sqlite3.Connection, note_id: str, run_id: str) -> None:
    con.execute("UPDATE notes SET processed_by_run=? WHERE id=?", (run_id, note_id))


def propose_observation(con: sqlite3.Connection, pid: str, observed_at: str, vitals: dict, discriminators: list[str],
                        note_id: str | None, extraction: dict | None, run_id: str) -> str:
    oid = new_id("OBS")
    con.execute(
        "INSERT INTO observations (id,patient_id,observed_at,vitals,discriminators,source,entered_by,note_id,status,extraction)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (oid, pid, observed_at, json.dumps(vitals), json.dumps(discriminators), "agent_extracted", f"agent:{run_id}",
         note_id, "proposed", json.dumps(extraction or {})),
    )
    return oid


def propose(con: sqlite3.Connection, kind: str, pid: str | None, payload: dict, rationale: str,
            citations: list[str], run_id: str, severity: str = "WATCH", dedupe_key: str | None = None,
            depends_on: list[str] | None = None, created_at: str | None = None) -> dict[str, Any]:
    if kind not in ACTION_KINDS:
        raise ValueError(f"unknown action kind {kind}")
    if dedupe_key:
        ph = ",".join("?" * len(CLOSED_FOR_DEDUPE))
        existing = con.execute(f"SELECT * FROM actions WHERE dedupe_key=? AND status IN ({ph})",
                               (dedupe_key, *CLOSED_FOR_DEDUPE)).fetchone()
        if existing:
            return {"action_id": existing["id"], "status": existing["status"], "deduplicated": True}
    aid = new_id("ACT")
    con.execute(
        "INSERT INTO actions (id,created_at,run_id,kind,patient_id,payload,rationale,citations,severity,dedupe_key,depends_on,status)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (aid, created_at or _now(), run_id, kind, pid, json.dumps(payload, default=str), rationale,
         json.dumps(citations), severity, dedupe_key, json.dumps(depends_on or []), "PROPOSED"),
    )
    audit(con, f"agent:{run_id}", "proposed", {"action_id": aid, "kind": kind, "patient_id": pid}, ts=created_at)
    return {"action_id": aid, "status": "PROPOSED", "deduplicated": False}


def supersede_open(con: sqlite3.Connection, pid: str, kind: str, keep: str) -> int:
    cur = con.execute("UPDATE actions SET status='SUPERSEDED' WHERE patient_id IS ? AND kind=? AND status='PROPOSED' AND id<>?",
                      (pid, kind, keep))
    return cur.rowcount


# ---------- human-side decisions (console only) ----------

def _require_human(who: str) -> None:
    if not who or not who.strip() or who.lower().startswith("agent"):
        raise GateError("A named human must make this decision.")


def decide(con: sqlite3.Connection, aid: str, decision: str, who: str, reason: str = "",
           edited_payload: dict | None = None, at: str | None = None) -> dict[str, Any]:
    _require_human(who)
    a = action(con, aid)
    if a is None:
        raise KeyError(aid)
    if a["status"] != "PROPOSED":
        raise GateError(f"{aid} is {a['status']}, not PROPOSED")
    if decision not in {"APPROVED", "EDITED", "VETOED"}:
        raise ValueError(decision)
    if decision in {"EDITED", "VETOED"} and not reason.strip():
        raise GateError("Edits and vetoes need a reason (it stays visible on the board).")
    at = at or _now()
    con.execute("UPDATE actions SET status=?, decided_by=?, decided_at=?, decision_reason=?, edited_payload=? WHERE id=?",
                (decision, who, at, reason, json.dumps(edited_payload) if edited_payload else None, aid))
    audit(con, who, decision.lower(), {"action_id": aid, "reason": reason, "edited": edited_payload}, ts=at)
    if decision == "VETOED":
        return {"action_id": aid, "status": "VETOED"}
    for dep in a["depends_on"] or []:
        d = action(con, dep)
        if d and d["status"] == "PROPOSED":
            decide(con, dep, "APPROVED", who, f"approved with {aid}", at=at)
    return apply_action(con, aid, at=at)


def apply_action(con: sqlite3.Connection, aid: str, at: str | None = None) -> dict[str, Any]:
    a = action(con, aid)
    if a is None:
        raise KeyError(aid)
    if a["status"] not in ("APPROVED", "EDITED") or not a["decided_by"]:
        raise GateError(f"{aid} has not been approved by a human (status {a['status']}).")
    if a["applied_at"]:
        return {"action_id": aid, "status": a["status"], "already_applied": True}
    payload = a["edited_payload"] or a["payload"]
    at = at or _now()
    kind = a["kind"]
    result: dict[str, Any] = {"action_id": aid, "status": a["status"], "kind": kind}
    if kind == "record_observation":
        con.execute("UPDATE observations SET status='recorded', entered_by=? WHERE id=?",
                    (f"{a['decided_by']} (confirmed agent extraction)", payload["observation_id"]))
    elif kind == "change_colour":
        con.execute("UPDATE patients SET assigned_colour=?, colour_source=? WHERE id=?",
                    (payload["to"], f"{a['decided_by']} via {aid}", a["patient_id"]))
    elif kind == "reorder_queue":
        for pos, pid in enumerate(payload["order"], 1):
            con.execute("UPDATE patients SET queue_position=? WHERE id=?", (pos, pid))
    elif kind == "draft_referral":
        rid = new_id("REF")
        con.execute("INSERT INTO referrals (id,patient_id,action_id,created_at,body,status) VALUES (?,?,?,?,?,?)",
                    (rid, a["patient_id"], aid, at, payload["body"], "DRAFT"))
        result["referral_id"] = rid
    # schedule_recheck and flag are acknowledgements: approving records that the nurse saw them.
    con.execute("UPDATE actions SET applied_at=? WHERE id=?", (at, aid))
    audit(con, a["decided_by"], "applied", result, ts=at)
    return result


def sign_referral(con: sqlite3.Connection, rid: str, who: str, at: str | None = None) -> dict[str, Any]:
    _require_human(who)
    r = con.execute("SELECT * FROM referrals WHERE id=?", (rid,)).fetchone()
    if r is None:
        raise KeyError(rid)
    if r["status"] != "DRAFT":
        raise GateError(f"{rid} is already {r['status']}")
    at = at or _now()
    body = r["body"].replace("*** DRAFT - NOT SENT - REQUIRES SIGNATURE ***",
                             f"*** SIGNED by {who} at {at[:16].replace('T', ' ')} - not sent electronically; print or phone ***")
    body = body.replace("Signed: ______________________  (name, cadre)   Time: ______", f"Signed: {who}   Time: {at[11:16]}")
    con.execute("UPDATE referrals SET status='SIGNED', signed_by=?, signed_at=?, body=? WHERE id=?", (who, at, body, rid))
    audit(con, who, "referral_signed", {"referral_id": rid}, ts=at)
    return {"referral_id": rid, "status": "SIGNED", "signed_by": who}


def record_nurse_observation(con: sqlite3.Connection, pid: str, observed_at: str, vitals: dict,
                             discriminators: list[str], who: str) -> str:
    _require_human(who)
    oid = new_id("OBS")
    con.execute(
        "INSERT INTO observations (id,patient_id,observed_at,vitals,discriminators,source,entered_by,status)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (oid, pid, observed_at, json.dumps(vitals), json.dumps(discriminators), "nurse", who, "recorded"),
    )
    audit(con, who, "observation_recorded", {"observation_id": oid, "patient_id": pid}, ts=observed_at)
    return oid


def mark_seen(con: sqlite3.Connection, pid: str, at: str, who: str) -> None:
    _require_human(who)
    con.execute("UPDATE patients SET seen_at=? WHERE id=?", (at, pid))
    con.execute("UPDATE actions SET status='SUPERSEDED' WHERE patient_id=? AND status='PROPOSED' AND kind<>'draft_referral'", (pid,))
    audit(con, who, "patient_seen", {"patient_id": pid}, ts=at)
