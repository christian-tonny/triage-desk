"""The nurse's one-screen board. Reads the ledger directly (this is the human side)."""
from __future__ import annotations

import io
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import ledger, rules
from . import scale as sc

STYLE = {"RED": "bold white on red", "ORANGE": "bold black on dark_orange", "YELLOW": "bold black on yellow",
         "GREEN": "bold white on green4", None: "dim"}
SEV_ORDER = {"ESCALATE": 0, "WATCH": 1, "CONFIRM": 2, "RECHECK": 3, "INFO": 4}


def chip(colour: str | None) -> Text:
    return Text(f" {colour or 'N/A':^6} ", style=STYLE.get(colour, "dim"))


def _action_line(a: dict[str, Any]) -> str:
    p = a["payload"] or {}
    k = a["kind"]
    if k == "change_colour":
        prov = " (PROVISIONAL: uses note values not yet confirmed)" if p.get("provisional") else ""
        return f"{p.get('from')} → {p.get('to')}{prov}. {p.get('reason', '')}"
    if k == "record_observation":
        ev = p.get("evidence", {})
        vals = ", ".join(f"{f}={v} «{ev.get(f, '')}»" for f, v in (p.get("vitals") or {}).items())
        signs = ", ".join(f"{s} «{p.get('sign_evidence', {}).get(s, '')}»" for s in p.get("signs") or [])
        rej = f"  REJECTED: {len(p['rejected'])}" if p.get("rejected") else ""
        return f"Confirm from note: {vals}{'; ' + signs if signs else ''}{rej}"
    if k == "reorder_queue":
        return a["rationale"] or ""
    if k == "draft_referral":
        miss = p.get("missing_fields") or []
        return f"Referral DRAFT to {p.get('receiving')}" + (f" (missing: {', '.join(miss)})" if miss else "")
    if k == "schedule_recheck":
        return p.get("message", "")
    return p.get("message") or a["rationale"] or ""


def build(con, now: str) -> Group:
    pts = ledger.patients(con)
    pts = [p for p in pts if p["arrival_time"] <= now]
    t = Table(title=f"Triage Desk · waiting room at {now[11:16]} · {now[:10]} (SYNTHETIC DATA)", expand=True,
              show_lines=False, header_style="bold")
    for col, kw in [("#", {"width": 2}), ("ID", {}), ("Patient", {}), ("Wait", {"justify": "right"}),
                    ("Triaged", {}), ("Now", {}), ("TEWS", {"justify": "right"}), ("Recheck", {}), ("Why / unknown", {"ratio": 1})]:
        t.add_column(col, **kw)
    waiting = [p for p in pts if not p["seen_at"]]
    for p in waiting:
        obs = ledger.observations(con, p["id"], include_proposed=True, until=now)
        res = rules.flag_deterioration(p, obs, now)
        cur = res.get("current") or {}
        waited = round(rules.minutes_between(p["arrival_time"], now))
        tgt = sc.target_minutes(p["assigned_colour"])
        wait = Text(f"{waited}m", style="bold red" if waited > tgt else "")
        if cur.get("out_of_scope"):
            now_chip, tews, why = Text(" OUT OF SCOPE ", style="bold magenta"), "-", cur["reason"][:70]
        else:
            now_chip = chip(cur.get("colour"))
            tews = str(cur.get("tews")) if cur.get("complete") else f"≥{cur.get('tews_min')}"
            why = cur.get("reason", "")
            if any(o["status"] == "proposed" for o in obs):
                why = "[unconfirmed note values] " + why
        rc = rules.recheck(res.get("effective") or p["assigned_colour"], obs[-1]["observed_at"], now) if obs else None
        rct = Text("-") if rc is None else Text(rc["due_at"][11:16] + (" OVERDUE" if rc["overdue"] else ""),
                                                 style="bold red" if rc["overdue"] else "")
        t.add_row(str(p["queue_position"]), p["id"], f"{p['name']} ({int(p['age'])}{p['sex']})", wait,
                  chip(p["assigned_colour"]), now_chip, tews, rct, why)
    seen = [p for p in pts if p["seen_at"]]

    acts = ledger.actions(con)
    pending = sorted([a for a in acts if a["status"] == "PROPOSED"],
                     key=lambda a: (SEV_ORDER.get(a["severity"], 9), a["created_at"]))
    alerts = [a for a in pending if a["severity"] in ("ESCALATE", "WATCH", "CONFIRM")]
    rechecks = [a for a in pending if a["severity"] == "RECHECK"]
    fyi = [a for a in pending if a["severity"] == "INFO"]
    d = Table(show_header=True, header_style="bold", expand=True, box=None)
    d.add_column("Action")
    d.add_column("Sev")
    d.add_column("Pt")
    d.add_column("Proposal", ratio=1)
    for a in alerts:
        sev_style = {"ESCALATE": "bold red", "WATCH": "yellow", "CONFIRM": "cyan"}.get(a["severity"], "")
        d.add_row(a["id"], Text(a["severity"], style=sev_style), a["patient_id"] or "-",
                  f"[{a['kind']}] {_action_line(a)}")
    parts: list[Any] = [t]
    parts.append(Panel(d if alerts else Text("Nothing needs your decision."), title=f"Needs your decision ({len(alerts)})",
                       subtitle="approve <id> · edit <id> · veto <id> <reason>", border_style="red" if alerts else "green"))
    if rechecks:
        parts.append(Text("Recheck due: " + ", ".join(f"{a['patient_id']} ({(a['payload'] or {}).get('overdue_minutes', '?')}m over)"
                                                      for a in rechecks) + "   (ack with approve <id>)", style="yellow"))
    if fyi:
        parts.append(Text("FYI (no action needed): " + " | ".join(f"{a['patient_id']}: {_action_line(a)}" for a in fyi), style="dim"))
    vetoed = [a for a in acts if a["status"] == "VETOED"]
    if vetoed:
        parts.append(Text("Vetoed (kept visible): " + " | ".join(
            f"{a['id']} {a['patient_id'] or 'queue'} [{a['kind']}] by {a['decided_by']}: {a['decision_reason']}" for a in vetoed),
            style="dim italic"))
    refs = ledger.referrals(con)
    if refs:
        parts.append(Text("Referrals: " + " | ".join(f"{r['id']} {r['patient_id']} {r['status']}" +
                                                     (f" by {r['signed_by']}" if r['signed_by'] else "") for r in refs)))
    if seen:
        parts.append(Text("Seen: " + ", ".join(f"{p['id']} {p['assigned_colour']} at {p['seen_at'][11:16]}" for p in seen), style="dim"))
    return Group(*parts)


def render_text(con, now: str, width: int = 160) -> str:
    buf = io.StringIO()
    Console(file=buf, width=width, height=400, color_system=None, legacy_windows=False).print(build(con, now))
    return buf.getvalue()


def print_board(con, now: str, console: Console | None = None) -> None:
    (console or Console()).print(build(con, now))
