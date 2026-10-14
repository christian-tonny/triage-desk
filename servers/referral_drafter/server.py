"""referral-drafter MCP server: builds an SBAR referral letter from recorded facts.

Output is always a DRAFT. It is templated (no model), quotes nurse notes
verbatim, and lists what is missing rather than filling gaps. Only a nurse or
doctor can sign it, in the console.
"""
from __future__ import annotations

import hashlib
from typing import Any

from mcp.server.fastmcp import FastMCP

from triage_desk import scale as sc

mcp = FastMCP("referral-drafter")

REQUIRED = ["name", "age", "sex", "complaint", "arrival_time", "assigned_colour"]


def _age(a: Any) -> str:
    return str(int(a)) if isinstance(a, (int, float)) else str(a or "?")


def _vitals_line(score: dict[str, Any]) -> str:
    pic = (score.get("picture") or {}).get("vitals", {})
    at = (score.get("picture") or {}).get("measured_at", {})
    order = [("rr", "RR", "/min"), ("hr", "HR", "/min"), ("sbp", "SBP", " mmHg"), ("temp_c", "Temp", " °C"),
             ("spo2", "SpO2", "%"), ("avpu", "AVPU", ""), ("mobility", "Mobility", ""), ("glucose_mmol", "Glucose", " mmol/L")]
    parts = []
    for k, label, unit in order:
        if k in pic:
            parts.append(f"{label} {pic[k]}{unit} (at {str(at.get(k, '?'))[11:16]})")
        else:
            parts.append(f"{label} NOT MEASURED")
    return "; ".join(parts)


@mcp.tool()
def draft_referral(patient: dict[str, Any], score: dict[str, Any], flags: list[dict[str, Any]], notes: list[str],
                   receiving_facility: str, sending_facility: str, now: str, reason: str = "") -> dict[str, Any]:
    """Draft an SBAR referral. Returns status DRAFT, the letter body, and missing fields."""
    missing = [f for f in REQUIRED if not patient.get(f)]
    missing += [f"vital:{m}" for m in score.get("missing", [])]
    drivers = score.get("top_drivers") or []
    lines = [
        "*** DRAFT - NOT SENT - REQUIRES SIGNATURE ***",
        f"From: {sending_facility}",
        f"To: {receiving_facility}",
        f"Prepared: {now[:16].replace('T', ' ')} (auto-drafted by Triage Desk; facts below are from the ledger)",
        "",
        "S - SITUATION",
        f"{patient.get('name', '?')}, {_age(patient.get('age'))} y, {patient.get('sex', '?')}. "
        f"Presenting complaint: {patient.get('complaint', '?')}.",
        f"Current SATS priority: {score.get('colour')} (assigned at desk: {patient.get('assigned_colour')}).",
        f"Reason for referral: {reason or 'deterioration while waiting; needs care not available here'}.",
        "",
        "B - BACKGROUND",
        f"Arrived {str(patient.get('arrival_time', '?'))[11:16]}. History: {patient.get('history') or 'none recorded'}.",
        "Nurse notes (verbatim):",
        *[f"  - \"{n}\"" for n in notes[-5:]],
        "",
        "A - ASSESSMENT",
        f"Latest values: {_vitals_line(score)}.",
        f"TEWS: {score.get('tews') if score.get('complete') else str(score.get('tews_min')) + ' (incomplete)'}.",
        "Priority drivers: " + ("; ".join(f"{d['label']} [{d['cite']}]" for d in drivers) or "none"),
        "Watcher flags: " + ("; ".join(f"{f['rule']}: {f['message']}" for f in flags) or "none"),
        "",
        "R - RECOMMENDATION / REQUEST",
        "Request acceptance for urgent review. Clinician to complete: treatment given, escort, transport.",
        "",
        "Signed: ______________________  (name, cadre)   Time: ______",
    ]
    if missing:
        lines.insert(4, f"MISSING (complete before signing): {', '.join(missing)}")
    body = "\n".join(lines)
    return {"status": "DRAFT", "body": body, "missing_fields": missing,
            "sha256": hashlib.sha256(body.encode()).hexdigest()[:16],
            "colour_cite": sc.scale()["priorities"].get(score.get("colour") or "RED", {}).get("cite")}


@mcp.tool()
def required_fields() -> dict[str, Any]:
    """Fields a referral needs before it can be signed."""
    return {"required": REQUIRED, "format": "SBAR"}


if __name__ == "__main__":
    mcp.run()
