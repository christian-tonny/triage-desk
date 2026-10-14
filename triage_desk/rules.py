"""Deterioration watch: deterministic rules over a patient's observation history.

The watcher only raises flags and proposals. It never changes a colour, and it
never proposes a downgrade (an improving patient gets an INFO note for the
nurse; lowering priority is always a human decision).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from . import scale as sc

STALE_MINUTES = 120


def parse_t(t: str | datetime) -> datetime:
    return t if isinstance(t, datetime) else datetime.fromisoformat(t)


def minutes_between(a: str | datetime, b: str | datetime) -> float:
    return (parse_t(b) - parse_t(a)).total_seconds() / 60.0


def merged_picture(observations: list[dict[str, Any]], now: str) -> dict[str, Any]:
    """Latest value of each parameter, with when it was measured."""
    vitals: dict[str, Any] = {}
    measured_at: dict[str, str] = {}
    discs: list[str] = []
    for ob in sorted(observations, key=lambda o: o["observed_at"]):
        for k, v in (ob.get("vitals") or {}).items():
            if v is not None:
                vitals[k] = v
                measured_at[k] = ob["observed_at"]
        for d in ob.get("discriminators") or []:
            if d not in discs:
                discs.append(d)
        for d in ob.get("resolved_discriminators") or []:
            if d in discs:
                discs.remove(d)
    stale = [k for k, t in measured_at.items() if minutes_between(t, now) > STALE_MINUTES]
    return {"vitals": vitals, "measured_at": measured_at, "discriminators": discs, "stale": stale}


def score_history(patient: dict[str, Any], observations: list[dict[str, Any]], now: str) -> dict[str, Any]:
    pic = merged_picture(observations, now)
    res = sc.score(pic["vitals"], pic["discriminators"], age_years=patient.get("age"),
                   pregnant=bool(patient.get("pregnant")), diabetic=bool(patient.get("diabetic")))
    res["picture"] = pic
    return res


def recheck(colour: str, last_obs_at: str, now: str) -> dict[str, Any]:
    pol = sc.policy()["recheck_minutes"][colour]
    due = parse_t(last_obs_at) + timedelta(minutes=pol["value"])
    return {
        "colour": colour,
        "interval_minutes": pol["value"],
        "due_at": due.isoformat(timespec="minutes"),
        "overdue_minutes": max(0, round(minutes_between(due, now))),
        "overdue": parse_t(now) > due,
        "cite": pol["cite"],
    }


def flag_deterioration(patient: dict[str, Any], observations: list[dict[str, Any]], now: str) -> dict[str, Any]:
    """Compare the patient's current picture with the colour they were given.

    Returns flags with severity ESCALATE (nurse should act now), WATCH (look
    again soon) or INFO (no action proposed). Each flag has a dedupe_key so the
    same finding on the same data is raised once, not every cycle.
    """
    assigned = patient["assigned_colour"]
    pol = sc.policy()
    flags: list[dict[str, Any]] = []
    if not observations:
        flags.append({"rule": "NO_OBSERVATIONS", "severity": "WATCH",
                      "message": "No vitals recorded since arrival. Triage this patient.",
                      "cite": "SATS-2012 Fig.4: measure vital signs and calculate TEWS",
                      "dedupe_key": f"{patient['id']}:NO_OBS"})
        return {"patient_id": patient["id"], "flags": flags, "current": None, "baseline": None}

    obs_sorted = sorted(observations, key=lambda o: o["observed_at"])
    latest_obs = obs_sorted[-1]
    baseline = score_history(patient, obs_sorted[:1], now)
    current = score_history(patient, obs_sorted, now)
    if current.get("out_of_scope"):
        flags.append({"rule": "OUT_OF_SCOPE", "severity": "WATCH", "message": current["reason"],
                      "cite": current["cite"], "dedupe_key": f"{patient['id']}:OOS"})
        return {"patient_id": patient["id"], "flags": flags, "current": current, "baseline": None}

    key_obs = latest_obs["id"]
    cur = current["colour"]
    seen = bool(patient.get("seen_at"))

    if sc.rank(cur) > sc.rank(assigned):
        flags.append({
            "rule": "COLOUR_UP", "severity": "ESCALATE",
            "message": f"Now {cur} (was triaged {assigned}). {current['reason']}",
            "proposed_colour": cur, "drivers": current["top_drivers"],
            "cite": "; ".join(dict.fromkeys(d["cite"] for d in current["top_drivers"])),
            "dedupe_key": f"{patient['id']}:UP:{cur}:{key_obs}",
        })
    elif sc.rank(cur) < sc.rank(assigned):
        flags.append({
            "rule": "IMPROVED", "severity": "INFO",
            "message": f"Current data scores {cur}, below assigned {assigned}. No downgrade proposed; nurse decides.",
            "cite": "Design rule: the agent never lowers a priority",
            "dedupe_key": f"{patient['id']}:IMPROVED:{key_obs}",
        })

    rise = current["tews_min"] - baseline["tews_min"]
    if rise >= pol["tews_rise_watch"]["value"] and not any(f["rule"] == "COLOUR_UP" for f in flags):
        flags.append({
            "rule": "TEWS_RISE", "severity": "WATCH",
            "message": f"TEWS rose {baseline['tews_min']} → {current['tews_min']} since triage; colour unchanged ({cur}).",
            "cite": pol["tews_rise_watch"]["cite"],
            "dedupe_key": f"{patient['id']}:RISE:{current['tews_min']}:{key_obs}",
        })

    if pol["single_parameter_3"]["value"]:
        base3 = {b["param"] for b in baseline.get("breakdown", []) if b["score"] == 3}
        new3 = [b for b in current["breakdown"] if b["score"] == 3 and b["param"] not in base3]
        if new3 and not any(f["rule"] == "COLOUR_UP" for f in flags):
            flags.append({
                "rule": "SINGLE_PARAM_3", "severity": "WATCH",
                "message": "New maximum-score parameter: " + ", ".join(f"{sc.PARAM_LABEL[b['param']]} {b['value']}" for b in new3),
                "cite": pol["single_parameter_3"]["cite"],
                "dedupe_key": f"{patient['id']}:P3:{'-'.join(b['param'] for b in new3)}:{key_obs}",
            })

    effective = sc.higher(assigned, cur)
    if not seen:
        waited = minutes_between(patient["arrival_time"], now)
        if effective == "RED" and any(f["rule"] == "COLOUR_UP" for f in flags):
            for f in flags:
                if f["rule"] == "COLOUR_UP":
                    f["message"] = f["message"].rstrip(". ") + ". Move to resus and hand over to senior now."
        elif effective == "RED":
            flags.append({
                "rule": "RED_IN_QUEUE", "severity": "ESCALATE",
                "message": "Red patient still in the waiting queue. Move to resus and hand over to senior.",
                "cite": sc.scale()["priorities"]["RED"]["cite"],
                "dedupe_key": f"{patient['id']}:REDQ",
            })
        else:
            tgt = sc.target_minutes(effective)
            if waited > tgt:
                flags.append({
                    "rule": "TARGET_BREACH", "severity": "ESCALATE" if effective == "ORANGE" else "INFO",
                    "message": f"Waited {round(waited)} min; {effective} target is {tgt} min.",
                    "cite": sc.scale()["priorities"][effective]["cite"],
                    "dedupe_key": f"{patient['id']}:BREACH:{effective}",
                })
        rc = recheck(effective, latest_obs["observed_at"], now)
        if rc["overdue"]:
            flags.append({
                "rule": "RECHECK_OVERDUE", "severity": "WATCH",
                "message": f"Recheck overdue by {rc['overdue_minutes']} min (every {rc['interval_minutes']} min for {effective}).",
                "cite": rc["cite"],
                "dedupe_key": f"{patient['id']}:RECHECK:{latest_obs['id']}",
            })

    if not current["complete"] and current["possible_colour"] != cur:
        flags.append({
            "rule": "INCOMPLETE", "severity": "WATCH",
            "message": f"Missing {', '.join(sc.PARAM_LABEL[m] for m in current['missing'])}; could be {current['possible_colour']} once measured.",
            "cite": "SATS-2012 Fig.4: measure vital signs and calculate TEWS",
            "dedupe_key": f"{patient['id']}:INCOMPLETE:{'-'.join(current['missing'])}:{key_obs}",
        })
    stale = [p for p in current["picture"]["stale"] if p in sc.TEWS_PARAMS and p not in ("mobility", "trauma")]
    if stale and not seen:
        flags.append({
            "rule": "STALE_VITALS", "severity": "INFO",
            "message": f"{', '.join(sc.PARAM_LABEL[p] for p in stale)} last measured over {STALE_MINUTES} min ago.",
            "cite": "LOCAL DEFAULT: values older than 120 min are shown as stale",
            "dedupe_key": f"{patient['id']}:STALE:{'-'.join(stale)}:{key_obs}",
        })

    return {"patient_id": patient["id"], "flags": flags, "current": current, "baseline": baseline,
            "assigned": assigned, "effective": effective}


def rerank(patients: list[dict[str, Any]], now: str) -> list[dict[str, Any]]:
    """Proposed seeing order: colour, then minutes left to target, then arrival."""
    rows = []
    for p in patients:
        if p.get("seen_at"):
            continue
        colour = p.get("effective_colour") or p["assigned_colour"]
        waited = minutes_between(p["arrival_time"], now)
        left = sc.target_minutes(colour) - waited
        rows.append({"patient_id": p["id"], "colour": colour, "minutes_to_target": round(left),
                     "waited": round(waited), "arrival_time": p["arrival_time"]})
    rows.sort(key=lambda r: (-sc.rank(r["colour"]), r["minutes_to_target"], r["arrival_time"]))
    for i, r in enumerate(rows, 1):
        r["position"] = i
    return rows
