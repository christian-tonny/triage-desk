"""Deterministic SATS adult scoring.

The LLM never calls into the priority decision. Every number here comes from
data/scale/sats_adult.json (SATS 2012) or data/scale/local_policy.json (site
policy, flagged for sign-off), and every output line carries its citation.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from . import config

COLOURS = ["GREEN", "YELLOW", "ORANGE", "RED"]
TEWS_PARAMS = ["mobility", "rr", "hr", "sbp", "temp_c", "avpu", "trauma"]
TEWS_MAX = {"mobility": 2, "rr": 3, "hr": 3, "sbp": 3, "temp_c": 2, "avpu": 3, "trauma": 1}
PARAM_LABEL = {
    "mobility": "Mobility", "rr": "RR", "hr": "HR", "sbp": "SBP",
    "temp_c": "Temp", "avpu": "AVPU", "trauma": "Trauma",
}
VALID_MOBILITY = {"walking", "with_help", "stretcher_immobile"}
VALID_AVPU = {"A", "C", "V", "P", "U"}


@lru_cache(maxsize=1)
def scale() -> dict[str, Any]:
    return json.loads(config.SCALE_FILE.read_text())


@lru_cache(maxsize=1)
def policy() -> dict[str, Any]:
    return json.loads(config.POLICY_FILE.read_text())


def rank(colour: str | None) -> int:
    return COLOURS.index(colour) + 1 if colour in COLOURS else 0


def higher(a: str | None, b: str | None) -> str | None:
    return a if rank(a) >= rank(b) else b


def target_minutes(colour: str) -> int:
    return scale()["priorities"][colour]["target_minutes"]


def discriminator_ids() -> list[str]:
    return [k for k in scale()["discriminators"] if not k.startswith("_")]


def _band(table: list[dict], value: float) -> int:
    for row in table:
        if value < row["lt"]:
            return row["score"]
    raise ValueError(f"value {value} outside table")


def score_param(param: str, value: Any) -> int:
    t = scale()["tews"]
    if param == "mobility":
        return t["mobility"][value]
    if param == "avpu":
        return t["avpu"][value]
    if param == "trauma":
        return t["trauma"]["true" if value else "false"]
    return _band(t[param], float(value))


def _clean_vitals(vitals: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop values that cannot be scored and say why, instead of guessing."""
    clean: dict[str, Any] = {}
    problems: list[str] = []
    for k, v in (vitals or {}).items():
        if v is None or v == "":
            continue
        if k == "mobility" and v not in VALID_MOBILITY:
            problems.append(f"mobility '{v}' not one of {sorted(VALID_MOBILITY)}")
            continue
        if k == "avpu" and v not in VALID_AVPU:
            problems.append(f"AVPU '{v}' not one of A/C/V/P/U")
            continue
        clean[k] = v
    return clean, problems


def tews_band(tews: int) -> dict[str, Any]:
    for b in scale()["tews_bands"]:
        if b["min"] <= tews <= b["max"]:
            return b
    raise ValueError(tews)


def score(
    vitals: dict[str, Any],
    discriminators: list[str] | None = None,
    age_years: float | None = None,
    pregnant: bool = False,
    diabetic: bool = False,
) -> dict[str, Any]:
    """Return the SATS colour with every driver and its citation.

    Missing TEWS parameters are never assumed normal silently: the result
    carries tews_min (missing = 0) and tews_max (missing = worst case), and the
    colour is computed from what is known, with possible_colour showing how bad
    it could be.
    """
    s = scale()
    pol = policy()
    if age_years is not None and age_years <= 12:
        return {
            "out_of_scope": True,
            "colour": None,
            "reason": "Patient is 12 or younger. Only the adult SATS chart is implemented; use the paediatric chart and nurse judgement.",
            "cite": "SATS-2012 Fig.14: adult TEWS applies to older than 12 years / taller than 150 cm",
        }

    vitals, problems = _clean_vitals(vitals)
    discs = list(dict.fromkeys(discriminators or []))
    unknown_discs = [d for d in discs if d not in s["discriminators"] or d.startswith("_")]
    discs = [d for d in discs if d not in unknown_discs]
    problems += [f"unknown discriminator '{d}' ignored" for d in unknown_discs]

    drivers: list[dict[str, Any]] = []
    derived: list[tuple[str, str]] = []

    if vitals.get("avpu") in {"C", "V", "P", "U"} and "reduced_loc_confused" not in discs:
        derived.append(("reduced_loc_confused", f"AVPU = {vitals['avpu']}"))
    g = vitals.get("glucose_mmol")
    if g is not None:
        g = float(g)
        if g < 3 and "hypoglycaemia_lt3" not in discs:
            derived.append(("hypoglycaemia_lt3", f"glucose {g} mmol/L"))
        elif g > 11 and vitals.get("ketonuria") is True and "diabetic_gt11_ketonuria" not in discs:
            derived.append(("diabetic_gt11_ketonuria", f"glucose {g} mmol/L with ketonuria"))
        elif g > 17 and vitals.get("ketonuria") is False and "diabetic_gt17_no_ketonuria" not in discs:
            derived.append(("diabetic_gt17_no_ketonuria", f"glucose {g} mmol/L, no ketonuria"))
    if pregnant and "abdominal_pain" in discs and "pregnancy_abdominal_pain" not in discs:
        derived.append(("pregnancy_abdominal_pain", "pregnant + abdominal pain"))

    level_cite = {
        "RED": s["discriminators"]["_cite_emergency"],
        "ORANGE": s["discriminators"]["_cite_very_urgent"],
        "YELLOW": s["discriminators"]["_cite_urgent"],
    }
    for d in discs:
        info = s["discriminators"][d]
        drivers.append({"kind": "clinical_sign", "id": d, "label": info["label"], "colour": info["level"],
                        "evidence": "reported", "cite": level_cite[info["level"]]})
    for d, why in derived:
        info = s["discriminators"][d]
        drivers.append({"kind": "clinical_sign", "id": d, "label": info["label"], "colour": info["level"],
                        "evidence": why, "cite": level_cite[info["level"]]})

    spo2 = vitals.get("spo2")
    if spo2 is not None and float(spo2) < pol["spo2_red_below"]["value"]:
        drivers.append({"kind": "local_policy", "id": "spo2_low", "label": f"SpO2 {spo2}% < {pol['spo2_red_below']['value']}%",
                        "colour": "RED", "evidence": f"SpO2 {spo2}%", "cite": pol["spo2_red_below"]["cite"]})

    breakdown = []
    tews_min = 0
    tews_max = 0
    missing = []
    for p in TEWS_PARAMS:
        if p in vitals:
            sc = score_param(p, vitals[p])
            tews_min += sc
            tews_max += sc
            breakdown.append({"param": p, "value": vitals[p], "score": sc})
        else:
            missing.append(p)
            tews_max += TEWS_MAX[p]
    # Trauma defaults to "No" only when the nurse recorded a complaint without injury;
    # we still list it as missing so the nurse sees it was not asked.
    complete = not missing

    band = tews_band(tews_min)
    drivers.append({"kind": "tews", "id": "tews", "label": f"TEWS {tews_min}" + ("" if complete else f" (known so far; up to {tews_max})"),
                    "colour": band["colour"], "evidence": ", ".join(f"{PARAM_LABEL[b['param']]} {b['value']}→{b['score']}" for b in breakdown),
                    "cite": band["cite"]})

    colour = "GREEN"
    for d in drivers:
        colour = higher(colour, d["colour"])
    possible = higher(colour, tews_band(min(tews_max, 99))["colour"]) if not complete else colour

    emergency = any(d["colour"] == "RED" and d["kind"] == "clinical_sign" for d in drivers)
    investigations = []
    ai = s["additional_investigations"]
    if "rr" in vitals and score_param("rr", vitals["rr"]) >= 1 and spo2 is None:
        investigations.append(ai["rr_scores_1_or_more"])
    if any(d["id"] == "reduced_loc_confused" for d in drivers) and g is None:
        investigations.append(ai["reduced_loc"])
    if "chest_pain" in discs:
        investigations.append(ai["chest_pain"])
    if diabetic and g is None:
        investigations.append(ai["history_of_diabetes"])
    if missing:
        investigations.append("Measure missing TEWS parameters: " + ", ".join(PARAM_LABEL[m] for m in missing))

    top = [d for d in drivers if d["colour"] == colour]
    others = [d for d in drivers if d["colour"] != colour and rank(d["colour"]) > 1]
    return {
        "out_of_scope": False,
        "colour": colour,
        "target_minutes": target_minutes(colour),
        "target_cite": s["priorities"][colour]["cite"],
        "tews": tews_min if complete else None,
        "tews_min": tews_min,
        "tews_max": tews_max,
        "complete": complete,
        "missing": missing,
        "possible_colour": possible,
        "breakdown": breakdown,
        "drivers": drivers,
        "top_drivers": top,
        "emergency_sign": emergency,
        "additional_investigations": investigations,
        "problems": problems,
        "reason": explain(colour, top, complete, missing, possible, others),
    }


def explain(colour: str, top: list[dict], complete: bool, missing: list[str], possible: str,
            others: list[dict] | None = None) -> str:
    """One-line reason, built from facts only (no LLM)."""
    parts = []
    for d in top:
        if d["kind"] == "tews":
            parts.append(f"{d['label']} [{d['evidence']}]" if d["evidence"] else d["label"])
        else:
            parts.append(f"{d['label']} ({d['evidence']})")
    line = f"{colour}: " + "; ".join(parts)
    if others:
        line += ". Also: " + "; ".join(f"{d['label']} → {d['colour']}" for d in others)
    if not complete:
        line += f". UNKNOWN: {', '.join(PARAM_LABEL[m] for m in missing)}"
        if possible != colour:
            line += f" - could be {possible} once measured"
    return line
