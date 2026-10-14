"""Timed triage drill: does the tool help a nurse decide faster *and* better?

manual   - read the vignette, pick a colour.
assisted - same, but the scorer's proposal (colour, TEWS breakdown, drivers,
           unknowns) is shown first; the nurse still picks the colour.

Results go to runs/drill/<participant>.jsonl. Use a participant code, never a name.
Run both modes on different vignette halves (the order is shuffled per
participant) so nobody sees the same case twice.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel

from . import config
from . import scale as sc

C = Console()
KEYS = {"r": "RED", "o": "ORANGE", "y": "YELLOW", "g": "GREEN"}


def _assist(v: dict) -> str:
    r = sc.score(v["vitals"], v.get("signs", []), v.get("age"), bool(v.get("pregnant")), bool(v.get("diabetic")))
    lines = [f"Proposed: {r['colour']}  (TEWS {r['tews'] if r['complete'] else '≥' + str(r['tews_min'])})", r["reason"]]
    if r["additional_investigations"]:
        lines.append("Also: " + "; ".join(r["additional_investigations"]))
    return "\n".join(lines)


def run_drill(mode: str, participant: str, limit: int = 8) -> None:
    vs = json.loads((config.FIXTURES / "drill_vignettes.json").read_text())["vignettes"]
    rng = random.Random(participant)
    rng.shuffle(vs)
    half = len(vs) // 2
    vs = vs[:half] if mode == "manual" else vs[half:]
    vs = vs[:limit]
    out = config.RUNS / "drill"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{participant}.jsonl"
    C.print(Panel(f"Mode: {mode}. {len(vs)} cases. Type r / o / y / g then Enter. Timing starts when the case appears.",
                  title="Triage drill (synthetic cases)"))
    correct = 0
    times = []
    for v in vs:
        body = v["text"] + (("\n\n" + _assist(v)) if mode == "assisted" else "")
        C.print(Panel(body, title=v["id"]))
        t0 = time.perf_counter()
        ans = ""
        while ans not in KEYS:
            ans = input("colour [r/o/y/g]> ").strip().lower()[:1]
        secs = time.perf_counter() - t0
        ok = KEYS[ans] == v["gold"]
        correct += ok
        times.append(secs)
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "participant": participant, "mode": mode, "vignette": v["id"],
               "answer": KEYS[ans], "gold": v["gold"], "correct": ok, "seconds": round(secs, 2),
               "under_triage": sc.rank(KEYS[ans]) < sc.rank(v["gold"])}
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    times.sort()
    C.print(f"{participant} {mode}: {correct}/{len(vs)} correct, median {times[len(times) // 2]:.1f} s per case. Saved to {path}")
