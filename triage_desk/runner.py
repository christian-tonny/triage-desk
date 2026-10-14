"""Run the agent: one cycle, or a whole scripted scenario into runs/<run_id>/."""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import board, config, ledger
from .agent import run_cycle
from .llm import Extractor, get_extractor
from .mcp_hub import McpHub
from .toollog import ToolLog

SCRIPTED = "Nurse Uwera (scripted-nurse)"
# Assumptions for the cost estimate; override with env vars. Stated in EVALS.md.
DEVICE_WATTS = float(os.environ.get("TRIAGE_DEVICE_WATTS", "40"))
USD_PER_KWH = float(os.environ.get("TRIAGE_USD_PER_KWH", "0.20"))


def new_run_id(label: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{label}"


def cost_block(log: ToolLog, wall_seconds: float) -> dict[str, Any]:
    kwh = DEVICE_WATTS * log.llm_seconds / 3600 / 1000
    return {
        "llm_calls": log.llm_calls, "tokens_in": log.llm_tokens_in, "tokens_out": log.llm_tokens_out,
        "llm_seconds": round(log.llm_seconds, 1), "wall_seconds": round(wall_seconds, 1),
        "mcp_tool_calls": log.tool_calls, "mcp_tool_errors": log.tool_errors,
        "energy_kwh_est": round(kwh, 6), "usd_est": round(kwh * USD_PER_KWH, 6),
        "assumptions": f"{DEVICE_WATTS} W while the model runs, USD {USD_PER_KWH}/kWh; no API fees (local model)",
    }


def _match(con, match: dict[str, Any]) -> dict[str, Any] | None:
    cands = [a for a in ledger.actions(con, status="PROPOSED")
             if all(a.get(k) == v for k, v in match.items())]
    return cands[-1] if cands else None


async def run_scenario(scenario_file: Path, backend: str | None = None, label: str | None = None,
                       extractor: Extractor | None = None, runs_dir: Path | None = None,
                       borrowed: bool | None = None, quiet: bool = False) -> dict[str, Any]:
    scenario = json.loads(Path(scenario_file).read_text())
    fixture = json.loads((Path(scenario_file).parent / scenario["fixture"]).read_text())
    extractor = extractor or get_extractor(backend)
    run_id = new_run_id(label or extractor.backend)
    run_dir = Path(runs_dir or config.RUNS) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    db = run_dir / "ledger.db"
    os.environ["TRIAGE_DB"] = str(db)
    ledger.reset(db)
    ledger.load_fixture(fixture, db)
    log = ToolLog(run_dir, run_id)
    log.event("run_start", {"scenario": scenario["name"], "backend": extractor.backend, "model": extractor.model,
                            "temperature": getattr(extractor, "temperature", None), "seed": getattr(extractor, "seed", None)})
    t0 = time.perf_counter()
    timeline: list[dict[str, Any]] = []
    async with McpHub(log, borrowed=borrowed) as hub:
        for ev in scenario["events"]:
            at = ev["at"]
            entry: dict[str, Any] = {"at": at, "type": ev["type"]}
            if ev["type"] == "cycle":
                st = await run_cycle(hub, extractor, log, at, run_id)
                entry["new_proposals"] = [p for p in st["proposals"] if not p.get("deduplicated")]
                entry["extractions"] = st["extractions"]
                with ledger.connect(db) as con:
                    snap = board.render_text(con, at)
                (run_dir / f"board_{at[11:16].replace(':', '')}.txt").write_text(snap)
                if not quiet:
                    print(snap)
            else:
                with ledger.connect(db) as con:
                    if ev["type"] == "note":
                        entry["note_id"] = ledger.add_note(con, ev["patient_id"], ev["text"], at, author="scripted-nurse")
                    elif ev["type"] == "vitals":
                        entry["obs_id"] = ledger.record_nurse_observation(con, ev["patient_id"], at, ev["vitals"],
                                                                          ev.get("signs", []), ev.get("by", SCRIPTED))
                    elif ev["type"] == "seen":
                        ledger.mark_seen(con, ev["patient_id"], at, SCRIPTED)
                    elif ev["type"] == "decide":
                        a = _match(con, ev["match"])
                        if a is None:
                            entry["error"] = f"no PROPOSED action matching {ev['match']}"
                        else:
                            entry["action_id"] = a["id"]
                            entry["result"] = ledger.decide(con, a["id"], ev["decision"], SCRIPTED,
                                                            ev.get("reason", ""), at=at)
                    elif ev["type"] == "sign_referral":
                        refs = [r for r in ledger.referrals(con) if r["patient_id"] == ev["patient_id"] and r["status"] == "DRAFT"]
                        if refs:
                            entry["result"] = ledger.sign_referral(con, refs[-1]["id"], ev.get("by", SCRIPTED), at=at)
                        else:
                            entry["error"] = "no draft referral to sign"
                log.event("scenario_event", entry)
            timeline.append(entry)
        borrowed_used = "filesystem" in hub.sessions
    wall = time.perf_counter() - t0
    with ledger.connect(db) as con:
        acts = ledger.actions(con)
        final_board = board.render_text(con, scenario["events"][-1]["at"])
    (run_dir / "board_final.txt").write_text(final_board)
    summary = {
        "run_id": run_id, "scenario": scenario["name"], "backend": extractor.backend, "model": extractor.model,
        "temperature": getattr(extractor, "temperature", None), "seed": getattr(extractor, "seed", None),
        "machine": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
        "borrowed_server": "filesystem (connected)" if borrowed_used else "off",
        "cost": cost_block(log, wall),
        "actions": [{k: a[k] for k in ("id", "created_at", "kind", "patient_id", "severity", "status", "dedupe_key",
                                       "decided_by", "decision_reason")} | {"payload": _slim(a)} for a in acts],
        "timeline": timeline,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _slim(a: dict[str, Any]) -> dict[str, Any]:
    p = a["payload"] or {}
    keep = {k: p[k] for k in ("from", "to", "provisional", "rule", "message", "vitals", "signs", "evidence",
                              "rejected", "order", "due_at", "receiving", "missing_fields") if k in p}
    return keep


async def one_cycle(now: str, backend: str | None = None, borrowed: bool | None = None) -> dict[str, Any]:
    extractor = get_extractor(backend)
    run_id = new_run_id("cycle")
    log = ToolLog(config.RUNS / "console", run_id)
    async with McpHub(log, borrowed=borrowed) as hub:
        st = await run_cycle(hub, extractor, log, now, run_id)
    return {"run_id": run_id, "new_proposals": [p for p in st["proposals"] if not p.get("deduplicated")],
            "extractions": st["extractions"], "cost": cost_block(log, 0)}
