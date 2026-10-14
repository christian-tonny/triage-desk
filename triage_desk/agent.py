"""The Triage Desk agent: a LangGraph graph that acts as an MCP client.

Graph:  load ─┬─> intake ─> assess ─> rerank ─┬─> refer ─> finish
              └──────────────> assess          └──────────> finish

The language model is used in one node only (intake), to turn nurse notes into
quoted, validated observations. Colours, flags, order and referral text all come
from deterministic MCP tools. Every write is a proposal for the nurse.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import scale as sc
from .llm import Extractor
from .mcp_hub import McpHub, ToolCallError
from .toollog import ToolLog

SENDING = "Demo District Hospital - Outpatient Department (synthetic)"


class State(TypedDict, total=False):
    run_id: str
    now: str
    queue: list[dict[str, Any]]
    extractions: list[dict[str, Any]]
    assessments: dict[str, dict[str, Any]]
    proposals: list[dict[str, Any]]
    errors: list[str]
    referral_candidates: list[str]


def _obs_for_watch(detail: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in detail["observations"] if o["status"] in ("recorded", "proposed")]


def build_graph(hub: McpHub, extractor: Extractor, log: ToolLog):
    async def load(state: State) -> State:
        hub.sim_time = state["now"]
        q = await hub.call("queue_state", "get_queue", {"now": state["now"]})
        return {"queue": q["patients"], "extractions": [], "assessments": {}, "proposals": [], "errors": [],
                "referral_candidates": []}

    def after_load(state: State) -> str:
        return "intake" if any(p["unprocessed_notes"] for p in state["queue"]) else "assess"

    async def intake(state: State) -> State:
        extractions = []
        for p in state["queue"]:
            for n in p["unprocessed_notes"]:
                ext = await asyncio.to_thread(extractor.extract, n["text"], log)
                rec = {"patient_id": p["id"], "note_id": n["id"], "note": n["text"], "vitals": ext["vitals"],
                       "signs": ext["signs"], "rejected": ext["rejected"], "unclear": ext["unclear"],
                       "error": ext.get("error"), "usage": ext.get("usage")}
                if ext["vitals"] or ext["signs"]:
                    r = await hub.call("queue_state", "record_vitals", {
                        "patient_id": p["id"], "observed_at": n["written_at"], "vitals": ext["vitals"],
                        "signs": ext["signs"], "run_id": state["run_id"], "note_id": n["id"],
                        "extraction": {k: ext[k] for k in ("evidence", "sign_evidence", "rejected", "unclear")}})
                    rec["proposal"] = r
                elif ext.get("error") or ext["unclear"] or ext["rejected"]:
                    r = await hub.call("queue_state", "propose_action", {
                        "kind": "flag", "patient_id": p["id"], "run_id": state["run_id"], "severity": "WATCH",
                        "payload": {"rule": "NOTE_UNREADABLE", "message": "Could not extract values from note; please enter vitals by hand.",
                                    "note": n["text"], "unclear": ext["unclear"], "rejected": ext["rejected"]},
                        "rationale": "Nothing in the note passed validation.", "citations": ["validator"],
                        "dedupe_key": f"{p['id']}:UNREADABLE:{n['id']}", "created_at": n["written_at"]})
                    rec["proposal"] = r
                await hub.call("queue_state", "mark_note_processed", {"note_id": n["id"], "run_id": state["run_id"]})
                extractions.append(rec)
        return {"extractions": extractions}

    async def assess(state: State) -> State:
        now, run_id = state["now"], state["run_id"]
        assessments: dict[str, dict[str, Any]] = {}
        proposals: list[dict[str, Any]] = []
        referral: list[str] = []
        for p in state["queue"]:
            if p["seen_at"]:
                continue
            detail = await hub.call("queue_state", "get_patient", {"patient_id": p["id"], "now": now})
            obs = _obs_for_watch(detail)
            res = await hub.call("deterioration_watcher", "flag_deterioration",
                                 {"patient": detail["patient"], "observations": obs, "now": now})
            assessments[p["id"]] = res
            pending_obs = [a["id"] for a in detail["actions"] if a["kind"] == "record_observation" and a["status"] == "PROPOSED"]
            provisional = any(o["status"] == "proposed" for o in obs)
            for f in res["flags"]:
                if f["rule"] == "COLOUR_UP":
                    args = {"kind": "change_colour", "patient_id": p["id"], "run_id": run_id, "severity": "ESCALATE",
                            "payload": {"from": p["assigned_colour"], "to": f["proposed_colour"], "provisional": provisional,
                                        "reason": res["current"]["reason"], "drivers": f["drivers"],
                                        "tews": res["current"]["tews_min"], "complete": res["current"]["complete"],
                                        "missing": res["current"]["missing"]},
                            "rationale": f["message"], "citations": [f["cite"]], "dedupe_key": f["dedupe_key"],
                            "depends_on": pending_obs, "created_at": now}
                elif f["rule"] == "RECHECK_OVERDUE":
                    rc = await hub.call("deterioration_watcher", "schedule_recheck", {
                        "colour": res["effective"], "last_observed_at": obs[-1]["observed_at"], "now": now})
                    args = {"kind": "schedule_recheck", "patient_id": p["id"], "run_id": run_id, "severity": "RECHECK",
                            "payload": {**rc, "message": f["message"]}, "rationale": f["message"], "citations": [f["cite"]],
                            "dedupe_key": f["dedupe_key"], "created_at": now}
                else:
                    args = {"kind": "flag", "patient_id": p["id"], "run_id": run_id, "severity": f["severity"],
                            "payload": {"rule": f["rule"], "message": f["message"]}, "rationale": f["message"],
                            "citations": [f["cite"]], "dedupe_key": f["dedupe_key"], "created_at": now}
                r = await hub.call("queue_state", "propose_action", args)
                proposals.append({"patient_id": p["id"], "rule": f["rule"], **r})
            cur = res.get("current") or {}
            if cur.get("colour") in sc.policy()["referral"]["trigger_colours"] or p["assigned_colour"] == "RED":
                referral.append(p["id"])
        return {"assessments": assessments, "proposals": proposals, "referral_candidates": referral}

    async def rerank(state: State) -> State:
        rows = []
        for p in state["queue"]:
            a = state["assessments"].get(p["id"]) or {}
            rows.append({"id": p["id"], "assigned_colour": p["assigned_colour"], "arrival_time": p["arrival_time"],
                         "seen_at": p["seen_at"], "queue_position": p["queue_position"],
                         "effective_colour": a.get("effective") or p["assigned_colour"]})
        rr = await hub.call("priority_scorer", "rerank_queue", {"patients": rows, "now": state["now"]})
        out = list(state["proposals"])
        if rr["changed"]:
            key = hashlib.sha1(json.dumps(rr["proposed_order"]).encode()).hexdigest()[:10]
            moves = "; ".join(f"{m['patient_id']} {m['from']}→{m['to']}" for m in rr["moves"][:6])
            r = await hub.call("queue_state", "propose_action", {
                "kind": "reorder_queue", "patient_id": None, "run_id": state["run_id"], "severity": "WATCH",
                "payload": {"order": rr["proposed_order"], "moves": rr["moves"], "rule": rr["rule"]},
                "rationale": f"Proposed order by colour then time to target: {moves}", "citations": [rr["cite"]],
                "dedupe_key": f"REORDER:{key}", "created_at": state["now"]})
            out.append({"patient_id": None, "rule": "REORDER", **r})
        return {"proposals": out}

    def after_rerank(state: State) -> str:
        return "refer" if state.get("referral_candidates") else "finish"

    async def refer(state: State) -> State:
        directory = ""
        if "filesystem" in hub.sessions:
            try:
                res = await hub.call("filesystem", "read_text_file", {"path": str(_ref_path())})
                if isinstance(res, dict):
                    res = res.get("content") or next((v for v in res.values() if isinstance(v, str)), "")
                directory = res if isinstance(res, str) else ""
            except ToolCallError as e:
                state["errors"].append(str(e))
        out = list(state["proposals"])
        for pid in state["referral_candidates"]:
            a = state["assessments"][pid]
            detail = await hub.call("queue_state", "get_patient", {"patient_id": pid, "now": state["now"]})
            receiving = _pick_receiving(directory, detail["patient"])
            notes = [n["text"] for n in detail["notes"]]
            draft = await hub.call("referral_drafter", "draft_referral", {
                "patient": detail["patient"], "score": a["current"], "flags": a["flags"], "notes": notes,
                "receiving_facility": receiving, "sending_facility": SENDING, "now": state["now"],
                "reason": f"computed {a['current']['colour']} while waiting: {a['current']['reason']}"})
            r = await hub.call("queue_state", "propose_action", {
                "kind": "draft_referral", "patient_id": pid, "run_id": state["run_id"], "severity": "ESCALATE",
                "payload": {"body": draft["body"], "missing_fields": draft["missing_fields"], "sha256": draft["sha256"],
                            "receiving": receiving},
                "rationale": "Patient computed RED at a facility without full resus capacity (local policy).",
                "citations": [sc.policy()["referral"]["cite"], draft.get("colour_cite") or ""],
                "dedupe_key": f"{pid}:REFERRAL", "created_at": state["now"]})
            out.append({"patient_id": pid, "rule": "REFERRAL", **r})
        return {"proposals": out}

    async def finish(state: State) -> State:
        new = [p for p in state["proposals"] if not p.get("deduplicated")]
        log.event("cycle_done", {"now": state["now"], "new_proposals": len(new),
                                 "deduplicated": len(state["proposals"]) - len(new),
                                 "extractions": len(state["extractions"])})
        return {}

    g = StateGraph(State)
    g.add_node("load", load)
    g.add_node("intake", intake)
    g.add_node("assess", assess)
    g.add_node("rerank", rerank)
    g.add_node("refer", refer)
    g.add_node("finish", finish)
    g.add_edge(START, "load")
    g.add_conditional_edges("load", after_load, {"intake": "intake", "assess": "assess"})
    g.add_edge("intake", "assess")
    g.add_edge("assess", "rerank")
    g.add_conditional_edges("rerank", after_rerank, {"refer": "refer", "finish": "finish"})
    g.add_edge("refer", "finish")
    g.add_edge("finish", END)
    return g.compile()


def _ref_path():
    from . import config
    return config.REFERENCE / "facilities.md"


def _pick_receiving(directory: str, patient: dict[str, Any]) -> str:
    if not directory:
        return "RECEIVING FACILITY NOT FOUND (reference directory unavailable) - clinician to complete"
    sections = [s.strip() for s in directory.split("### ")[1:]]
    if patient.get("pregnant"):
        for s in sections:
            if "Maternity" in s.splitlines()[0]:
                return s.splitlines()[0]
    return sections[0].splitlines()[0] if sections else "RECEIVING FACILITY NOT FOUND - clinician to complete"


async def run_cycle(hub: McpHub, extractor: Extractor, log: ToolLog, now: str, run_id: str) -> State:
    graph = build_graph(hub, extractor, log)
    return await graph.ainvoke({"run_id": run_id, "now": now})
