"""Nurse console. Every human decision in the system happens here.

  triage init                       load the synthetic waiting room (clock 08:30)
  triage board                      one-screen view
  triage cycle                      run the agent once at the current clock
  triage clock 09:20                move the simulated clock
  triage note P07 "RR 28 ..."       add a free-text note, then run the agent
  triage vitals P13 --sbp 112       record measured vitals (nurse entry), then run the agent
  triage show ACT-xxxx              full proposal with evidence and citations
  triage approve ACT-xxxx           approve (applies dependent note values too)
  triage edit ACT-xxxx --to YELLOW --reason "..."
  triage veto ACT-xxxx --reason "..."
  triage seen P07                   patient called in
  triage referral REF-xxxx          print a referral
  triage sign REF-xxxx              sign a draft referral
  triage console                    interactive loop
  triage scenario                   replay the scripted demo into runs/
  triage drill                      timed triage drill (manual vs assisted)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from . import board, config, ledger

con_out = Console()
CLOCK = lambda: config.DATA / ".clock"  # noqa: E731
DEFAULT_DATE = "2026-10-12"


def get_now(arg: str | None = None) -> str:
    if arg:
        return arg if "T" in arg else f"{DEFAULT_DATE}T{arg}"
    if CLOCK().exists():
        return CLOCK().read_text().strip()
    return datetime.now().isoformat(timespec="minutes")


def set_now(t: str) -> str:
    t = get_now(t)
    CLOCK().write_text(t)
    return t


def who(args) -> str:
    return args.by or os.environ.get("TRIAGE_USER") or ""


def _need_who(args) -> str:
    w = who(args)
    if not w:
        con_out.print("[red]Say who you are: --by 'Nurse Name' or export TRIAGE_USER.[/red]")
        sys.exit(2)
    return w


def cmd_init(args) -> None:
    fixture = json.loads((config.FIXTURES / args.fixture).read_text())
    ledger.reset()
    ledger.load_fixture(fixture)
    t = set_now(args.at)
    con_out.print(f"Loaded {len(fixture['patients'])} synthetic patients into {config.db_path()}. Clock {t}.")


def cmd_board(args) -> None:
    with ledger.connect() as con:
        board.print_board(con, get_now(args.now), con_out)


def _cycle(now: str, backend: str | None) -> None:
    from .runner import one_cycle
    with con_out.status(f"Agent cycle at {now[11:16]} (backend {backend or config.llm_backend()})…"):
        res = asyncio.run(one_cycle(now, backend))
    for e in res["extractions"]:
        con_out.print(f"[cyan]Note {e['note_id']} ({e['patient_id']}):[/cyan] kept {e['vitals']} signs {e['signs']}"
                      + (f" [red]rejected {len(e['rejected'])}[/red]" if e["rejected"] else "")
                      + (f" [yellow]unclear: {e['unclear']}[/yellow]" if e["unclear"] else ""))
    c = res["cost"]
    con_out.print(f"[dim]{len(res['new_proposals'])} new proposals · {c['mcp_tool_calls']} tool calls · "
                  f"{c['llm_calls']} LLM calls ({c['tokens_in']}+{c['tokens_out']} tokens, {c['llm_seconds']} s) · log runs/console/[/dim]")


def cmd_cycle(args) -> None:
    now = get_now(args.now)
    _cycle(now, args.backend)
    if not args.quiet:
        with ledger.connect() as con:
            board.print_board(con, now, con_out)


def cmd_clock(args) -> None:
    con_out.print(f"Clock set to {set_now(args.time)}")


def cmd_note(args) -> None:
    now = get_now(args.now)
    with ledger.connect() as con:
        if ledger.patient(con, args.patient) is None:
            con_out.print(f"[red]No patient {args.patient}[/red]")
            sys.exit(1)
        nid = ledger.add_note(con, args.patient, args.text, now, author=who(args) or "nurse")
    con_out.print(f"Note {nid} added.")
    if not args.no_cycle:
        cmd_cycle(argparse.Namespace(now=now, backend=args.backend, quiet=False))


def cmd_vitals(args) -> None:
    w = _need_who(args)
    now = get_now(args.now)
    v = {k: getattr(args, k) for k in ("rr", "hr", "sbp", "temp_c", "spo2", "avpu", "mobility", "glucose_mmol") if getattr(args, k) is not None}
    if args.trauma is not None:
        v["trauma"] = args.trauma == "yes"
    with ledger.connect() as con:
        oid = ledger.record_nurse_observation(con, args.patient, now, v, args.sign or [], w)
    con_out.print(f"Recorded {oid}: {v}")
    if not args.no_cycle:
        cmd_cycle(argparse.Namespace(now=now, backend=args.backend, quiet=False))


def cmd_show(args) -> None:
    with ledger.connect() as con:
        a = ledger.action(con, args.id)
        if a is None:
            con_out.print(f"[red]No action {args.id}[/red]")
            sys.exit(1)
        deps = [ledger.action(con, d) for d in a["depends_on"] or []]
    body = json.dumps({k: a[k] for k in ("kind", "patient_id", "severity", "status", "rationale", "citations", "payload",
                                         "decided_by", "decision_reason")}, indent=2, default=str)
    con_out.print(Panel(body, title=f"{a['id']} ({a['created_at']})"))
    for d in deps:
        con_out.print(Panel(json.dumps(d["payload"], indent=2), title=f"depends on {d['id']} [{d['kind']}] {d['status']}"))


def _decide(args, decision: str, edited=None) -> None:
    w = _need_who(args)
    now = get_now(getattr(args, "now", None))
    with ledger.connect() as con:
        try:
            res = ledger.decide(con, args.id, decision, w, getattr(args, "reason", "") or "", edited, at=now)
        except (ledger.GateError, KeyError, ValueError) as e:
            con_out.print(f"[red]Refused: {e}[/red]")
            sys.exit(1)
    con_out.print(f"{decision}: {res}")


def cmd_approve(args) -> None:
    _decide(args, "APPROVED")


def cmd_veto(args) -> None:
    _decide(args, "VETOED")


def cmd_edit(args) -> None:
    with ledger.connect() as con:
        a = ledger.action(con, args.id)
    if a is None:
        con_out.print(f"[red]No action {args.id}[/red]")
        sys.exit(1)
    payload = dict(a["payload"])
    if a["kind"] == "change_colour":
        if not args.to:
            con_out.print("[red]--to COLOUR is required to edit a colour change[/red]")
            sys.exit(2)
        payload["to"] = args.to.upper()
    elif args.json:
        payload.update(json.loads(args.json))
    _decide(args, "EDITED", payload)


def cmd_seen(args) -> None:
    w = _need_who(args)
    with ledger.connect() as con:
        ledger.mark_seen(con, args.patient, get_now(args.now), w)
    con_out.print(f"{args.patient} marked seen.")


def cmd_referral(args) -> None:
    with ledger.connect() as con:
        refs = [r for r in ledger.referrals(con) if r["id"] == args.id or r["patient_id"] == args.id]
    if not refs:
        con_out.print("[red]No such referral[/red]")
        sys.exit(1)
    for r in refs:
        con_out.print(Panel(r["body"], title=f"{r['id']} {r['status']}" + (f" signed by {r['signed_by']}" if r["signed_by"] else "")))


def cmd_sign(args) -> None:
    w = _need_who(args)
    with ledger.connect() as con:
        try:
            con_out.print(ledger.sign_referral(con, args.id, w, at=get_now(None)))
        except (ledger.GateError, KeyError) as e:
            con_out.print(f"[red]Refused: {e}[/red]")
            sys.exit(1)


def cmd_scenario(args) -> None:
    from .runner import run_scenario
    s = asyncio.run(run_scenario(config.FIXTURES / args.file, backend=args.backend, label=args.label, quiet=args.quiet))
    c = s["cost"]
    con_out.print(f"[bold]Run {s['run_id']}[/bold] saved in runs/{s['run_id']}/ · {c['mcp_tool_calls']} tool calls · "
                  f"{c['llm_calls']} LLM calls · {c['tokens_in']}+{c['tokens_out']} tokens · {c['llm_seconds']} s model time · "
                  f"{c['wall_seconds']} s wall · est. USD {c['usd_est']}")


def cmd_log(args) -> None:
    path = Path(args.path) if args.path else config.RUNS / "console" / "tool_calls.jsonl"
    lines = path.read_text().splitlines()[-args.n:]
    for ln in lines:
        r = json.loads(ln)
        if r["kind"] == "mcp_tool":
            con_out.print(f"{r['ts'][11:23]} [cyan]{r['server']}.{r['tool']}[/cyan] ok={r['ok']} {r['latency_ms']}ms")
        elif r["kind"] == "llm":
            con_out.print(f"{r['ts'][11:23]} [magenta]llm {r['model']}[/magenta] {r['tokens_in']}+{r['tokens_out']} tok {r['seconds']}s ok={r['ok']}")
        else:
            con_out.print(f"{r['ts'][11:23]} [dim]{r['name']}[/dim]")


def cmd_drill(args) -> None:
    from .drill import run_drill
    run_drill(args.mode, args.name, args.limit)


INTERACTIVE_HELP = """commands: b(oard) | c(ycle) | t HH:MM | n P07 <note text> | a ACT-id | v ACT-id <reason> | e ACT-id <COLOUR> <reason>
          s ACT-id (show) | seen P07 | r REF-id (referral) | sign REF-id | q"""


def cmd_console(args) -> None:
    w = _need_who(args)
    os.environ["TRIAGE_USER"] = w
    con_out.print(Panel(f"Signed in as {w}. Clock {get_now(None)}.\n{INTERACTIVE_HELP}", title="Triage Desk console"))
    cmd_board(argparse.Namespace(now=None))
    while True:
        try:
            line = input("triage> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = shlex.split(line)
        c, rest = parts[0], parts[1:]
        try:
            if c in ("q", "quit", "exit"):
                break
            if c in ("b", "board"):
                cmd_board(argparse.Namespace(now=None))
            elif c in ("c", "cycle"):
                cmd_cycle(argparse.Namespace(now=None, backend=args.backend, quiet=False))
            elif c == "t":
                cmd_clock(argparse.Namespace(time=rest[0]))
            elif c == "n":
                cmd_note(argparse.Namespace(patient=rest[0], text=" ".join(rest[1:]), now=None, by=w, backend=args.backend, no_cycle=False))
            elif c == "a":
                cmd_approve(argparse.Namespace(id=rest[0], by=w, now=None))
            elif c == "v":
                cmd_veto(argparse.Namespace(id=rest[0], by=w, reason=" ".join(rest[1:]), now=None))
            elif c == "e":
                cmd_edit(argparse.Namespace(id=rest[0], by=w, to=rest[1], reason=" ".join(rest[2:]), json=None, now=None))
            elif c == "s":
                cmd_show(argparse.Namespace(id=rest[0]))
            elif c == "seen":
                cmd_seen(argparse.Namespace(patient=rest[0], by=w, now=None))
            elif c == "r":
                cmd_referral(argparse.Namespace(id=rest[0]))
            elif c == "sign":
                cmd_sign(argparse.Namespace(id=rest[0], by=w))
            else:
                con_out.print(INTERACTIVE_HELP)
        except SystemExit:
            pass
        except IndexError:
            con_out.print(INTERACTIVE_HELP)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="triage", description="Triage Desk nurse console (synthetic data only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.add_argument("--fixture", default="patients.json")
    s.add_argument("--at", default="08:30")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("board")
    s.add_argument("--now")
    s.set_defaults(fn=cmd_board)

    s = sub.add_parser("cycle")
    s.add_argument("--now")
    s.add_argument("--backend", choices=["ollama", "rules"])
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_cycle)

    s = sub.add_parser("clock")
    s.add_argument("time")
    s.set_defaults(fn=cmd_clock)

    s = sub.add_parser("note")
    s.add_argument("patient")
    s.add_argument("text")
    s.add_argument("--now")
    s.add_argument("--by")
    s.add_argument("--backend", choices=["ollama", "rules"])
    s.add_argument("--no-cycle", action="store_true")
    s.set_defaults(fn=cmd_note)

    s = sub.add_parser("vitals")
    s.add_argument("patient")
    s.add_argument("--rr", type=int)
    s.add_argument("--hr", type=int)
    s.add_argument("--sbp", type=int)
    s.add_argument("--temp", dest="temp_c", type=float)
    s.add_argument("--spo2", type=int)
    s.add_argument("--glucose", dest="glucose_mmol", type=float)
    s.add_argument("--avpu", choices=["A", "C", "V", "P", "U"])
    s.add_argument("--mobility", choices=["walking", "with_help", "stretcher_immobile"])
    s.add_argument("--trauma", choices=["yes", "no"])
    s.add_argument("--sign", action="append", help="SATS sign id, repeatable")
    s.add_argument("--now")
    s.add_argument("--by")
    s.add_argument("--backend", choices=["ollama", "rules"])
    s.add_argument("--no-cycle", action="store_true")
    s.set_defaults(fn=cmd_vitals)

    s = sub.add_parser("show")
    s.add_argument("id")
    s.set_defaults(fn=cmd_show)

    for name, fn in (("approve", cmd_approve), ("veto", cmd_veto), ("edit", cmd_edit)):
        s = sub.add_parser(name)
        s.add_argument("id")
        s.add_argument("--by")
        s.add_argument("--now")
        if name != "approve":
            s.add_argument("--reason", required=True)
        if name == "edit":
            s.add_argument("--to")
            s.add_argument("--json")
        s.set_defaults(fn=fn)

    s = sub.add_parser("seen")
    s.add_argument("patient")
    s.add_argument("--by")
    s.add_argument("--now")
    s.set_defaults(fn=cmd_seen)

    s = sub.add_parser("referral")
    s.add_argument("id", help="REF id or patient id")
    s.set_defaults(fn=cmd_referral)

    s = sub.add_parser("sign")
    s.add_argument("id")
    s.add_argument("--by")
    s.set_defaults(fn=cmd_sign)

    s = sub.add_parser("console")
    s.add_argument("--by")
    s.add_argument("--backend", choices=["ollama", "rules"])
    s.set_defaults(fn=cmd_console)

    s = sub.add_parser("scenario")
    s.add_argument("--file", default="demo_scenario.json")
    s.add_argument("--backend", choices=["ollama", "rules"])
    s.add_argument("--label")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_scenario)

    s = sub.add_parser("log")
    s.add_argument("--path")
    s.add_argument("-n", type=int, default=30)
    s.set_defaults(fn=cmd_log)

    s = sub.add_parser("drill")
    s.add_argument("--mode", choices=["manual", "assisted"], required=True)
    s.add_argument("--name", required=True, help="participant code (no real names)")
    s.add_argument("--limit", type=int, default=8)
    s.set_defaults(fn=cmd_drill)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
