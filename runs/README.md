# Runs

Each folder is one real, complete replay of `data/fixtures/demo_scenario.json`
(a synthetic morning clinic: 14 patients, 6 nurse notes, 6 agent cycles, scripted
nurse decisions). It was produced by:

```
TRIAGE_TEMPERATURE=0 .venv/bin/triage scenario --backend ollama --label <label> --quiet
```

on a MacBook Pro M2 Pro (16 GB) with `qwen2.5:7b-instruct` (Q4_K_M, Ollama),
fully offline apart from the first `npx` download of the borrowed server.

| File | Contents |
|---|---|
| `tool_calls.jsonl` | Every MCP tool call (server, tool, args, result preview, latency) and every model call (tokens in/out, seconds), in order |
| `summary.json` | Run metadata, cost block, every action with its final status, scenario timeline with extractions |
| `ledger.db` | The SQLite ledger at the end of the run |
| `board_HHMM.txt` | The nurse board after each agent cycle |

## The two official open-weights runs

| Run | Code | Result |
|---|---|---|
| `20261005T122720Z-qwen2.5-7b-t0` | before the pain-band and injection guards | 5/5 deteriorations proposed on time. **1 false escalation:** P08's note "Feels fine, pain 3/10" was mapped by the model to *moderate pain*, proposing GREEN → YELLOW. Kept here on purpose. |
| `20261005T122929Z-qwen2.5-7b-t0-fix2` | after adding local pain-score bands (a written N/10 sets the pain sign) | 5/5 on time, 0 false escalations. The bad sign appears in the rejected list for P08. |

Both runs are real. Neither was edited. Cosmetic changes made after these runs
(the wording of the "move to resus" message, age formatting in referrals, and
the signed-referral header) do not affect any proposal.

`runs/console/` (git-ignored) collects the logs of interactive console sessions.
`runs/drill/` (git-ignored) collects drill results; no clinician has done the
drill yet.
