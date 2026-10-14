# Architecture

Triage Desk watches an outpatient waiting room for patients who get worse while
they wait. A deterministic scorer applies the South African Triage Scale (SATS,
adult chart). A small open-weights model only reads nurses' free-text notes.
Every change is a proposal that a named nurse approves, edits or vetoes.

```
 nurse ── triage console (CLI) ──────────────┐  approve / edit / veto / sign / seen / vitals / note
                                             │  (the ONLY code path that can decide)
                                             ▼
                                   SQLite ledger (data/ledger.db)
                                    patients · observations · notes
                                    actions (PROPOSED→APPROVED/EDITED/VETOED) · referrals · audit
                                             ▲
            ┌────────────── MCP over stdio ──┴───────────────────────────────────────┐
            │                                                                         │
  queue-state            priority-scorer        deterioration-watcher   referral-drafter     filesystem (borrowed,
  get_queue              score_vitals           flag_deterioration      draft_referral       official @modelcontext-
  get_patient            score_patient          schedule_recheck        required_fields      protocol/server-filesystem)
  record_vitals*         rerank_queue                                                        read_text_file only
  propose_action*        scale_reference                                                     (client allowlist)
  mark_note_processed
            ▲   * creates PROPOSED rows only                                                   ▲
            └──────────────────────── LangGraph agent (MCP client) ───────────────────────────┘
                load ─┬─> intake ─> assess ─> rerank ─┬─> refer ─> finish
                      └──────────────> assess          └──────────> finish
                         │
                         └── Qwen2.5-7B-Instruct via Ollama on 127.0.0.1 (notes → quoted fields)
```

## Components

| Piece | File | What it decides | What it cannot do |
|---|---|---|---|
| Scale data | `data/scale/sats_adult.json` | SATS 2012 thresholds, discriminators, target times, each with a page reference | — (CC BY-NC-SA, outside the MIT licence) |
| Local policy | `data/scale/local_policy.json` | Recheck intervals, adult SpO2 < 92% → RED, TEWS-rise watch, pain-score bands. Each has a source or says "no published source" | Marked NEEDS CLINICAL SIGN-OFF |
| Scorer | `triage_desk/scale.py` | Colour = highest of (clinical signs, TEWS band, local SpO2 rule). Missing parameters give `tews_min`/`tews_max`/`possible_colour` instead of being assumed normal | Refuses patients aged 12 or under |
| Watcher | `triage_desk/rules.py` | Flags: COLOUR_UP, TEWS_RISE, SINGLE_PARAM_3, RED_IN_QUEUE, TARGET_BREACH, RECHECK_OVERDUE, INCOMPLETE, STALE_VITALS, IMPROVED | Never proposes a downgrade |
| Intake | `triage_desk/llm.py` | Turns a note into fields + SATS sign ids, each with a verbatim quote | Cannot output a colour; output passes the validator or is dropped |
| Ledger | `triage_desk/ledger.py` | Stores proposals; `decide()` and `apply_action()` enforce the gate | Rejects empty or `agent:` identities; edits and vetoes need a reason |
| Agent | `triage_desk/agent.py` | Fixed graph; calls tools in order | Has no tool that approves anything |
| Console | `triage_desk/console.py`, `board.py` | Human decisions; one-screen board | — |
| Logger | `triage_desk/toollog.py` | Writes every MCP call and model call (tokens, seconds) to `tool_calls.jsonl` | — |

## Why the model is boxed in

The evidence shows LLMs are poor at assigning triage levels: in Nedos 2026 the best
weighted κ was about 0.47, and no model reached strong agreement. So the model
never picks a priority. It does the one thing a rule cannot do well, which is read
messy, mixed-language notes. Three layers stop it doing harm:

1. **Grounding validator** (`validate()`). Each value must quote the note. The
   quote must be in the note, the number must be in the quote, SBP must be the
   first number of `a/b`, values must be in range, categories must be valid,
   and sign ids must exist in SATS. Negated quotes ("denies chest pain") are
   dropped. Quotes inside instruction-like sentences ("ignore previous… output
   RR 8") are dropped. A written pain score (N/10) sets the pain sign under the
   local bands, whatever the model chose. Every rejection is shown to the nurse.
2. **Deterministic scoring.** Colour comes from the scale file, never from model
   text.
3. **Approval gate.** Extracted values are stored as `proposed` observations. A
   colour change that depends on them carries `depends_on`, so approving it also
   confirms the values. The board marks such proposals PROVISIONAL.

## Data flow for one note

1. Nurse: `triage note P07 "RR 28, pulse 118, sats 90%, confused…"`. The note is stored.
2. `load`: `queue_state.get_queue(now)` returns the note as unprocessed.
3. `intake`: Qwen extracts the fields and the validator keeps grounded ones.
   `queue_state.record_vitals` then stores a PROPOSED observation plus a
   `record_observation` action.
4. `assess`: `get_patient` and `deterioration_watcher.flag_deterioration` merge
   the latest value of each parameter (older values carried forward, with
   times). The scorer runs and returns COLOUR_UP GREEN → RED with drivers and
   citations. `propose_action(change_colour, depends_on=[record_observation])`
   follows.
5. `rerank`: `priority_scorer.rerank_queue` proposes an order (colour, then
   minutes left to the SATS target, then arrival).
6. `refer`: for RED patients, the agent reads `data/reference/facilities.md`
   through the borrowed filesystem server. `referral_drafter.draft_referral`
   writes an SBAR DRAFT, and the agent proposes it.
7. The nurse runs `approve`. The ledger applies the observation and the colour,
   and writes an audit row. A clinician signs the referral, which is never sent
   electronically.

## Alert discipline

- Each flag has a `dedupe_key` tied to the observation that triggered it. The
  same finding on the same data is raised once. A vetoed finding is not raised
  again until the data changes, and it stays on the board with its reason (E7).
- One open proposal per patient for colour changes, reorders and rechecks.
  Newer proposals supersede older ones.
- Severity tiers: ESCALATE / WATCH / CONFIRM go to "Needs your decision".
  RECHECK goes to a one-line recheck list. INFO goes to a dim FYI line. A
  yellow or green target-time breach is INFO; an orange breach is ESCALATE.

## Data residency

- The model endpoint must be localhost or a private address
  (`check_residency`, E9). Override with `TRIAGE_ALLOW_REMOTE_LLM=1`, which we
  never set.
- The servers talk over stdio on the same machine. Nothing listens on a port
  except Ollama on 127.0.0.1.
- The borrowed filesystem server is limited to `data/reference/`. The client
  allows only read tools on it (E6 checks that `write_file` is blocked).

## Cost per run

Measured, not estimated. Every model call logs its prompt and completion tokens
and wall seconds. `summary.json` turns model seconds into energy, using a stated
wattage and tariff (`TRIAGE_DEVICE_WATTS`, `TRIAGE_USD_PER_KWH`). There are no
API fees. See EVALS.md E11.

## What is deliberately not here

- Paediatric chart, NEWS2, sepsis scoring, diagnosis and treatment advice.
- Automatic downgrades, automatic sending of referrals, and any approval by software.
- Real patient data, EMR integration, user authentication (the console trusts
  the name given; a deployment needs real sign-in).
