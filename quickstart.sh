#!/usr/bin/env bash
# Triage Desk quickstart: from a clean clone to a scored waiting room.
#   ./quickstart.sh            full setup, uses the local Qwen model through Ollama
#   ./quickstart.sh --no-llm   skip the model (regex extractor); runs anywhere in about a minute
set -euo pipefail
cd "$(dirname "$0")"

NO_LLM=0
[[ "${1:-}" == "--no-llm" ]] && NO_LLM=1
MODEL="${TRIAGE_MODEL:-qwen2.5:7b-instruct}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/5 Python environment"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1
fi
uv venv --python 3.12 .venv >/dev/null
uv pip install --python .venv/bin/python -e ".[dev]" >/dev/null
echo "ok: $(.venv/bin/python --version)"

say "2/5 Borrowed MCP server (official filesystem server via npx)"
if command -v npx >/dev/null 2>&1; then
  # Download once so the first agent run does not stall; the server waits on stdin, so stop it after a while.
  ( npx -y @modelcontextprotocol/server-filesystem@2026.8.31 data/reference </dev/null >/dev/null 2>&1 & pid=$!; sleep 40; kill $pid 2>/dev/null ) || true
  echo "ok: npx available"
else
  echo "npx not found: the agent will run without the borrowed server (referral directory lookup disabled)."
  export TRIAGE_BORROWED_FS=0
fi

BACKEND=rules
if [[ $NO_LLM == 0 ]]; then
  say "3/5 Local open-weights model ($MODEL via Ollama)"
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found. Install from https://ollama.com (macOS: brew install ollama), or rerun with --no-llm."; exit 1
  fi
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null || (ollama serve >/dev/null 2>&1 &) ; sleep 2
  ollama pull "$MODEL"
  BACKEND=ollama
else
  say "3/5 Skipping the model (--no-llm): notes are read by the regex extractor"
fi

say "4/5 Tests"
.venv/bin/python -m pytest -q

say "5/5 Replay the synthetic morning clinic"
TRIAGE_TEMPERATURE=0 .venv/bin/triage scenario --backend "$BACKEND" --label "quickstart-$BACKEND" --quiet
RUN=$(ls -d runs/*quickstart-$BACKEND | tail -1)
cat "$RUN/board_0950.txt"

cat <<EOF

Done. Everything above is synthetic.
  Run folder:      $RUN  (tool_calls.jsonl, summary.json, board snapshots)
  Try the console: .venv/bin/triage init && .venv/bin/triage console --by "Nurse Demo" --backend $BACKEND
  Evals:           .venv/bin/python evals/run_evals.py $( [[ $BACKEND == rules ]] && echo --no-llm )
EOF
