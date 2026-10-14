"""Paths and runtime settings. Everything is overridable by environment variable."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TRIAGE_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
SCALE_FILE = DATA / "scale" / "sats_adult.json"
POLICY_FILE = DATA / "scale" / "local_policy.json"
FIXTURES = DATA / "fixtures"
REFERENCE = DATA / "reference"
RUNS = ROOT / "runs"


def db_path() -> Path:
    return Path(os.environ.get("TRIAGE_DB", DATA / "ledger.db"))


def llm_backend() -> str:
    return os.environ.get("TRIAGE_LLM", "ollama")


def llm_model() -> str:
    return os.environ.get("TRIAGE_MODEL", "qwen2.5:7b-instruct")


def llm_url() -> str:
    return os.environ.get("TRIAGE_LLM_URL", "http://127.0.0.1:11434")


def llm_temperature() -> float:
    return float(os.environ.get("TRIAGE_TEMPERATURE", "0"))


def llm_seed() -> int:
    return int(os.environ.get("TRIAGE_SEED", "7"))


def use_borrowed_fs() -> bool:
    return os.environ.get("TRIAGE_BORROWED_FS", "1") != "0"
