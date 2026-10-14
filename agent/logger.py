"""Structured JSON-lines logging: one line per event, easy to grep or load in pandas."""
import json
import time
from pathlib import Path

LOG_PATH = Path("logs/agent.jsonl")


def log_event(run_id: str, event: str, **fields) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": round(time.time(), 3), "run_id": run_id, "event": event, **fields}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
