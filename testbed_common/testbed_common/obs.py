"""Structured event log: one JSON line per event, one file per service.

Every branch an agent takes is logged with the same shape so a governance layer
(or a test) can replay a request end to end by trace id.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

from .config import LOG_DIR

_lock = threading.Lock()


def log_event(service: str, event: str, trace_id: str | None = None, **fields: Any) -> None:
    """Append one event to ``logs/<service>.jsonl`` and echo a short line to stderr."""
    record = {"ts": round(time.time(), 3), "service": service, "event": event, "trace_id": trace_id, **fields}
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _lock:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / f"{service}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    print(f"[{service}] {event} trace={trace_id}", file=sys.stderr, flush=True)
