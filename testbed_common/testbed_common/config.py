"""Static configuration: where each service listens and where state is written."""
from __future__ import annotations

import os
from pathlib import Path

# <repo>/testbed_common/testbed_common/config.py -> parents[2] is the repo root.
ROOT = Path(os.environ.get("TESTBED_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("TESTBED_DATA", ROOT / "data"))
LOG_DIR = Path(os.environ.get("TESTBED_LOGS", ROOT / "logs"))

HOST = "127.0.0.1"

# One port per service. The LLM gateway is a service too: every stack's model
# traffic goes through it, which is the natural interception point to test.
PORTS = {
    "llm": 9100,
    "router": 9101,
    "policy": 9102,
    "risk": 9103,
    "refund": 9104,
    "notify": 9105,
}

LLM_BASE_URL = os.environ.get("TESTBED_LLM_BASE_URL", f"http://{HOST}:{PORTS['llm']}/v1")
LLM_API_KEY = os.environ.get("TESTBED_LLM_API_KEY", "testbed-key")
LLM_MODEL = os.environ.get("TESTBED_LLM_MODEL", "testbed-model")


def agent_url(name: str) -> str:
    """Base URL of the named agent service."""
    return f"http://{HOST}:{PORTS[name]}"


def agent_data_dir(name: str) -> Path:
    """Directory where one agent keeps its own session and memory files."""
    path = DATA_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
