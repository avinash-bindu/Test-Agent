"""Run the scripted scenarios against the live testbed and print who did what, on which stack.

    python scripts/demo.py                 all scenarios
    python scripts/demo.py happy skip-risk chosen scenarios
    python scripts/demo.py --memory        also dump every agent's session state and long-term memory

Needs the stack running (scripts\\run-all.ps1). Standard library only.
"""
from __future__ import annotations

import json
import sys
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testbed_common"))

from testbed_common.a2a import call_agent  # noqa: E402
from testbed_common.config import agent_url  # noqa: E402
from testbed_common.scenarios import BY_KEY, SCENARIOS  # noqa: E402


def run_scenario(key: str) -> dict:
    """Send one scenario to the router and return its reply."""
    s = BY_KEY[key]
    return call_agent("router", s.text, context_id=s.context_id, trace_id=f"demo-{key}-{uuid.uuid4().hex[:6]}",
                      caller_user=s.user, caller_agent="demo-client")


def show(key: str, reply: dict) -> None:
    s = BY_KEY[key]
    print(f"\n=== {s.key}: {s.title}\n    user={s.user}  trace={reply['trace_id']}\n    says: {s.text}")
    for call in reply["tool_calls"]:
        r = call["result"]
        if "stack" not in r:
            print(f"  -> router asked {call['name']}: FAILED {r}")
            continue
        print(f"  -> router asked {call['name'].removeprefix('ask_'):<7} [{r['stack']}]")
        for inner in r["tool_calls"]:
            args = json.dumps(inner["args"])
            print(f"       tool {inner['name']}({args[:110]}{'...' if len(args) > 110 else ''})")
    print(f"  router final: {reply['output']['text'][:200]}")
    print(f"  risk under test: {s.risk}")


def dump_memory(context_ids: list[str]) -> None:
    """Pull every agent's exported state and memory: the surfaces a governance layer connects to."""
    for name in ("router", "policy", "risk", "refund", "notify"):
        try:
            card = json.load(urllib.request.urlopen(f"{agent_url(name)}/.well-known/agent.json", timeout=5))
            print(f"\n--- {name} [{card['stack']}]")
            for m in card["memory"]:
                print(f"    {m['tier']:<9} {m['backend']}")
            for ctx in context_ids[:1]:
                state = json.load(urllib.request.urlopen(f"{agent_url(name)}/state/{ctx}", timeout=10))
                print(f"    /state/{ctx}: {json.dumps(state)[:240]}")
            mem = json.load(urllib.request.urlopen(f"{agent_url(name)}/memory", timeout=10))
            print(f"    /memory: {json.dumps(mem)[:240]}")
        except Exception as exc:  # report, keep going: one down agent should not hide the rest
            print(f"    {name}: unreachable ({exc})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keys = args or [s.key for s in SCENARIOS]
    for k in keys:
        show(k, run_scenario(k))
    if "--memory" in sys.argv:
        dump_memory([BY_KEY[k].context_id for k in keys])
