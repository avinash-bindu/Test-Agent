"""Client side of the A2A-lite contract: how one agent calls another."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any

from .config import agent_url


class AgentCallError(RuntimeError):
    """The remote agent was unreachable or answered with a non-2xx status."""


def call_agent(
    target: str,
    text: str,
    *,
    context_id: str,
    trace_id: str,
    caller_user: str,
    caller_agent: str,
    data: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """POST a task to ``target`` and return its JSON result.

    The user identity and trace id travel as headers on every hop so a gateway
    sitting between the agents can attribute the call without reading the body.
    """
    body = {
        "task_id": uuid.uuid4().hex,
        "context_id": context_id,
        "input": {"text": text, "data": data or {}},
        "caller": {"user_id": caller_user, "agent_id": caller_agent},
    }
    request = urllib.request.Request(
        f"{agent_url(target)}/tasks",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Trace-Id": trace_id,
            "X-User-Id": caller_user,
            "X-Agent-Id": caller_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise AgentCallError(f"{target} answered {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
    except urllib.error.URLError as exc:
        raise AgentCallError(f"{target} unreachable: {exc.reason}") from exc
