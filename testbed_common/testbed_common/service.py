"""The A2A-lite HTTP contract every agent exposes, whatever framework is inside.

    GET  /.well-known/agent.json   agent card: name, stack, skills, and where its memory lives
    GET  /health
    POST /tasks                    run one task (the only way work enters the agent)
    GET  /state/<context_id>       the framework's own session/thread state, exported as JSON
    GET  /memory                   the framework's long-term memory, exported as JSON

The last two are the *connection points* a governance layer would attach to:
they are how an agent owner lets a third party read the state that never
appears in a task response.
"""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import HOST, PORTS
from .obs import log_event


@dataclass
class Task:
    """One unit of work handed to an agent."""

    task_id: str
    context_id: str
    text: str
    data: dict[str, Any]
    user_id: str
    caller_agent: str
    trace_id: str


@dataclass
class ToolLog:
    """Tool calls made while serving the current task, returned to the caller."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, args: dict[str, Any], result: Any) -> Any:
        """Remember one tool call and hand the result back unchanged."""
        self.calls.append({"name": name, "args": args, "result": result})
        return result


class AgentApp:
    """Base class: subclasses plug a framework in behind four methods."""

    name: str = "agent"        # key in config.PORTS
    stack: str = "unknown"     # framework label shown on the agent card
    skills: list[str] = []
    memory_map: list[dict[str, str]] = []   # [{tier, backend, location, export}]

    def __init__(self) -> None:
        self.tools = ToolLog()

    def handle(self, task: Task) -> str:
        """Run the task through the framework and return the final answer text."""
        raise NotImplementedError

    def export_state(self, context_id: str) -> dict[str, Any]:
        """Session/thread state for ``context_id`` in the framework's own shape."""
        return {"note": "no session state"}

    def export_memory(self) -> dict[str, Any]:
        """Long-term memory in the framework's own shape."""
        return {"note": "no long-term memory"}

    def card(self) -> dict[str, Any]:
        """The agent card served at /.well-known/agent.json."""
        return {
            "name": self.name,
            "stack": self.stack,
            "url": f"http://{HOST}:{PORTS[self.name]}",
            "skills": self.skills,
            "memory": self.memory_map,
            "contract": "a2a-lite/0.1",
        }


def serve(app: AgentApp) -> None:
    """Block forever serving ``app`` on its configured port.

    Requests are serialised with a lock: the tool log is per-task and the
    frameworks keep process-global state, so concurrency would only add noise
    to a testbed.
    """
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:  # request lines go to obs.log_event instead
            return

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            path = self.path.split("?")[0]
            try:
                if path == "/health":
                    self._send(200, {"status": "ok", "agent": app.name})
                elif path == "/.well-known/agent.json":
                    self._send(200, app.card())
                elif path == "/memory":
                    with lock:
                        self._send(200, app.export_memory())
                elif path.startswith("/state/"):
                    with lock:
                        self._send(200, app.export_state(path.removeprefix("/state/")))
                else:
                    self._send(404, {"error": "not found", "path": path})
            except Exception as exc:  # export failures must be visible, not swallowed
                log_event(app.name, "export_failed", path=path, error=repr(exc))
                self._send(500, {"error": repr(exc)})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/tasks":
                self._send(404, {"error": "not found", "path": self.path})
                return
            trace_id = self.headers.get("X-Trace-Id") or uuid.uuid4().hex
            try:
                raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                task = Task(
                    task_id=raw.get("task_id") or uuid.uuid4().hex,
                    context_id=raw.get("context_id") or uuid.uuid4().hex,
                    text=(raw.get("input") or {}).get("text", ""),
                    data=(raw.get("input") or {}).get("data") or {},
                    user_id=(raw.get("caller") or {}).get("user_id", self.headers.get("X-User-Id", "anonymous")),
                    caller_agent=(raw.get("caller") or {}).get("agent_id", self.headers.get("X-Agent-Id", "client")),
                    trace_id=trace_id,
                )
            except (ValueError, TypeError) as exc:
                log_event(app.name, "task_rejected", trace_id, error=repr(exc))
                self._send(400, {"error": f"bad task: {exc}"})
                return

            log_event(app.name, "task_received", trace_id, task_id=task.task_id, context_id=task.context_id,
                      user=task.user_id, caller=task.caller_agent, text=task.text[:300])
            with lock:
                app.tools = ToolLog()
                try:
                    answer = app.handle(task)
                    status = "completed"
                except Exception as exc:
                    log_event(app.name, "task_failed", trace_id, error=repr(exc), tb=traceback.format_exc()[-1500:])
                    answer, status = f"agent error: {exc!r}", "failed"
                calls = list(app.tools.calls)
            log_event(app.name, "task_finished", trace_id, task_id=task.task_id, status=status,
                      tool_calls=[c["name"] for c in calls], answer=answer[:300])
            self._send(200 if status == "completed" else 500, {
                "task_id": task.task_id,
                "context_id": task.context_id,
                "agent": app.name,
                "stack": app.stack,
                "status": status,
                "output": {"text": answer},
                "tool_calls": calls,
                "trace_id": trace_id,
            })

    server = ThreadingHTTPServer((HOST, PORTS[app.name]), Handler)
    log_event(app.name, "listening", stack=app.stack, port=PORTS[app.name])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_event(app.name, "stopped")
