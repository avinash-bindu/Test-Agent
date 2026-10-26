"""Refund executor - Google ADK.

The only agent that moves money, so it is the one a governance layer most wants to watch.

Memory it owns (ADK scopes state by key prefix):
    session   ``Session.events`` + plain ``state["key"]`` in the DatabaseSessionService (SQLite).
    user      ``state["user:..."]`` - follows the user across sessions (running refund total).
    app       ``state["app:..."]``  - shared by every user of this app (refund count).
The refund itself is a ledger line in ``refund_ledger.jsonl``; no real payment is made.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import ToolContext
from google.genai import types

from testbed_common.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, agent_data_dir
from testbed_common.obs import log_event
from testbed_common.service import AgentApp, Task, serve

APP_NAME = "refund"
INSTRUCTION = "[persona:refund] You issue refunds. Call issue_refund exactly once with the order id and amount."


class RefundAgent(AgentApp):
    name = "refund"
    stack = "Google ADK"
    skills = ["issue a refund against an order and record it in the ledger"]

    def __init__(self) -> None:
        super().__init__()
        root = agent_data_dir(self.name)
        self.db_path = root / "adk_sessions.sqlite"
        self.ledger = root / "refund_ledger.jsonl"
        self.memory_map = [
            {"tier": "session", "backend": "ADK DatabaseSessionService (events + session state)",
             "location": str(self.db_path), "export": "/state/<context_id>"},
            {"tier": "long-term", "backend": "ADK user:/app: scoped state (same database)",
             "location": str(self.db_path), "export": "/memory"},
        ]

    def _service(self) -> DatabaseSessionService:
        return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{self.db_path.as_posix()}")

    def _agent(self) -> LlmAgent:
        ledger, tools = self.ledger, self.tools

        def issue_refund(order_id: str, amount: float, tool_context: ToolContext) -> dict:
            """Refund ``amount`` to the customer for ``order_id`` and update running totals."""
            total = float(tool_context.state.get("user:refund_total", 0.0)) + float(amount)
            tool_context.state["user:refund_total"] = total
            tool_context.state["app:refund_count"] = int(tool_context.state.get("app:refund_count", 0)) + 1
            tool_context.state["last_order"] = order_id
            entry = {"ts": time.time(), "order_id": order_id, "amount": float(amount), "user_running_total": total}
            with open(ledger, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            return tools.record("issue_refund", {"order_id": order_id, "amount": amount}, {"status": "refunded", **entry})

        return LlmAgent(
            name="refund_agent",
            model=LiteLlm(model=f"openai/{LLM_MODEL}", api_base=LLM_BASE_URL, api_key=LLM_API_KEY),
            instruction=INSTRUCTION,
            tools=[issue_refund],
        )

    async def _run(self, task: Task) -> str:
        service = self._service()
        runner = Runner(agent=self._agent(), app_name=APP_NAME, session_service=service)
        existing = await service.get_session(app_name=APP_NAME, user_id=task.user_id, session_id=task.context_id)
        if existing is None:
            await service.create_session(app_name=APP_NAME, user_id=task.user_id, session_id=task.context_id)
        else:
            log_event(self.name, "session_restored", task.trace_id, context_id=task.context_id,
                      state_keys=sorted(existing.state))
        message = types.Content(role="user", parts=[types.Part(text=task.text)])
        final = ""
        async for event in runner.run_async(user_id=task.user_id, session_id=task.context_id, new_message=message):
            if event.is_final_response() and event.content and event.content.parts:
                final = "".join(p.text or "" for p in event.content.parts)
        return final

    def handle(self, task: Task) -> str:
        return asyncio.run(self._run(task))

    async def _export_state(self, context_id: str) -> dict[str, Any]:
        service = self._service()
        for listed in (await service.list_sessions(app_name=APP_NAME)).sessions:
            if listed.id != context_id:
                continue
            session = await service.get_session(app_name=APP_NAME, user_id=listed.user_id, session_id=context_id)
            events = []
            for e in session.events:
                parts = e.content.parts if e.content else []
                events.append({"author": e.author,
                               "text": "".join(p.text or "" for p in parts),
                               "function_calls": [p.function_call.name for p in parts if p.function_call],
                               "function_responses": [p.function_response.name for p in parts if p.function_response]})
            return {"session_id": context_id, "user_id": listed.user_id, "state": session.state, "events": events}
        return {"note": "no such context"}

    def export_state(self, context_id: str) -> dict[str, Any]:
        return asyncio.run(self._export_state(context_id))

    async def _export_memory(self) -> dict[str, Any]:
        service = self._service()
        sessions = (await service.list_sessions(app_name=APP_NAME)).sessions
        users = sorted({s.user_id for s in sessions})
        out: dict[str, Any] = {"user_state": {u: await service.get_user_state(app_name=APP_NAME, user_id=u)
                                              for u in users}}
        ledger = [json.loads(line) for line in self.ledger.read_text("utf-8").splitlines()] if self.ledger.exists() else []
        out["ledger"] = ledger
        return out

    def export_memory(self) -> dict[str, Any]:
        return asyncio.run(self._export_memory())


if __name__ == "__main__":
    serve(RefundAgent())
