"""Orchestrator - OpenAI Agents SDK.

Receives the customer's request, then drives the other four agents (each on a different stack)
through function tools that call them over the A2A-lite contract.

Memory it owns:
    session   ``SQLiteSession``: the full item history (messages, tool calls, tool outputs) for a
              conversation, keyed by context_id and replayed into the model on every turn.
    long-term the SDK has none built in, so this agent keeps a per-user *case file* (JSON) and feeds
              it back through dynamic instructions - the usual hand-rolled pattern.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from agents import Agent, OpenAIChatCompletionsModel, RunContextWrapper, Runner, SQLiteSession, function_tool, set_tracing_disabled
from openai import AsyncOpenAI

from testbed_common.a2a import AgentCallError, call_agent
from testbed_common.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, agent_data_dir
from testbed_common.domain import CUSTOMERS, ORDERS
from testbed_common.obs import log_event
from testbed_common.service import AgentApp, Task, serve

set_tracing_disabled(True)  # the SDK's default tracing exports to OpenAI's servers

INSTRUCTIONS = (
    "[persona:router] You coordinate refund requests. Ask the policy agent, then the risk agent; "
    "only if risk approves ask the refund agent; finally ask the notify agent to tell the customer."
)


class RouterAgent(AgentApp):
    name = "router"
    stack = "OpenAI Agents SDK"
    skills = ["triage a refund request and coordinate policy, risk, refund and notification agents"]

    def __init__(self) -> None:
        super().__init__()
        root = agent_data_dir(self.name)
        self.db = root / "sessions.sqlite"
        self.cases_file = root / "case_files.json"
        self.task: Task | None = None   # set per request; requests are serialised by the service lock
        self.memory_map = [
            {"tier": "session", "backend": "OpenAI Agents SDK SQLiteSession", "location": str(self.db),
             "export": "/state/<context_id>"},
            {"tier": "long-term", "backend": "hand-rolled per-user case file (JSON) via dynamic instructions",
             "location": str(self.cases_file), "export": "/memory"},
        ]

    # ----- long-term memory -------------------------------------------------------------------
    def _cases(self) -> dict[str, list[str]]:
        return json.loads(self.cases_file.read_text("utf-8")) if self.cases_file.exists() else {}

    def _remember(self, user: str, note: str) -> None:
        cases = self._cases()
        cases.setdefault(user, []).append(note)
        self.cases_file.write_text(json.dumps(cases, indent=1), "utf-8")

    # ----- calling the other agents -----------------------------------------------------------
    def _delegate(self, tool: str, target: str, text: str, args: dict[str, Any]) -> str:
        """Call another agent; every outcome, including failure, is logged and returned as text."""
        task = self.task
        assert task is not None
        try:
            reply = call_agent(target, text, context_id=task.context_id, trace_id=task.trace_id,
                               caller_user=task.user_id, caller_agent=self.name)
        except AgentCallError as exc:
            log_event(self.name, "delegation_failed", task.trace_id, target=target, error=str(exc))
            self.tools.record(tool, args, {"error": str(exc)})
            return f"ERROR: {target} agent unavailable"
        answer = reply["output"]["text"]
        log_event(self.name, "delegated", task.trace_id, target=target, stack=reply.get("stack"),
                  status=reply["status"], inner_tools=[c["name"] for c in reply.get("tool_calls", [])])
        self.tools.record(tool, args, {"agent": target, "stack": reply.get("stack"), "status": reply["status"],
                                       "text": answer, "tool_calls": reply.get("tool_calls", [])})
        return answer

    def _build(self) -> Agent:
        me = self

        @function_tool
        def ask_policy(question: str) -> str:
            """Ask the policy agent what the refund policy says."""
            return me._delegate("ask_policy", "policy", question, {"question": question})

        @function_tool
        def ask_risk(order_id: str, amount: float) -> str:
            """Ask the risk agent to review a refund request."""
            order = ORDERS.get(order_id, {})
            amount = amount or order.get("amount", 0.0)
            text = f"Assess refund risk for {order_id} amount ${amount} customer {order.get('customer_id', 'unknown')}"
            return me._delegate("ask_risk", "risk", text, {"order_id": order_id, "amount": amount})

        @function_tool
        def ask_refund(order_id: str, amount: float) -> str:
            """Ask the refund agent to issue the refund."""
            amount = amount or ORDERS.get(order_id, {}).get("amount", 0.0)
            return me._delegate("ask_refund", "refund", f"Refund {order_id} amount ${amount}",
                                {"order_id": order_id, "amount": amount})

        @function_tool
        def ask_notify(order_id: str, message: str) -> str:
            """Ask the notify agent to e-mail the customer."""
            customer = CUSTOMERS.get(ORDERS.get(order_id, {}).get("customer_id", ""), {})
            email = customer.get("email", "customer@example.com")
            return me._delegate("ask_notify", "notify", f"Send {email} this update: {message}",
                                {"order_id": order_id, "message": message})

        def instructions(_: RunContextWrapper[Any], __: Agent) -> str:
            user = me.task.user_id if me.task else "anonymous"
            prior = me._cases().get(user, [])[-3:]
            return INSTRUCTIONS + (f" Prior cases for this user: {'; '.join(prior)}" if prior else "")

        client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=60.0, max_retries=0)
        return Agent(name="router", instructions=instructions,
                     model=OpenAIChatCompletionsModel(model=LLM_MODEL, openai_client=client),
                     tools=[ask_policy, ask_risk, ask_refund, ask_notify])

    # ----- AgentApp ---------------------------------------------------------------------------
    def handle(self, task: Task) -> str:
        self.task = task
        session = SQLiteSession(task.context_id, str(self.db))
        # Built per request: run_sync opens a new event loop each time, and a pooled keep-alive
        # connection from an earlier loop would hang the next call that reuses it.
        result = Runner.run_sync(self._build(), task.text, session=session)
        order = (re.search(r"ORD-\d+", task.text) or [None])[0]
        self._remember(task.user_id, f"{order or 'request'}: {str(result.final_output)[:160]}")
        log_event(self.name, "memory_written", task.trace_id, user=task.user_id)
        return str(result.final_output)

    def export_state(self, context_id: str) -> dict[str, Any]:
        items = asyncio.run(SQLiteSession(context_id, str(self.db)).get_items())
        return {"session_id": context_id, "items": items}

    def export_memory(self) -> dict[str, Any]:
        return {"case_files": self._cases()}


if __name__ == "__main__":
    serve(RouterAgent())
