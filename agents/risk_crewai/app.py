"""Refund risk reviewer - CrewAI.

Memory it owns:
    long-term CrewAI's unified ``Memory`` (LanceDB on disk): one record per reviewed refund, scoped
              ``/customer/<id>``, recalled by similarity on the next review of that customer.
              Used *explicitly* (remember/recall with fixed scopes and a local embedder) so no
              memory step needs an LLM; ``Crew(memory=True)`` would call the model for scope
              inference and fact extraction.
    session   CrewAI has no session store - a crew is stateless per ``kickoff``. The kickoff
              transcript per context_id is therefore kept by this service in JSON. That is exactly the
              kind of state a framework hides from an observer, which is why it is exported.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from testbed_common.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, agent_data_dir

# Must be set before crewai is imported: stop telemetry, keep CrewAI's own files inside data/.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_STORAGE_DIR", str(agent_data_dir("risk") / "crewai_home"))

from crewai import LLM, Agent, Crew, Memory, Process, Task as CrewTask  # noqa: E402
from crewai.tools import tool  # noqa: E402

from testbed_common.domain import ORDERS, risk_score  # noqa: E402
from testbed_common.mockllm import embed  # noqa: E402
from testbed_common.obs import log_event  # noqa: E402
from testbed_common.service import AgentApp, Task, serve  # noqa: E402

BACKSTORY = "[persona:risk] You review refund requests for abuse. Always call assess_risk, then report its decision."


class RiskAgent(AgentApp):
    name = "risk"
    stack = "CrewAI"
    skills = ["score a refund request and decide approve or escalate"]

    def __init__(self) -> None:
        super().__init__()
        root = agent_data_dir(self.name)
        self.memory_dir = root / "crewai_memory"
        self.transcripts = root / "transcripts"
        self.transcripts.mkdir(exist_ok=True)
        self.memory_map = [
            {"tier": "session", "backend": "service-kept kickoff transcript (CrewAI has no session store)",
             "location": str(self.transcripts), "export": "/state/<context_id>"},
            {"tier": "long-term", "backend": "CrewAI Memory (LanceDB), scope /customer/<id>",
             "location": str(self.memory_dir), "export": "/memory"},
        ]
        llm = LLM(model=f"openai/{LLM_MODEL}", base_url=LLM_BASE_URL, api_key=LLM_API_KEY, temperature=0)
        self.memory = Memory(llm=llm, storage=str(self.memory_dir), embedder=lambda texts: [embed(t) for t in texts])
        self.llm = llm

    def _crew(self, description: str) -> Crew:
        tools = self.tools

        @tool("assess_risk")
        def assess_risk(order_id: str, amount: float, customer_id: str) -> str:
            """Score a refund request. Returns JSON with score, factors and decision (approve|escalate)."""
            result = json.dumps(risk_score(order_id, float(amount), customer_id))
            return tools.record("assess_risk", {"order_id": order_id, "amount": amount, "customer_id": customer_id}, result)

        analyst = Agent(role="Refund risk analyst", goal="Decide whether a refund is safe to pay",
                        backstory=BACKSTORY, llm=self.llm, tools=[assess_risk], allow_delegation=False, verbose=False)
        job = CrewTask(description=description, expected_output="One line starting 'Risk review:' with the decision",
                       agent=analyst)
        return Crew(agents=[analyst], tasks=[job], process=Process.sequential, memory=False, verbose=False)

    def handle(self, task: Task) -> str:
        order = next((w for w in task.text.split() if w.startswith("ORD-")), "")
        customer = ORDERS.get(order, {}).get("customer_id", "unknown")
        scope = f"/customer/{customer}"
        history = self.memory.recall(f"refund review {customer}", scope=scope, limit=3, depth="shallow")
        log_event(self.name, "memory_recalled", task.trace_id, scope=scope, hits=len(history))
        prior = " Earlier reviews: " + "; ".join(m.record.content[:100] for m in history) if history else ""
        result = self._crew(task.text + prior).kickoff()
        answer = str(result.raw)
        self.memory.remember(f"{order} {answer[:200]}", scope=scope, categories=["refund_review"],
                             importance=0.6, source=task.user_id)
        self.memory.drain_writes()
        log_event(self.name, "memory_written", task.trace_id, scope=scope)
        path = self.transcripts / f"{task.context_id}.json"
        runs = json.loads(path.read_text("utf-8")) if path.exists() else []
        runs.append({"ts": time.time(), "task_id": task.task_id, "input": task.text, "tool_calls": self.tools.calls,
                     "output": answer})
        path.write_text(json.dumps(runs, default=str, indent=1), "utf-8")
        return answer

    def export_state(self, context_id: str) -> dict[str, Any]:
        path = self.transcripts / f"{context_id}.json"
        return {"context_id": context_id, "runs": json.loads(path.read_text("utf-8")) if path.exists() else []}

    def export_memory(self) -> dict[str, Any]:
        records = self.memory.list_records(limit=500)
        return {"records": [{"scope": r.scope, "categories": r.categories, "content": r.content,
                             "importance": r.importance, "source": r.source} for r in records]}


if __name__ == "__main__":
    serve(RiskAgent())
