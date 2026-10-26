"""Customer notifier - Microsoft AutoGen (agentchat 0.7).

Memory it owns:
    session   the agent's own ``save_state()`` dict (its chat model-context), written to one JSON
              file per context_id and re-loaded into a fresh agent on the next task.
    long-term an AutoGen ``ListMemory`` (a Memory-protocol object injected into the prompt on every
              run), persisted to JSON. Here: a contact log of who was e-mailed about what.
The "e-mail" tool only appends to an outbox file - nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient

from testbed_common.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, agent_data_dir
from testbed_common.obs import log_event
from testbed_common.service import AgentApp, Task, serve

SYSTEM = "[persona:notify] You write to customers. Use send_email to deliver the message."


class NotifyAgent(AgentApp):
    name = "notify"
    stack = "AutoGen"
    skills = ["e-mail the customer the outcome of their refund request"]

    def __init__(self) -> None:
        super().__init__()
        root = agent_data_dir(self.name)
        self.state_dir = root / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.contacts_file = root / "contact_log.json"
        self.outbox = root / "outbox.jsonl"
        self.memory_map = [
            {"tier": "session", "backend": "AutoGen agent.save_state() JSON", "location": str(self.state_dir),
             "export": "/state/<context_id>"},
            {"tier": "long-term", "backend": "AutoGen ListMemory persisted to JSON", "location": str(self.contacts_file),
             "export": "/memory"},
        ]

    def _contacts(self) -> list[str]:
        return json.loads(self.contacts_file.read_text("utf-8")) if self.contacts_file.exists() else []

    def send_email(self, to: str, subject: str, body: str) -> str:
        """Deliver an e-mail to a customer (appends to the outbox file)."""
        record = {"ts": time.time(), "to": to, "subject": subject, "body": body}
        with open(self.outbox, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return self.tools.record("send_email", {"to": to, "subject": subject, "body": body}, f"email queued for {to}")

    async def _run(self, task: Task) -> str:
        memory = ListMemory(name="contact_log")
        for entry in self._contacts():
            await memory.add(MemoryContent(content=entry, mime_type=MemoryMimeType.TEXT))
        agent = AssistantAgent(
            name="notifier",
            model_client=OpenAIChatCompletionClient(
                model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, temperature=0,
                # ListMemory injects its contents as an extra system message after the history.
                model_info={"vision": False, "function_calling": True, "json_output": False,
                            "family": "unknown", "structured_output": False, "multiple_system_messages": True}),
            tools=[self.send_email],
            system_message=SYSTEM,
            memory=[memory],
            reflect_on_tool_use=True,
        )
        state_file = self.state_dir / f"{task.context_id}.json"
        if state_file.exists():
            await agent.load_state(json.loads(state_file.read_text("utf-8")))
            log_event(self.name, "session_restored", task.trace_id, context_id=task.context_id)
        result = await agent.run(task=task.text)
        state_file.write_text(json.dumps(await agent.save_state(), default=str), "utf-8")
        for call in self.tools.calls:
            self._append_contact(f"{call['args'].get('to')}: {call['args'].get('subject')}")
        log_event(self.name, "memory_written", task.trace_id, entries=len(self._contacts()))
        return str(result.messages[-1].to_text())

    def _append_contact(self, entry: str) -> None:
        self.contacts_file.write_text(json.dumps([*self._contacts(), entry]), "utf-8")

    def handle(self, task: Task) -> str:
        return asyncio.run(self._run(task))

    def export_state(self, context_id: str) -> dict[str, Any]:
        path: Path = self.state_dir / f"{context_id}.json"
        return json.loads(path.read_text("utf-8")) if path.exists() else {"note": "no such context"}

    def export_memory(self) -> dict[str, Any]:
        outbox = [json.loads(line) for line in self.outbox.read_text("utf-8").splitlines()] if self.outbox.exists() else []
        return {"contact_log": self._contacts(), "outbox": outbox}


if __name__ == "__main__":
    serve(NotifyAgent())
