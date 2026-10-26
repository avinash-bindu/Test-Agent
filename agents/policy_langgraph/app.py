"""Policy researcher - LangGraph.

Memory it owns:
    session   LangGraph *checkpointer* (SQLite): a snapshot of the whole graph state at every
              step, keyed by thread_id (= context_id). Lets a conversation resume and be replayed.
    long-term LangGraph *store* (SQLite): namespaced key/value records that outlive any thread.
              Here: the questions this user asked before, recalled into the next prompt.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from testbed_common.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, agent_data_dir
from testbed_common.domain import search_policy as search_policy_docs
from testbed_common.obs import log_event
from testbed_common.service import AgentApp, Task, serve

SYSTEM = "[persona:policy] You are the refund policy researcher. Answer only from search_policy results."
MEMORY_NS = ("policy", "asked_before")


class PolicyAgent(AgentApp):
    name = "policy"
    stack = "LangGraph"
    skills = ["answer refund policy questions from the knowledge base"]

    def __init__(self) -> None:
        super().__init__()
        db = str(agent_data_dir(self.name) / "langgraph.sqlite")
        self.saver = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
        self.store = SqliteStore(sqlite3.connect(db, check_same_thread=False, isolation_level=None))
        self.store.setup()
        self.memory_map = [
            {"tier": "session", "backend": "LangGraph SqliteSaver (checkpointer)", "location": db,
             "export": "/state/<context_id>"},
            {"tier": "long-term", "backend": "LangGraph SqliteStore (namespaced KV)", "location": db,
             "export": "/memory"},
        ]
        self.graph = self._build()

    def _build(self) -> Any:
        agent = self

        @tool
        def search_policy(question: str) -> str:
            """Search the refund policy knowledge base."""
            docs = search_policy_docs(question)
            return agent.tools.record("search_policy", {"question": question}, json.dumps(docs))

        model = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, temperature=0)
        model = model.bind_tools([search_policy])

        def recall(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
            """Pull what this user asked before from the long-term store into the prompt."""
            user = config["configurable"]["user_id"]
            items = store.search(MEMORY_NS + (user,), limit=3)
            log_event(agent.name, "memory_recalled", config["configurable"]["trace_id"], user=user, hits=len(items))
            if not items:
                return {}
            note = "Earlier questions from this user: " + "; ".join(i.value["question"][:80] for i in items)
            return {"messages": [SystemMessage(content=note)]}

        def call_model(state: MessagesState) -> dict:
            return {"messages": [model.invoke([SystemMessage(content=SYSTEM), *state["messages"]])]}

        def remember(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
            """Write this turn's question to the long-term store once the answer is final."""
            user = config["configurable"]["user_id"]
            question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
            store.put(MEMORY_NS + (user,), config["configurable"]["task_id"], {"question": question})
            log_event(agent.name, "memory_written", config["configurable"]["trace_id"], user=user)
            return {}

        graph = StateGraph(MessagesState)
        graph.add_node("recall", recall)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode([search_policy]))
        graph.add_node("remember", remember)
        graph.add_edge(START, "recall")
        graph.add_edge("recall", "agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "remember"})
        graph.add_edge("tools", "agent")
        graph.add_edge("remember", END)
        return graph.compile(checkpointer=self.saver, store=self.store)

    def handle(self, task: Task) -> str:
        config = {"configurable": {"thread_id": task.context_id, "user_id": task.user_id,
                                   "trace_id": task.trace_id, "task_id": task.task_id}}
        result = self.graph.invoke({"messages": [HumanMessage(content=task.text)]}, config)
        return str(result["messages"][-1].content)

    @staticmethod
    def _message(m: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"type": m.type, "content": m.content}
        if isinstance(m, AIMessage) and m.tool_calls:
            out["tool_calls"] = m.tool_calls
        if isinstance(m, ToolMessage):
            out["tool_call_id"] = m.tool_call_id
        return out

    def export_state(self, context_id: str) -> dict[str, Any]:
        snap = self.graph.get_state({"configurable": {"thread_id": context_id}})
        return {"thread_id": context_id, "next": list(snap.next),
                "messages": [self._message(m) for m in snap.values.get("messages", [])],
                "checkpoints": len(list(self.graph.get_state_history({"configurable": {"thread_id": context_id}})))}

    def export_memory(self) -> dict[str, Any]:
        items = self.store.search(("policy",), limit=500)
        return {"namespace": list(MEMORY_NS),
                "items": [{"namespace": list(i.namespace), "key": i.key, "value": i.value} for i in items]}


if __name__ == "__main__":
    serve(PolicyAgent())
