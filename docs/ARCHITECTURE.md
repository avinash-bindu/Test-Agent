# Architecture

Companion to the [README](../README.md). Diagrams: [components](diagrams/components.mmd), [sequence](diagrams/refund-sequence.mmd), [memory model](diagrams/memory-model.mmd).

## Design goals

1. **Heterogeneous on purpose.** Real estates mix frameworks. A governance layer that only works against one SDK's hooks is not governance. So each agent uses a different stack and the only thing they share is an HTTP contract.
2. **State stays inside the agent.** Each framework keeps its session and memory in its own format and location. The testbed does not normalise that; it only adds read-only export endpoints, because *getting at that state* is the problem governance has to solve.
3. **Deterministic.** A scripted model makes each run repeatable, free and offline. Real models can be swapped in by changing one base URL.
4. **One place to watch the model.** All model traffic goes through one gateway, so there is a single interception point as well as the per-agent ones.

## The contract (A2A-lite)

Implemented once in `testbed_common/service.py`; every agent subclasses `AgentApp` and implements `handle`, `export_state`, `export_memory`.

| Route | Purpose |
|---|---|
| `GET /.well-known/agent.json` | name, stack, skills, memory tiers (`tier`, `backend`, `location`, `export`) |
| `GET /health` | liveness |
| `POST /tasks` | `{task_id, context_id, input:{text,data}, caller:{user_id,agent_id}}` returns `{status, output, tool_calls[], trace_id}` |
| `GET /state/<context_id>` | the framework's session or thread state |
| `GET /memory` | the framework's long-term memory |

Headers on every hop: `X-Trace-Id`, `X-User-Id`, `X-Agent-Id`. Requests are serialised per agent by a lock, so the per-task tool log is unambiguous.

Task responses include `tool_calls` with arguments and results, and the router nests the downstream agents' tool calls inside its own, so one reply shows the whole causal chain.

## Memory tiers by stack

| Tier | Meaning | router | policy | risk | refund | notify |
|---|---|---|---|---|---|---|
| Session | one conversation, resumable | `SQLiteSession` items | LangGraph checkpoints per `thread_id` | per-context JSON transcript | ADK session events + state | `save_state()` JSON |
| Long-term | outlives conversations | case file per user | `SqliteStore` namespace `("policy","asked_before",user)` | `Memory` scope `/customer/<id>` | `user:` / `app:` state, ledger | `ListMemory` contact log |

Observations that matter for governance:

- **Three of five frameworks store state in SQLite, but in three different schemas**; the other two use JSON files and LanceDB. There is no common format to scan.
- **CrewAI has no session store at all.** Whatever a crew did in a kickoff is gone unless the host app records it. State can exist outside the framework's own view.
- **ADK scopes state by key prefix** (`user:`, `app:`), so *visibility is encoded in the key name*. A connector must understand that to know who can see a value.
- **AutoGen injects long-term memory into the prompt as an extra system message on every run**, so memory content reaches the model without appearing in the task text.
- **LangGraph recalls memory inside a graph node** (`recall`), so a retrieval step happens that no caller sees.

## Domain and failure behaviour

`testbed_common/domain.py` holds the fake business: four orders, three customers, four policy documents (one poisoned) and `risk_score()`, which sums points (amount over $500: 50; three or more refunds in 90 days: 40; customer under one year: 20) and escalates at 50 or more.

The router **fails closed**: if the risk agent is unreachable it does not refund, logs `delegation_failed`, and still tries to notify. A delegation failure is returned to the model as text and recorded in `tool_calls`, never swallowed.

## Observability

`obs.log_event(service, event, trace_id, **fields)` writes `logs/<service>.jsonl`. Events per agent: `task_received`, `task_finished` / `task_failed`, `memory_recalled`, `memory_written`, `session_restored`, `delegated`, `delegation_failed`, `export_failed`. The gateway logs `completion` (persona, action, tool, offered tools, prompt size) and `planned_tool_not_offered`. Nothing secret is logged; the planted SSN and card number are synthetic.

## Extending

- **New agent**: new folder under `agents/` with `app.py`, `requirements.txt` (optionally `python.txt`); add a port to `config.PORTS`, a line to `run-all.ps1`, a persona branch in `mockllm._next_tool`.
- **Real model**: set `TESTBED_LLM_BASE_URL`, `TESTBED_LLM_API_KEY`, `TESTBED_LLM_MODEL`. The persona markers in system prompts are harmless to a real model. Expect the `expect` values in `scenarios.py` to stop holding; use the tests only against the scripted gateway.
- **New unsafe scenario**: add a marker handled in `mockllm`, a `Scenario` in `scenarios.py`, and its expected ungoverned outcome.
