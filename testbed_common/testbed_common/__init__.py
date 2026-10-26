"""Stdlib-only glue shared by every agent service in the governance testbed.

Modules:
    config   - ports, data/log directories, LLM gateway URL.
    obs      - one JSON line per event, per agent, correlated by trace id.
    service  - the A2A-lite HTTP wrapper every agent exposes (card, tasks, state, memory).
    a2a      - client used by an agent to call another agent over that contract.
    mockllm  - deterministic OpenAI-compatible model so every stack runs offline.
"""
