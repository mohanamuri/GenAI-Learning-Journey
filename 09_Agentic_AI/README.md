# Module 09 — Agentic AI

Multiple specialized agents coordinating — divide by role, run in parallel, supervise each other.

---

## What We Built

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | `01_multi_agent_basics.py` | Role-based sequential pipeline, single vs multi-agent comparison |
| 02 | `02_orchestrator_worker.py` | Orchestrator decomposes task → dispatches to typed workers → aggregates |
| 03 | `03_agent_communication.py` | Direct call, message bus (pub/sub), shared state (blackboard) |
| 04 | `04_parallel_agents.py` | ThreadPoolExecutor, fan-out/fan-in, DAG with dependencies |
| 05 | `05_agent_handoffs.py` | Hard/soft/escalation handoffs, intent-based routing |
| 06 | `06_supervisor_pattern.py` | Supervisor assigns → evaluates quality → sends feedback → iterates |
| 07 | `07_agent_workflows.py` | Sequential, parallel, conditional, loop workflow engine (DAG) |
| 08 | `08_specialized_agents.py` | Research → Write → Review → Edit pipeline with typed data contracts |
| 09 | `09_agent_state_machine.py` | IDLE/RUNNING/WAITING_FOR_TOOL/DONE/ERROR with transition guards |
| 10 | `10_production_multi_agent.py` | Circuit breaker, TTL cache, health checks, metrics dashboard |
| — | `project/research_pipeline/` | 4-agent pipeline: Research → FactCheck → Summarize → Write |

---

## Key Concepts to Remember

- **Multi-agent ≠ calling LLM twice**: each agent has a distinct role, scoped context, and clear I/O contract
- **Orchestrator rule**: orchestrators plan and route — they never execute tasks directly
- **Worker rule**: workers execute one task type — they never plan or route
- **Message bus**: loose coupling — swap any agent without touching others; prefer over direct call in production
- **Parallel safe when**: agents have no shared data dependency — fan-out then fan-in
- **DAG speedup**: independent nodes run in parallel; dependent nodes wait via `threading.Event`
- **Hard handoff**: full context transferred, original agent stops — use for domain specialists
- **Soft handoff**: sub-agent spawned inline, result folded back in, primary continues
- **Supervisor max_rounds**: always cap revision rounds — prevents infinite feedback loops
- **Specific feedback beats generic**: "missing prices and numbers" > "try again"
- **State machine**: validates transitions — prevents agents from acting in wrong state
- **Circuit breaker**: CLOSED → OPEN (N failures) → HALF_OPEN (probe) → CLOSED — fail fast
- **TTL cache**: same query within TTL → return cached result, skip agent call entirely
- **Health check before dispatch**: skip OPEN-circuit agents in fan-out, don't wait for timeouts

---

## What NOT to Do

- Don't add agents for every step — coordination overhead > benefit for simple linear tasks
- Don't share mutable state between threads without a lock — race conditions are silent bugs
- Don't hardcode handoff targets — use a registry so routing is configurable
- Don't skip handoff depth limit — cyclic handoffs loop forever
- Don't use one shared context for all agents — each agent should get only what it needs
- Don't ignore circuit breaker state — open circuit means service is degraded, alert on it
- Don't set TTL too long — stale cached results can silently serve wrong data

---

## Quick Start

```bash
pip install openai   # only for Ollama-enhanced examples

# No Ollama needed for most examples
python 01_multi_agent_basics.py
python 03_agent_communication.py
python 04_parallel_agents.py
python 05_agent_handoffs.py
python 07_agent_workflows.py
python 09_agent_state_machine.py
python 10_production_multi_agent.py

# After ollama pull llama3.2:3b && ollama serve
python 02_orchestrator_worker.py
python 06_supervisor_pattern.py
python 08_specialized_agents.py

cd project/research_pipeline/
python agent.py
python agent.py --topic "TechCorp pricing"
```

---

## Frameworks

| Framework | Purpose |
|-----------|---------|
| CrewAI | Role-based multi-agent pipelines — see `frameworks/crewai_card.md` |
| AutoGen | Conversational multi-agent with code execution — see `frameworks/autogen_card.md` |
| `openai` (Ollama) | LLM calls for Orchestrator/Worker and Supervisor examples |
