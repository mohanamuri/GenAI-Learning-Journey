# Agentic AI Interview Questions & Answers

## Core Concepts

**Q: What is agentic AI vs a single AI agent?**
A: Single agent: one LLM + tool loop handling all tasks.
Agentic AI: multiple specialized agents coordinating to complete complex tasks.
Key differences: role specialization, agent-to-agent communication, parallel execution.
Single agents hit context limits and lack specialization; agentic systems distribute the cognitive load.

**Q: When should you use multi-agent instead of single-agent?**
A: Multi-agent when: the task has distinct specialization needs (research ≠ writing ≠ reviewing), tasks can be parallelized, or a single context window is too small for the full task.
Single-agent when: steps are sequential and simple, latency is critical, overhead of coordination outweighs benefits.
Rule: start single-agent, add agents only when you hit concrete limitations.

**Q: What is the Orchestrator/Worker pattern?**
A: Orchestrator: receives high-level goal, decomposes into subtasks, dispatches to workers, aggregates results.
Worker: receives one typed subtask, executes it, returns result.
Orchestrator never executes directly; workers never plan.
Used in: LangGraph, AutoGen supervisor mode, CrewAI hierarchical process.

---

## Communication and Coordination

**Q: What are the three agent communication patterns?**
A: Direct call: agent_a.call(agent_b) — tight coupling, simple pipelines.
Message bus: publish/subscribe via shared queue — loose coupling, event-driven.
Shared state (blackboard): agents read/write a shared dict — simple same-process coordination.
Production default: message bus or shared state. Direct call for prototypes only.

**Q: What is a handoff? What are the three types?**
A: Handoff: agent detects it can't handle a task and transfers to a specialist.
Hard handoff: full context transferred, original agent stops (e.g., general → billing).
Soft handoff: sub-agent spawned inline, result folded back in, primary continues.
Escalation: task transferred to a higher-authority agent (e.g., human, manager).
Always log handoff traces for debugging and audit.

**Q: How do you prevent handoff loops?**
A: Cap handoff depth (max 3). Track visited agents per request — reject if same agent would be visited twice.
Add reason validation: only allow handoff if the reason maps to a known specialist.

---

## Architecture

**Q: How does parallel agent execution work and when is it safe?**
A: Use ThreadPoolExecutor for I/O-bound agents (LLM calls, DB, APIs).
Safe when agents are independent — no agent needs the output of another to start.
Pattern: fan-out (dispatch N agents) → fan-in (collect results) → merge.
Speedup = sequential_time / max(individual_times).
Unsafe when: agents share mutable state without locks, or task B depends on task A's result.

**Q: What is a DAG workflow and how does it enable mixed parallel/sequential execution?**
A: DAG (Directed Acyclic Graph): nodes are tasks, edges are dependencies.
Independent nodes (no shared dependencies) run in parallel.
Dependent nodes wait for all their upstream nodes to complete.
Implementation: threading.Event per node — each node waits for its deps' events before starting.

**Q: What is the Supervisor pattern?**
A: SupervisorAgent assigns tasks, evaluates output quality, sends feedback for revision if below threshold.
Loop: assign → worker generates → evaluate → (approve OR revise, up to max_rounds).
If max_rounds exceeded: escalate to human or return best effort.
Key: feedback must be specific (list exact issues), not generic "try again".

**Q: What is the difference between CrewAI and AutoGen?**
A: CrewAI: role-based, task-pipeline model. Agent = role + goal + backstory. Tasks flow A → B → C.
AutoGen: conversation-based. Agents take turns in a chat loop. Better for back-and-forth critique and code execution.
Use CrewAI for structured pipelines; AutoGen for iterative debates or human-in-loop flows.

---

## Production

**Q: What is a circuit breaker in a multi-agent system?**
A: Wraps agent calls: after N failures → trips OPEN (blocks calls, returns fallback).
After recovery_timeout → HALF_OPEN (allows one probe call).
If probe succeeds → back to CLOSED (normal). If fails → stays OPEN.
Without circuit breakers: one flaky agent causes entire system to hang on timeouts.

**Q: How do you cache agent results across calls?**
A: Shared memory with TTL: key = "AgentName:query", value = result, expires after N seconds.
On cache hit: return stored result without calling the agent.
On cache miss: call agent, store result with TTL.
TTL prevents stale data: set TTL based on how frequently the underlying data changes.

**Q: How do you collect metrics for a multi-agent system?**
A: Per agent: call count, success count, total latency, error count.
Derived: success_rate = successes/calls, avg_latency = total_latency/calls.
Dashboard: print metrics after every request batch.
Alert on: success_rate < 95%, avg_latency > 2s, circuit breaker OPEN.

**Q: What is graceful degradation in a multi-agent system?**
A: If an agent is unavailable (circuit open, health check fails): return a fallback string instead of crashing.
For fan-out: skip unhealthy agents, return results from healthy ones.
For critical agents: fallback to a cached response or a simpler rule-based answer.
Never propagate an exception to the user — catch at agent boundary.

---

## State and Memory

**Q: What is an agent state machine and why use it?**
A: Explicit states: IDLE → RUNNING → WAITING_FOR_TOOL → DONE | ERROR.
Transitions are validated — prevents agents from doing work in wrong state (e.g., tool call before RUNNING).
Benefits: auditable lifecycle logs, pausable/resumable agents, clear error recovery.
WAITING_FOR_TOOL state = explicit acknowledgment that agent is blocked on external call.

**Q: What is shared state in multi-agent systems? What are the risks?**
A: Shared state: a dict readable/writable by all agents in the system.
Benefits: zero-coupling agent communication, simple to implement.
Risks: key collision (two agents write same key), stale data (no TTL), race conditions (no lock).
Fixes: namespace keys by agent name, add TTL, use a threading.Lock for all reads/writes.

---

## System Design

**Q: Design a customer onboarding multi-agent system.**
A: Agents:
  - IntakeAgent: validates user info, routes to department-specific agents
  - AccountAgent: creates account in the system
  - PricingAgent: retrieves recommended plan based on company size
  - NotificationAgent: sends welcome email + onboarding links
  - AuditAgent: logs all actions for compliance

Coordination: sequential (intake → account → pricing) + parallel (notification || audit).
Memory: shared state with TTL for user context, entity memory for name/company.
Guardrails: PII detection before storing, rate limit per IP.
Metrics: onboarding completion rate, avg steps, error rate per agent.

**Q: How would you test a multi-agent system?**
A: Unit test each agent in isolation with mock inputs.
Integration test the pipeline: run full task, check each agent's output type and content.
Fault injection: inject failures into individual agents, verify circuit breaker trips and fallback fires.
Load test: run N concurrent requests, verify thread safety and metrics accuracy.
Eval set: for each query, define expected_agents (which agents ran), expected_output (key content).
Metrics: task success rate, agent precision (correct agents called / total called), step efficiency.
