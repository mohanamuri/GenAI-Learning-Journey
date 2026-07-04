# AI Agents Interview Questions & Answers

## Core Concepts

**Q: What is an AI agent?**
A: An AI agent = LLM + loop + tools. It perceives a task, reasons (LLM), takes an action (tool call), observes the result, and repeats until the task is done.
Unlike a single LLM call, an agent is stateful and self-directed — it decides what to do next.

**Q: What is the ReAct pattern?**
A: ReAct = Reason + Act. The agent interleaves Thought (reasoning) and Action (tool call) steps, with Observation (tool output) fed back after each action.
Format: Thought → Action → Observation → Thought → ... → Final Answer.
Forces explicit reasoning before acting — reduces wrong tool calls.

**Q: What is the difference between a chain and an agent?**
A: Chain: fixed sequence of LLM calls — steps are predefined.
Agent: dynamic — the LLM decides which step to take next based on the current state.
Use chains when the steps are always the same; use agents when the path depends on intermediate results.

**Q: What is Plan-and-Execute?**
A: Two-phase agent: (1) Planner LLM generates all steps upfront. (2) Executor LLM runs each step.
Better than ReAct for long, structured tasks. Worse for tasks that require re-planning mid-way.

---

## Tools

**Q: How does tool calling work?**
A: LLM is given JSON tool definitions (name + description + parameters schema).
First call: LLM returns `tool_calls` JSON specifying which tool and what args.
You execute the tool, then send the result back as a `tool` role message.
Second call: LLM generates the final answer using the tool result.

**Q: What makes a good tool description?**
A: Specific, unique domain — no overlap with other tools.
Include the use case: "Use for any weather-related questions."
Include an example: "e.g. get_weather('London')".
Bad: "Get information" (too vague). Good: "Search the product KB for pricing and SLA."

**Q: How many tools should an agent have?**
A: Cap at 8-10 tools. LLM accuracy on tool selection degrades significantly above 10 tools.
Group related tools, or use a router agent to dispatch to specialized sub-agents.

---

## Memory

**Q: What are the types of agent memory?**
A: Buffer memory: last N messages verbatim. Cheapest, simplest.
Summary memory: LLM compresses old turns + keeps recent verbatim. Good for long sessions.
Entity memory: extract and track facts (names, preferences, locations) across turns.
Vector memory: embed past interactions, retrieve by semantic similarity. Cross-session recall.

**Q: Why cap buffer memory?**
A: LLMs have context window limits (e.g. 128k tokens for llama3.2). Unbounded history hits the limit, causing silent truncation — the model loses earlier context without warning.
Rule: set a max_turns and use summary memory for conversations beyond that.

**Q: What is the difference between short-term and long-term agent memory?**
A: Short-term: conversation history in the current session (chat messages list). Lost on restart.
Long-term: persisted storage (vector DB, key-value store) that survives across sessions.
Use ChromaDB or SQLite for long-term; use deque(maxlen=N) for short-term.

---

## Architecture

**Q: How do you add RAG to an agent?**
A: Make retrieval a tool: `search_kb(query: str) → str`. Register it alongside other tools.
The agent calls it only when needed — not on every query. This is Agentic RAG.
Benefit: agent retrieves for factual questions, uses calculator for math — not forced RAG on everything.

**Q: What is a code execution agent?**
A: Agent that generates Python code, executes it in a sandbox, reads the output, and feeds it back.
Better than pure LLM for math, data analysis, and string transformations — no hallucinated calculations.
Critical: always sandbox with restricted builtins — never exec() raw LLM output with full permissions.

**Q: How do you prevent an agent from looping infinitely?**
A: Set max_steps = 5-8. Check finish_reason == "stop" after each LLM call.
If max_steps reached: return a graceful fallback message, don't crash.
Also: detect repeated tool calls with same args (loop detection).

---

## Production

**Q: What is a guardrail in the context of agents?**
A: Pre-processing checks before the agent loop runs:
- Block harmful queries (regex or classifier)
- Reject too-long inputs (DoS prevention)
- Validate tool outputs before sending to LLM
- Cap max cost (track token usage per session)

**Q: How do you add retry logic to tool calls?**
A: Wrap tool calls in try/except. On failure: log the error, wait with exponential backoff (0.1s → 0.2s → 0.4s), retry up to 2-3 times.
After max retries: return a clear error string — the agent can tell the user the tool failed.

**Q: What observability should a production agent have?**
A: Log every tool call: tool name, args, result, latency in ms, timestamp.
Track: total steps, retries, errors, total latency per run.
Store full run trace for post-hoc debugging.
Never expose raw stack traces to users — show friendly messages, log internals.

**Q: How would you evaluate an AI agent?**
A: Build a test suite with: task, expected tools (sequence), expected answer content.
Metrics: task success rate, tool precision (correct tools / tools called), tool recall (expected tools found), step efficiency (expected steps / actual steps), answer quality (string matching or LLM judge).
Evaluate retrieval and generation failure modes separately.

---

## System Design

**Q: How would you design a customer support agent?**
A: Tools: search_kb (product docs), search_order(order_id), escalate_to_human(reason), create_ticket.
Memory: entity memory for customer name + order ID, buffer memory for conversation.
Guardrails: block PII extraction attempts, require authentication before order queries.
Evaluation: task resolution rate, escalation rate, avg turns to resolution.

**Q: When should you use agents vs. a simple chain?**
A: Use agents when: the path depends on intermediate results, tools must be selected dynamically, the task requires multi-step reasoning with branching.
Use chains when: steps are always the same, latency matters (agents are slower), the task is well-defined and predictable.
Rule: prefer the simplest solution that works — agents add complexity and latency.
