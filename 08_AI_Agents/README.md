# Module 08 — AI Agents

LLM + loop + tools. The agent decides what to do next — not you.

---

## What We Built

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | `01_what_is_an_agent.py` | Agent anatomy — observe → think → act loop, tool registry |
| 02 | `02_react_pattern.py` | Thought → Action → Observation loop, regex parsing, stop tokens |
| 03 | `03_tool_calling.py` | OpenAI tool format, two-call pattern, 5 tools registered |
| 04 | `04_agent_memory.py` | Buffer, summary, entity, and vector memory systems |
| 05 | `05_planning.py` | Plan-and-Execute: planner generates steps, executor runs each |
| 06 | `06_code_execution_agent.py` | Generate Python → restricted sandbox → interpret output |
| 07 | `07_multi_tool_agent.py` | ToolRegistry class, dynamic tool selection from 6 tools |
| 08 | `08_agent_with_rag.py` | RAG as a tool — Agentic RAG, agent retrieves only when needed |
| 09 | `09_agent_evaluation.py` | Tool precision/recall, step efficiency, answer quality metrics |
| 10 | `10_production_agent.py` | Retries, guardrails, structured logging, full run traces |
| — | `project/personal_assistant/` | Interactive personal assistant with KB, tools, entity memory |

---

## Key Concepts to Remember

- **Agent = LLM + loop + tools**: single LLM call ≠ agent — an agent repeats until task is done
- **Chain vs agent**: chain = fixed steps; agent = LLM decides what step to take next
- **ReAct format**: `Thought: → Action: → Observation:` — forces reasoning before acting
- **Tool calling > ReAct**: structured JSON tool calls, no fragile regex parsing
- **Two-call pattern**: first call picks tool, you execute it, second call synthesizes answer
- **`tool_choice="auto"`**: LLM decides; `"required"` = must call a tool; `{"function": ...}` = force specific tool
- **Cap tools at 8-10**: LLM tool selection accuracy degrades significantly above 10 tools
- **Max steps = 5-8**: prevents infinite loops — always set a hard limit
- **Buffer memory**: `deque(maxlen=N)` — cheapest, simplest, loses old context
- **Entity memory**: regex-extract and track facts (name, location, preferences) per session
- **Agentic RAG**: RAG as one tool among many — agent calls it only when facts are needed
- **Code sandbox**: `eval()` with `{"__builtins__": {}}` — blocks os, subprocess, network
- **Guardrails run before the loop**: check input length, block harmful patterns, validate tool outputs

---

## What NOT to Do

- Don't use agents when a chain is enough — agents add latency and non-determinism
- Don't call tools with `exec()` on raw LLM output — always sandbox with restricted builtins
- Don't let history grow unbounded — set `deque(maxlen=8)` for buffer memory
- Don't skip `max_steps` — one bad query can spin an agent forever
- Don't trust tool argument values from LLM blindly — validate before execution
- Don't use ReAct with LLMs that support tool calling — tool calling is cheaper and more reliable
- Don't expose raw stack traces to users — log internally, show friendly message externally

---

## Quick Start

```bash
pip install openai chromadb sentence-transformers

# Examples 01, 04, 09 work without Ollama
python 01_what_is_an_agent.py
python 04_agent_memory.py
python 09_agent_evaluation.py

# After ollama pull llama3.2:3b && ollama serve
python 02_react_pattern.py
python 03_tool_calling.py
python 08_agent_with_rag.py

cd project/personal_assistant/
python agent.py
python agent.py --task "What are the pricing plans?"
```

---

## Frameworks

| Framework | Purpose |
|-----------|---------|
| `openai` (Ollama-compatible) | Tool calling, chat completions via local Ollama |
| `chromadb` | Persistent vector store for Agentic RAG |
| `sentence-transformers` | Embeddings for vector memory |
