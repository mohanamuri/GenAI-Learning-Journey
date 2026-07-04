# Prerequisites — Module 09 Agentic AI

## Runs Locally — No API Keys Needed

All examples run with standard Python.
Examples that use LLMs require Ollama (free, local). See examples marked "Ollama Optional" below.

---

## Install

```bash
pip install openai
```

No additional libraries required — all examples use stdlib (threading, dataclasses, queue).

---

## Ollama Setup (for LLM-enhanced examples)

```bash
# Install Ollama
# macOS:  brew install ollama  OR  download from https://ollama.ai
# Linux:  curl -fsSL https://ollama.ai/install.sh | sh

# Pull model (one-time, ~2 GB)
ollama pull llama3.2:3b

# Start server (auto-starts on macOS)
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

---

## What Each Example Does & How to Run

| Example | What it does | Needs | Run |
|---------|-------------|-------|-----|
| `01_multi_agent_basics.py` | Agent roles, sequential pipeline, single vs multi comparison | stdlib only | `python 01_multi_agent_basics.py` |
| `02_orchestrator_worker.py` | OrchestratorAgent decomposes task, WorkerAgents execute | Ollama (mock fallback) | `python 02_orchestrator_worker.py` |
| `03_agent_communication.py` | Direct call, message bus (pub/sub), shared state patterns | stdlib only | `python 03_agent_communication.py` |
| `04_parallel_agents.py` | ThreadPoolExecutor, fan-out/fan-in, DAG with deps | stdlib only | `python 04_parallel_agents.py` |
| `05_agent_handoffs.py` | Hard/soft/escalation handoffs, intent routing | stdlib only | `python 05_agent_handoffs.py` |
| `06_supervisor_pattern.py` | Supervisor assigns, evaluates, sends feedback, iterates | Ollama (mock fallback) | `python 06_supervisor_pattern.py` |
| `07_agent_workflows.py` | Sequential, parallel, conditional, loop workflow engine | stdlib only | `python 07_agent_workflows.py` |
| `08_specialized_agents.py` | Research → Write → Review → Edit pipeline | Ollama (mock fallback) | `python 08_specialized_agents.py` |
| `09_agent_state_machine.py` | IDLE/RUNNING/WAITING/DONE/ERROR states, transition guards | stdlib only | `python 09_agent_state_machine.py` |
| `10_production_multi_agent.py` | Circuit breaker, shared memory TTL, health checks, metrics | stdlib only | `python 10_production_multi_agent.py` |

---

## Suggested Run Order

**Start here — no model download needed:**
```bash
python 01_multi_agent_basics.py
python 03_agent_communication.py
python 04_parallel_agents.py
python 05_agent_handoffs.py
python 07_agent_workflows.py
python 09_agent_state_machine.py
python 10_production_multi_agent.py
```

**After `ollama pull llama3.2:3b` + `ollama serve`:**
```bash
python 02_orchestrator_worker.py
python 06_supervisor_pattern.py
python 08_specialized_agents.py
```

> All Ollama-dependent examples fall back to mock content — you can run them and see the structure without the LLM.

**Full project (research pipeline):**
```bash
cd project/research_pipeline/
python agent.py                                 # interactive mode
python agent.py --topic "TechCorp pricing"      # one-shot
python agent.py --topic "TechCorp security"
```

---

## Model Downloads (One-Time)

| Example | Model | Size | Cached at |
|---------|-------|------|-----------|
| `02`, `06`, `08`, project | `llama3.2:3b` (Ollama) | ~2 GB | `~/.ollama/models` |

Total first-run download: **~2 GB** (Ollama model, only if not already pulled from Module 08).

---

## Check Cache

```bash
du -sh ~/.ollama/models   # Ollama models
```

---

## No API Keys — No Cost

Everything runs locally. Ollama serves llama3.2:3b on your CPU/GPU.
If you completed Module 08, the model is already downloaded — no extra download needed.
