# Prerequisites — Module 08 AI Agents

## Runs Locally — Ollama for Free LLM

Core examples (01-06, 09) run with standard Python + sentence-transformers.
Examples 02-03, 05-08, 10 and the project use Ollama for local LLM — free, no API key.

---

## Install

```bash
pip install openai chromadb sentence-transformers
```

Verify:
```bash
python -c "import openai, chromadb, sentence_transformers; print('All good')"
```

---

## Ollama Setup (for most examples)

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
| `01_what_is_an_agent.py` | Agent loop: observe → think → act, anatomy, pseudocode | stdlib only | `python 01_what_is_an_agent.py` |
| `02_react_pattern.py` | Thought → Action → Observation loop, prompt parsing | Ollama (mock fallback) | `python 02_react_pattern.py` |
| `03_tool_calling.py` | OpenAI tool format, two-call pattern, tool definitions | Ollama (mock fallback) | `python 03_tool_calling.py` |
| `04_agent_memory.py` | Buffer, summary, entity, vector memory systems | sentence-transformers | `python 04_agent_memory.py` |
| `05_planning.py` | Plan-and-execute: planner + executor agents | Ollama (mock fallback) | `python 05_planning.py` |
| `06_code_execution_agent.py` | Generate Python → sandbox execute → interpret | Ollama (mock fallback) | `python 06_code_execution_agent.py` |
| `07_multi_tool_agent.py` | Dynamic tool selection from a rich tool registry | Ollama (mock fallback) | `python 07_multi_tool_agent.py` |
| `08_agent_with_rag.py` | RAG as a tool inside an agent loop (Agentic RAG) | chromadb, sentence-transformers, Ollama | `python 08_agent_with_rag.py` |
| `09_agent_evaluation.py` | Tool precision/recall, step efficiency, answer quality | stdlib only | `python 09_agent_evaluation.py` |
| `10_production_agent.py` | Retries, guardrails, structured logging, run traces | Ollama (mock fallback) | `python 10_production_agent.py` |

---

## Suggested Run Order

**Start here — no model download needed:**
```bash
python 01_what_is_an_agent.py
python 09_agent_evaluation.py
```

**After `pip install sentence-transformers` (~80 MB download once):**
```bash
python 04_agent_memory.py
```

**After `ollama pull llama3.2:3b` (~2 GB, one-time) + `ollama serve`:**
```bash
python 02_react_pattern.py
python 03_tool_calling.py
python 05_planning.py
python 06_code_execution_agent.py
python 07_multi_tool_agent.py
python 08_agent_with_rag.py
python 10_production_agent.py
```

> All examples fall back to a mock trace if Ollama is not running — you can run them and see the structure without the LLM.

**Full project (personal assistant):**
```bash
cd project/personal_assistant/
python agent.py                          # interactive
python agent.py --task "What plans does TechCorp offer?"  # one-shot
```

---

## Model Downloads (Automatic on First Run)

| Example | Model | Size | Cached at |
|---------|-------|------|-----------|
| `04_agent_memory.py` | `all-MiniLM-L6-v2` | ~80 MB | `~/.cache/huggingface` |
| `08_agent_with_rag.py`, project | `all-MiniLM-L6-v2` | shared with above | |
| `02`, `03`, `05`–`10`, project | `llama3.2:3b` (Ollama) | ~2 GB | `~/.ollama/models` |

Total first-run download: **~2.1 GB** (mostly the Ollama model).

---

## Check Cache

```bash
du -sh ~/.cache/huggingface   # HuggingFace models
du -sh ~/.ollama/models        # Ollama models
```

---

## No API Keys — No Cost

Everything runs locally. Ollama serves llama3.2:3b on your CPU/GPU.
Module 07 RAG knowledge base is reused in example 08 and the project.
