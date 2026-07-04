# Prerequisites — Module 06 LLM Engineering

## Most Examples Run on Ollama — No API Key Needed

Examples 03–11 default to **Ollama (local, free)**. Only `01_openai_basics.py` and
`02_anthropic_basics.py` require cloud API keys (they specifically teach those SDKs).

---

## Install

```bash
pip install openai anthropic tiktoken pydantic
```

---

## Ollama Setup (Free — Start Here)

```bash
# 1. Install Ollama
brew install ollama              # macOS
# Windows/Linux: https://ollama.com/download

# 2. Pull models (one-time download)
ollama pull llama3.2:3b          # ~2 GB — used by most examples
ollama pull llama3.1:8b          # ~5 GB — needed for function calling (example 07)
ollama pull nomic-embed-text     # ~275 MB — for embeddings (example 11)

# 3. Verify server is running
curl http://localhost:11434/api/tags
```

Ollama server starts automatically after install. If not running: `ollama serve`

---

## What Each Example Needs

| Example | Runs On | Requires | Notes |
|---------|---------|----------|-------|
| `01_openai_basics.py` | ☁️ Cloud | `OPENAI_API_KEY` | Teaches OpenAI SDK specifically |
| `02_anthropic_basics.py` | ☁️ Cloud | `ANTHROPIC_API_KEY` | Teaches Anthropic SDK specifically |
| `03_system_prompts.py` | 💻 Local | Ollama + `llama3.2:3b` | Free |
| `04_prompt_templates.py` | 💻 Local | Ollama + `llama3.2:3b` | Free |
| `05_streaming.py` | 💻 Local | Ollama + `llama3.2:3b` | Free, streaming works locally |
| `06_json_mode.py` | ☁️ Cloud | `OPENAI_API_KEY` | JSON mode is OpenAI-specific feature |
| `07_function_calling.py` | 💻 Local | Ollama + `llama3.1:8b` | Needs tool-support model |
| `08_chat_history.py` | 💻 Local | Ollama + `llama3.2:3b` | Free |
| `09_cost_optimization.py` | ☁️ Cloud | `OPENAI_API_KEY` | Needs real pricing data to demonstrate |
| `10_error_handling.py` | 💻 Local | Ollama + `llama3.2:3b` | Free |
| `11_ollama_local_llm.py` | 💻 Local | Ollama | Dedicated Ollama example |

**7 of 11 examples run completely free on Ollama.**

---

## API Keys (Only for Examples 01, 02, 06, 09)

### Option A: Export in terminal (session only)
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Option B: .env file (recommended)
```bash
pip install python-dotenv
```
Create `.env` in project root:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```
Load in script:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Never commit `.env` to git.** Add to `.gitignore`:
```
.env
*.env
```

---

## Suggested Run Order

**Start here — completely free:**
```bash
python 03_system_prompts.py
python 04_prompt_templates.py
python 05_streaming.py
python 08_chat_history.py
python 10_error_handling.py
python 11_ollama_local_llm.py
```

**Needs llama3.1:8b (larger model):**
```bash
python 07_function_calling.py
```

**Needs API keys:**
```bash
python 01_openai_basics.py      # OPENAI_API_KEY
python 02_anthropic_basics.py   # ANTHROPIC_API_KEY
python 06_json_mode.py          # OPENAI_API_KEY
python 09_cost_optimization.py  # OPENAI_API_KEY
```

---

## Cost Estimate (Cloud examples only)

| Example | Approx cost |
|---------|-------------|
| `01_openai_basics.py` | ~$0.001 |
| `02_anthropic_basics.py` | ~$0.001 |
| `06_json_mode.py` | ~$0.001 |
| `09_cost_optimization.py` | ~$0.001 |
| **Total cloud cost** | **< $0.005** |

---

## Sanity Check

```bash
# Ollama (for most examples)
curl http://localhost:11434/api/tags
python -c "from openai import OpenAI; c = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama'); print('Ollama OK')"

# OpenAI (for examples 01, 06, 09)
python -c "from openai import OpenAI; c = OpenAI(); print(c.models.list().data[0].id)"

# Anthropic (for example 02)
python -c "from anthropic import Anthropic; c = Anthropic(); print('Anthropic OK')"
```
